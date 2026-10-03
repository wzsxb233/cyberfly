"""Byte-preserving PCM boundaries for real stateful audio input and output.

Input is never resampled: 16 kHz mono PCM16 is cut into one-second blocks,
with zero padding only at its final boundary. Output concatenation preserves
every supplied 24 kHz mono PCM16 payload byte and never invents silence or
speech. Descriptors record provenance; they do not replace model-run evidence.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import wave

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'artifacts'
MAX_INPUT_SECONDS = 30
MAX_INPUT_BYTES = 4 * 1024 * 1024
MAX_JOIN_CHUNKS = 256
MAX_JOIN_PCM_BYTES = 32 * 1024 * 1024


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _artifact(path):
    if not isinstance(path, (str, Path)) or not str(path):
        raise ValueError('Audio paths must name local artifacts')
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(ARTIFACTS.resolve()):
        raise ValueError('Audio files and output directories must stay inside project artifacts')
    return resolved


def _read_pcm(path, rate, *, max_bytes):
    path = _artifact(path)
    if not path.is_file() or not 0 < path.stat().st_size <= max_bytes:
        raise ValueError('Audio must be a bounded existing regular WAV file')
    raw = path.read_bytes()
    if len(raw) > max_bytes:
        raise ValueError('Audio file exceeded its size limit while reading')
    try:
        with wave.open(io.BytesIO(raw), 'rb') as source:
            if (source.getnchannels() != 1 or source.getsampwidth() != 2 or
                    source.getframerate() != rate or source.getcomptype() != 'NONE'):
                raise ValueError(f'Expected exactly {rate} Hz mono uncompressed PCM16; no automatic conversion')
            samples = source.getnframes()
            if samples <= 0:
                raise ValueError('An actual audio chunk must contain at least one sample')
            pcm = source.readframes(samples)
            if len(pcm) != samples * 2:
                raise ValueError('WAV payload is truncated relative to its declared sample count')
    except (wave.Error, EOFError) as exc:
        raise ValueError('Expected a valid uncompressed PCM WAV file') from exc
    return path, pcm, {'path': str(path), 'sha256': _sha(raw), 'pcm_sha256': _sha(pcm),
        'sample_rate': rate, 'channels': 1, 'sample_width_bytes': 2,
        'samples': samples, 'duration_seconds': samples / rate}


def _fresh_directory(outdir):
    path = _artifact(outdir)
    if path.exists():
        raise FileExistsError('Audio output needs a new directory; existing evidence is never overwritten')
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_pcm(path, pcm, rate):
    if len(pcm) % 2:
        raise ValueError('PCM16 payload must contain complete two-byte samples')
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(rate)
        target.writeframes(pcm)
    raw = buffer.getvalue()
    with Path(path).open('xb') as output:
        output.write(raw)
    return {'path': str(path), 'sha256': _sha(raw), 'pcm_sha256': _sha(pcm),
        'sample_rate': rate, 'channels': 1, 'sample_width_bytes': 2,
        'samples': len(pcm) // 2, 'duration_seconds': len(pcm) / (2 * rate)}


def _manifest(path, value):
    with path.open('x', encoding='utf-8') as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write('\n')


def split_actual_audio(path, outdir):
    """Return a manifest with immutable 16,000-sample WAV chunk descriptors.

Source/start/end sample positions are zero-based and end-exclusive. The final
chunk's padded_samples counts explicitly added zero-valued PCM16 samples.
All earlier samples are byte-identical to the supplied source WAV payload.
"""
    _, pcm, source = _read_pcm(path, 16000, max_bytes=MAX_INPUT_BYTES)
    if source['samples'] > MAX_INPUT_SECONDS * 16000:
        raise ValueError('Actual input audio is limited to 30 seconds; crop explicitly before calling')
    directory = _fresh_directory(outdir)
    chunks = []
    for index, start in enumerate(range(0, source['samples'], 16000)):
        end = min(start + 16000, source['samples'])
        original_samples = end - start
        padding = 16000 - original_samples
        actual = pcm[start * 2:end * 2]
        chunk = _write_pcm(directory / f'input-{index:03d}.wav', actual + b'\x00\x00' * padding, 16000)
        chunks.append({'schema': 1, **chunk, 'index': index,
            'start_sample': start, 'end_sample': end, 'original_samples': original_samples,
            'padded_samples': padding, 'source_path': source['path'],
            'source_sha256': source['sha256'], 'source_pcm_sha256': source['pcm_sha256'],
            'original_segment_pcm_sha256': _sha(actual)})
    result = {'schema': 1, 'operation': 'actual_pcm_split', 'source': source,
        'chunk_samples': 16000, 'chunk_count': len(chunks), 'chunks': chunks,
        'resampled': False, 'padding_rule': 'Zero-valued PCM16 samples only at the last chunk boundary',
        'total_padded_samples': sum(chunk['padded_samples'] for chunk in chunks),
        'manifest_path': str(directory / 'split_manifest.json')}
    _manifest(Path(result['manifest_path']), result)
    return result


def _context(descriptor):
    def text(key, alternative=None):
        value = descriptor.get(key, descriptor.get(alternative) if alternative else None)
        if alternative in descriptor and key in descriptor and descriptor[key] != descriptor[alternative]:
            raise ValueError(f'Conflicting native audio {key}/{alternative} values')
        if not isinstance(value, str) or not 1 <= len(value) <= 256:
            raise ValueError(f'Native audio needs its actual nonempty {key}')
        return value
    event = text('event_id')
    session = text('session_id', 'session')
    sequence = descriptor.get('sequence', descriptor.get('seq'))
    if 'sequence' in descriptor and 'seq' in descriptor and descriptor['sequence'] != descriptor['seq']:
        raise ValueError('Conflicting native audio sequence/seq values')
    if type(sequence) is not int or sequence < 0:
        raise ValueError('Native audio needs its actual nonnegative integer sequence')
    epoch = descriptor.get('epoch')
    if epoch is not None and not (type(epoch) is int and epoch >= 0 or isinstance(epoch, str) and 1 <= len(epoch) <= 128):
        raise ValueError('Native audio epoch must be an actual integer or short string')
    return {'event_id': event, 'session_id': session, 'sequence': sequence, 'epoch': epoch}


def join_native_audio(descriptors, outdir):
    """Concatenate actual duplex audio_file descriptors in supplied turn order.

Descriptors require path, SHA256, event_id, session_id and sequence (session/seq
aliases are accepted only when unambiguous). All belong to one session/epoch;
sequence may skip listening turns that emitted no audio, but never decreases.
There is no inserted silence for gaps, mixing, normalization, or resampling.
"""
    if not isinstance(descriptors, (list, tuple)) or not 1 <= len(descriptors) <= MAX_JOIN_CHUNKS:
        raise ValueError(f'Provide 1–{MAX_JOIN_CHUNKS} actual emitted audio descriptors')
    payloads, chunks = [], []
    seen_events, previous, total = set(), None, 0
    for descriptor in descriptors:
        if not isinstance(descriptor, dict) or descriptor.get('schema', 1) != 1:
            raise ValueError('Native audio descriptors must use schema 1')
        context = _context(descriptor)
        if previous is not None and (context['session_id'] != previous['session_id'] or
                context['epoch'] != previous['epoch'] or context['sequence'] <= previous['sequence']):
            raise ValueError('Join only increasing turns from the same native session and epoch')
        if context['event_id'] in seen_events:
            raise ValueError('A native audio event cannot be appended twice')
        _, pcm, actual = _read_pcm(descriptor.get('path'), 24000, max_bytes=MAX_JOIN_PCM_BYTES + 1024 * 1024)
        if actual['sha256'] != descriptor.get('sha256'):
            raise ValueError('Native audio file SHA256 differs from its actual output descriptor')
        for key in ('sample_rate', 'channels', 'sample_width_bytes', 'samples', 'pcm_sha256'):
            if key in descriptor and descriptor[key] != actual[key]:
                raise ValueError('Native audio descriptor differs from its actual WAV: ' + key)
        total += len(pcm)
        if total > MAX_JOIN_PCM_BYTES:
            raise ValueError('Combined native audio exceeds the bounded PCM output size')
        start = sum(len(value) // 2 for value in payloads)
        chunks.append({**actual, **context, 'index': len(chunks),
            'output_start_sample': start, 'output_end_sample': start + actual['samples'],
            'source': descriptor.get('source'), 'input_descriptor': dict(descriptor)})
        payloads.append(pcm)
        previous = context
        seen_events.add(context['event_id'])
    directory = _fresh_directory(outdir)
    joined = _write_pcm(directory / 'native-joined.wav', b''.join(payloads), 24000)
    result = {'schema': 1, **joined, 'operation': 'native_pcm_concatenation',
        'session_id': chunks[0]['session_id'], 'epoch': chunks[0]['epoch'],
        'event_ids': [chunk['event_id'] for chunk in chunks],
        'sequence_start': chunks[0]['sequence'], 'sequence_end': chunks[-1]['sequence'],
        'chunk_count': len(chunks), 'chunks': chunks,
        'source': 'Byte-identical concatenation of the supplied actual native duplex PCM payloads',
        'inserted_silence_samples': 0, 'resampled': False, 'normalized': False,
        'manifest_path': str(directory / 'join_manifest.json')}
    _manifest(Path(result['manifest_path']), result)
    return result
