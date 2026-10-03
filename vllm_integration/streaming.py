"""Consume actual Omni SSE text and standalone WAV chunks without changing tokens."""
from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
import time
import wave
from copy import deepcopy
import threading


class LatestProgress:
    """One bounded thread-safe snapshot, independent of the physical tick."""
    def __init__(self):
        self._lock = threading.Lock()
        self._value = None

    def publish(self, value):
        with self._lock:
            self._value = deepcopy(value)

    def snapshot(self):
        with self._lock:
            return deepcopy(self._value)


def consume_stream(response, *, request_id, started, speak, audio_directory=None,
                   on_progress=None, timeout=180):
    maximum = 40 * 1024 * 1024
    total = sequence = 0
    text = ''
    pcm, chunks, finishes = [], [], {}
    first_text = first_audio = None
    identity = None
    metadata = {}
    done = False
    samples = 0
    event_lines = []

    def event(payload):
        nonlocal text, samples, sequence, first_text, first_audio, identity, metadata
        value = json.loads(payload)
        if value.get('error'):
            raise RuntimeError('Omni stream failed: ' + str(value['error']))
        current = value.get('id'), value.get('model')
        if not all(isinstance(item, str) and item for item in current):
            raise ValueError('Missing streamed response identity')
        if identity is not None and current != identity:
            raise ValueError('Omni response identity changed within a stream')
        identity = current
        metadata.update({k: v for k, v in value.items() if k not in {'choices', 'prompt_text', 'prompt_token_ids'}})
        modality = value.get('modality')
        for choice in value.get('choices', []):
            delta = choice.get('delta') or {}
            content = delta.get('content')
            if not content:
                if choice.get('finish_reason'):
                    finishes[modality] = choice['finish_reason']
                continue
            if choice.get('index') != 0 or modality not in {'text', 'audio'} or not isinstance(content, str):
                raise ValueError('Expected one explicitly identified text/audio stream')
            if modality == 'text':
                text += content
                if len(text) > 64000:
                    raise ValueError('Streamed text exceeds output budget')
                if first_text is None:
                    first_text = time.monotonic() - started
            else:
                if not speak:
                    raise ValueError('Unexpected native audio')
                raw = base64.b64decode(content, validate=True)
                with wave.open(io.BytesIO(raw), 'rb') as wav:
                    if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (24000, 1, 2):
                        raise ValueError('Expected actual native 24kHz mono PCM16 WAV chunk')
                    count = wav.getnframes()
                    frames = wav.readframes(count)
                if count <= 0 or len(frames) != count * 2 or samples + count > 24000 * 120:
                    raise ValueError('Truncated or oversized native audio stream')
                samples += count
                pcm.append(frames)
                if first_audio is None:
                    first_audio = time.monotonic() - started
                descriptor = {'sequence': len(chunks), 'sha256': hashlib.sha256(raw).hexdigest(),
                              'samples': count, 'sample_rate': 24000,
                              'elapsed_s': time.monotonic() - started}
                if audio_directory is not None:
                    directory = Path(audio_directory)
                    directory.mkdir(parents=True, exist_ok=True)
                    path = directory / f'chunk-{len(chunks):04d}.wav'
                    with path.open('xb') as stream:
                        stream.write(raw)
                    descriptor['path'] = str(path)
                chunks.append(descriptor)
            if choice.get('finish_reason'):
                finishes[modality] = choice['finish_reason']
            sequence += 1
            if on_progress:
                on_progress({'schema': 1, 'sequence': sequence, 'request_id': request_id,
                    'response_id': identity[0], 'model': identity[1], 'text': text,
                    'progress_clock': time.monotonic(),
                    'elapsed_s': time.monotonic() - started, 'first_text_s': first_text,
                    'first_audio_s': first_audio, 'audio_samples': samples,
                    'audio_chunks': [dict(row) for row in chunks], 'complete': False})

    while True:
        raw = response.readline(maximum - total + 1)
        if not raw:
            break
        total += len(raw)
        if total > maximum or time.monotonic() - started > timeout:
            raise ValueError('Omni stream exceeds byte/time budget')
        line = raw.rstrip(b'\r\n')
        if line.startswith(b'data:'):
            event_lines.append(line[5:].lstrip(b' '))
        elif not line and event_lines:
            payload = b'\n'.join(event_lines)
            event_lines.clear()
            if payload == b'[DONE]':
                done = True
                break
            event(payload)
    if not done:
        raise RuntimeError('Omni stream ended before its completion marker')
    if not text.strip() or identity is None:
        raise RuntimeError('Omni returned no actual streamed text')
    if speak and not pcm:
        raise RuntimeError('Omni returned no requested native speech')
    choices = [{'index': 0, 'message': {'content': text}, 'finish_reason': finishes.get('text')}]
    if pcm:
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(b''.join(pcm))
        choices.append({'index': 1, 'message': {'audio': {'data': base64.b64encode(buffer.getvalue()).decode()}},
                        'finish_reason': finishes.get('audio')})
    return {**metadata, 'id': identity[0], 'model': identity[1], 'choices': choices,
            'streaming': {'first_text_s': first_text, 'first_audio_s': first_audio,
                          'progress_clock': time.monotonic(),
                          'events': sequence, 'audio_chunks': chunks, 'received_bytes': total,
                          'completed_marker': True, 'audio_join': 'Original PCM frames in received order'}}
