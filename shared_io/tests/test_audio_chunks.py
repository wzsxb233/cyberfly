"""Byte-for-byte checks on archived actual speech PCM; no model calls."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import unittest
import uuid
import wave

from shared_io.audio_chunks import ROOT, split_actual_audio, join_native_audio


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def pcm(path):
    with wave.open(str(path), 'rb') as source:
        return source.readframes(source.getnframes()), source.getparams()


def write_wave(path, payload, rate, *, channels=1, width=2):
    with wave.open(str(path), 'wb') as target:
        target.setnchannels(channels)
        target.setsampwidth(width)
        target.setframerate(rate)
        target.writeframes(payload)
    return path


class AudioChunkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = ROOT / 'artifacts/audio_chunk_checks' / uuid.uuid4().hex[:12]
        cls.output.mkdir(parents=True)
        cls.input = ROOT / 'artifacts/shared_io_modal_ab/980268a5fd7d/native_a_16k_3s.wav'
        cls.native = ROOT / 'artifacts/lab-integration/minicpm-duplex-epoch-1.wav'
        cls.input_pcm, _ = pcm(cls.input)
        cls.native_pcm, _ = pcm(cls.native)
        cls.source_hashes = {str(path): sha(path.read_bytes()) for path in (cls.input, cls.native)}
        cls.results = {}
        # Exact slices of a previously generated native waveform. Session/event
        # metadata below are explicit test fixtures, not another live generation.
        a = write_wave(cls.output / 'native-slice-a.wav', cls.native_pcm[:-2], 24000)
        b = write_wave(cls.output / 'native-slice-b-one-sample.wav', cls.native_pcm[-2:], 24000)
        cls.descriptors = [{
            'schema': 1, 'path': str(path), 'sha256': sha(path.read_bytes()),
            'event_id': 'offline-wave-fixture-' + str(index), 'session_id': 'offline-fixture-session',
            'epoch': 0, 'sequence': index * 2, 'sample_rate': 24000,
            'samples': len(pcm(path)[0]) // 2,
            'source': 'Exact segment of archived actual duplex PCM; test-only session metadata',
            'original_audio_path': str(cls.native), 'original_audio_sha256': cls.source_hashes[str(cls.native)]
        } for index, path in enumerate((a, b))]

    @classmethod
    def tearDownClass(cls):
        after = {path: sha(Path(path).read_bytes()) for path in cls.source_hashes}
        if after != cls.source_hashes:
            raise AssertionError('A test changed its original recorded audio')
        (cls.output / 'evidence.json').write_text(json.dumps({
            'status': 'passed', 'scope': 'Actual PCM byte preservation and boundary/identity validation only',
            'new_model_or_microphone_calls': False, 'fixture_session_metadata': True,
            'source_file_hashes_unchanged': cls.source_hashes, 'results': cls.results
        }, indent=2, ensure_ascii=False) + '\n')
        print('EVIDENCE', cls.output / 'evidence.json')

    def test_exact_seconds_preserve_all_input_bytes(self):
        result = split_actual_audio(self.input, self.output / 'exact-seconds')
        chunks = result['chunks']
        self.assertEqual(result['chunk_count'], 3)
        self.assertEqual(result['total_padded_samples'], 0)
        self.assertEqual(b''.join(pcm(chunk['path'])[0] for chunk in chunks), self.input_pcm)
        self.assertEqual([(c['start_sample'], c['end_sample']) for c in chunks],
                         [(0, 16000), (16000, 32000), (32000, 48000)])
        self.results['exact_seconds'] = result

    def test_only_tail_gets_explicit_zero_padding(self):
        source = write_wave(self.output / 'actual-cropped-16037.wav', self.input_pcm[:16037 * 2], 16000)
        result = split_actual_audio(source, self.output / 'partial-second')
        first, tail = result['chunks']
        self.assertEqual(first['padded_samples'], 0)
        self.assertEqual(tail['original_samples'], 37)
        self.assertEqual(tail['padded_samples'], 15963)
        actual_tail, info = pcm(tail['path'])
        self.assertEqual(info.nframes, 16000)
        self.assertEqual(actual_tail[:74], self.input_pcm[32000:32074])
        self.assertEqual(actual_tail[74:], b'\x00\x00' * 15963)
        recovered = pcm(first['path'])[0] + actual_tail[:74]
        self.assertEqual(recovered, self.input_pcm[:32074])
        self.results['partial_second'] = result

    def test_input_format_duration_and_truncation_are_rejected(self):
        wrong = write_wave(self.output / 'wrong-rate.wav', self.input_pcm, 24000)
        stereo = write_wave(self.output / 'stereo.wav', self.input_pcm, 16000, channels=2)
        too_long = write_wave(self.output / 'repeated-pcm-33s-format-fixture.wav', self.input_pcm * 11, 16000)
        empty = write_wave(self.output / 'empty.wav', b'', 16000)
        truncated = self.output / 'truncated.wav'
        truncated.write_bytes(self.input.read_bytes()[:-2])
        for index, path in enumerate((wrong, stereo, too_long, empty, truncated)):
            destination = self.output / f'invalid-input-{index}'
            with self.subTest(path=path), self.assertRaises(ValueError):
                split_actual_audio(path, destination)
            self.assertFalse(destination.exists())

    def test_native_output_pcm_concatenates_without_silence_or_rounding(self):
        result = join_native_audio(self.descriptors, self.output / 'joined')
        actual, params = pcm(result['path'])
        self.assertEqual(actual, self.native_pcm)
        self.assertEqual(params.framerate, 24000)
        self.assertEqual(result['samples'], len(self.native_pcm) // 2)
        self.assertEqual(result['inserted_silence_samples'], 0)
        self.assertFalse(result['resampled'])
        self.assertEqual(result['sequence_start'], 0)
        self.assertEqual(result['sequence_end'], 2)
        self.assertEqual(result['chunks'][-1]['samples'], 1)
        self.assertEqual(result['pcm_sha256'], sha(self.native_pcm))
        self.results['native_join'] = result

    def test_join_rejects_changed_identity_format_and_turn_order(self):
        invalid = []
        for field, value in [('sha256', 'bad'), ('samples', 123), ('sample_rate', 16000),
                             ('session_id', 'wrong-session'), ('epoch', 2), ('sequence', 0),
                             ('event_id', self.descriptors[0]['event_id'])]:
            records = deepcopy(self.descriptors)
            records[1][field] = value
            invalid.append(records)
        wrong_format = deepcopy(self.descriptors)
        wrong_format[1].update(path=str(self.input), sha256=self.source_hashes[str(self.input)])
        invalid.append(wrong_format)
        invalid += [list(reversed(self.descriptors)), []]
        for index, records in enumerate(invalid):
            destination = self.output / f'invalid-join-{index}'
            with self.subTest(index=index), self.assertRaises(ValueError):
                join_native_audio(records, destination)
            self.assertFalse(destination.exists())

    def test_paths_and_new_directory_requirement(self):
        existing = self.output / 'existing'
        existing.mkdir()
        sentinel = existing / 'keep.txt'
        sentinel.write_text('keep')
        with self.assertRaises(FileExistsError):
            split_actual_audio(self.input, existing)
        with self.assertRaises(FileExistsError):
            join_native_audio(self.descriptors, existing)
        with self.assertRaises(ValueError):
            split_actual_audio(self.input, '/tmp/forbidden-audio-chunk-output')
        with self.assertRaises(ValueError):
            split_actual_audio('/etc/passwd', self.output / 'invalid-source')
        self.assertEqual(sentinel.read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()
