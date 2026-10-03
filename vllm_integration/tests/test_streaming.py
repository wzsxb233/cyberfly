import base64
import io
import json
import time
import unittest
import wave

from vllm_integration.streaming import consume_stream, LatestProgress


def chunk(modality, content, **extra):
    value = {'id': 'actual-test-stream', 'model': 'test', 'modality': modality,
             'choices': [{'index': 0, 'delta': {'content': content}}], **extra}
    return b'data: ' + json.dumps(value).encode() + b'\n\n'


def wav(frames):
    out = io.BytesIO()
    with wave.open(out, 'wb') as w:
        w.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
        w.writeframes(frames)
    return base64.b64encode(out.getvalue()).decode()


class StreamTests(unittest.TestCase):
    def read(self, raw, **kwargs):
        return consume_stream(io.BytesIO(raw), request_id='test', started=time.monotonic(),
                              speak=kwargs.pop('speak', False), **kwargs)

    def test_original_pcm_order_and_progress_before_completion(self):
        a, b = b'\x01\x00' * 2400, b'\xfe\xff' * 2400
        events = []
        result = self.read(chunk('text', '真实') + chunk('audio', wav(a)) +
                           chunk('text', '声音') + chunk('audio', wav(b)) + b'data: [DONE]\n\n',
                           speak=True, on_progress=events.append)
        audio = result['choices'][1]['message']['audio']['data']
        with wave.open(io.BytesIO(base64.b64decode(audio))) as w:
            self.assertEqual(w.readframes(w.getnframes()), a + b)
        self.assertEqual(events[0]['text'], '真实')
        self.assertEqual(events[0]['audio_samples'], 0)
        self.assertEqual(events[-1]['audio_samples'], 4800)
        self.assertEqual(result['choices'][0]['message']['content'], '真实声音')

    def test_abrupt_eof_rejected_even_with_valid_partial_text(self):
        with self.assertRaisesRegex(RuntimeError, 'completion marker'):
            self.read(chunk('text', '部分'))

    def test_foreign_response_rejected(self):
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            self.read(chunk('text', 'a') + chunk('text', 'b', id='foreign') + b'data: [DONE]\n\n')

    def test_failed_stream_not_promoted(self):
        with self.assertRaisesRegex(RuntimeError, 'failed'):
            self.read(chunk('text', 'partial') + b'data: {"error":"GPU failed"}\n\n')

    def test_unrequested_audio_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unexpected native audio'):
            self.read(chunk('text', 'a') + chunk('audio', wav(b'\x00\x00' * 2400)))

    def test_truncated_wav_rejected(self):
        original = base64.b64decode(wav(b'\x00\x00' * 2400))
        with self.assertRaisesRegex(ValueError, 'Truncated'):
            self.read(chunk('text', 'a') + chunk('audio', base64.b64encode(original[:-2]).decode()), speak=True)

    def test_progress_owns_its_snapshot(self):
        progress = LatestProgress()
        row = {'audio_chunks': [{'sequence': 0}]}
        progress.publish(row)
        row['audio_chunks'].clear()
        got = progress.snapshot()
        got['audio_chunks'].clear()
        self.assertEqual(progress.snapshot()['audio_chunks'], [{'sequence': 0}])


if __name__ == '__main__':
    unittest.main()
