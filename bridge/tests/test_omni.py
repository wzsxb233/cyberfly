"""Mock HTTP transport checks only; no generated fixture is model evidence."""
import base64
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

from bridge.minicpm import BridgeError
from bridge.omni import OmniBridge, OmniBridgeConfig
from vllm_integration.client import OmniClient


MODEL = 'cyberfly-minicpm-o-4.5'


def wav_fixture(rate=24000):
    target = io.BytesIO()
    with wave.open(target, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(bytes(rate // 5 * 2))
    return target.getvalue()


def response(text, *, audio=False, model=MODEL):
    choices = [{'message': {'content': text}, 'finish_reason': 'stop'}]
    if audio:
        choices.append({'message': {'audio': {'data': base64.b64encode(wav_fixture()).decode()}}, 'finish_reason': 'stop'})
    return io.BytesIO(json.dumps({'id': 'mock-http-response', 'model': model, 'choices': choices}).encode())


class OmniBridgeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='cyberfly-omni-http-fixture-')
        self.addCleanup(temporary.cleanup)
        for target in ('bridge.omni.ROOT', 'vllm_integration.client.ROOT'):
            replacement = patch(target, Path(temporary.name))
            replacement.start()
            self.addCleanup(replacement.stop)

    def test_config_explicit_and_unverified(self):
        bridge = OmniBridge(OmniBridgeConfig())
        self.assertEqual(bridge.status()['provider'], 'vllm_omni')
        self.assertFalse(bridge.status()['connected'])
        with self.assertRaises(BridgeError):
            OmniBridge(OmniBridgeConfig(base_url='http://127.0.0.1:18645/v1'))

    def test_plan_roles_and_original_parser_rejection(self):
        bridge = OmniBridge(OmniBridgeConfig())
        good = '{"task":"navigate","goal_mm":[4,1],"say":"前往目标。"}'
        with patch('vllm_integration.client.urllib.request.urlopen', return_value=response(good)) as http:
            result = bridge.plan('前往4,1', {})
            body = json.loads(http.call_args.args[0].data)
            self.assertEqual([m['role'] for m in body['messages']], ['system', 'user'])
            self.assertEqual(body['modalities'], ['text'])
            self.assertFalse(body['chat_template_kwargs']['use_tts_template'])
            self.assertEqual(result.goal_mm, (4., 1.))
            self.assertEqual(result.provider, 'vllm_omni')
        with patch('vllm_integration.client.urllib.request.urlopen', return_value=response('{"task":"navigate","goal_mm":[NaN,1],"say":"x"}')):
            with self.assertRaises(BridgeError): bridge.plan('任务', {})

    def test_audio_actual_bytes_sent_and_native_audio_returned(self):
        bridge = OmniBridge(OmniBridgeConfig())
        incoming = wav_fixture(16000)
        with patch('vllm_integration.client.urllib.request.urlopen', return_value=response('测试转写')) as http:
            result = bridge.transcribe(incoming)
            body = json.loads(http.call_args.args[0].data)
            encoded = body['messages'][0]['content'][0]['input_audio']['data']
            self.assertEqual(base64.b64decode(encoded), incoming)
            self.assertEqual(result['provider'], 'vllm_omni')
        with patch('vllm_integration.client.urllib.request.urlopen', return_value=response('你好。', audio=True)) as http:
            self.assertEqual(bridge.speak('你好'), wav_fixture())
            body = json.loads(http.call_args.args[0].data)
            self.assertEqual(body['modalities'], ['text', 'audio'])
            self.assertTrue(body['chat_template_kwargs']['use_tts_template'])

    def test_no_audio_no_faithful_text_no_substitute(self):
        bridge = OmniBridge(OmniBridgeConfig())
        for reply in (response('你好'), response('完全不同的回复', audio=True)):
            with patch('vllm_integration.client.urllib.request.urlopen', return_value=reply):
                with self.assertRaises(BridgeError): bridge.speak('你好')

    def test_custom_messages_cannot_shift_latent_slots(self):
        client = OmniClient()
        messages = [{'role': 'user', 'content': 'hello'}]
        with patch('vllm_integration.client.urllib.request.urlopen') as http:
            for extras in ({'text': 'x'}, {'image_path': 'x'}, {'audio_path': 'x'},
                           {'condition': {}}, {'additional_information': {}}):
                with self.assertRaises(ValueError): client.generate(messages=messages, **extras)
            http.assert_not_called()

    def test_vqa_explicit_image_role_retained(self):
        client = OmniClient()
        # Synthetic embedded image marker used only to inspect serialization.
        uri = 'data:image/png;base64,' + base64.b64encode(b'\x89PNG\r\n\x1a\nfixture').decode()
        messages = [{'role': 'system', 'content': '回答实际图片问题'}, {'role': 'user', 'content': [
            {'type': 'text', 'text': '图里有什么？'}, {'type': 'image_url', 'image_url': {'url': uri}}]}]
        with patch('vllm_integration.client.urllib.request.urlopen', return_value=response('mock仅用于协议')) as http:
            client.generate(messages=messages)
            self.assertEqual(json.loads(http.call_args.args[0].data)['messages'], messages)


if __name__ == '__main__': unittest.main()
