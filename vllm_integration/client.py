"""Bounded local vLLM-Omni client preserving separate text and native audio choices."""
from __future__ import annotations
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request
import uuid
import wave

ROOT = Path(__file__).resolve().parents[1]


class OmniClient:
    def __init__(self, endpoint='http://127.0.0.1:18649', model='cyberfly-minicpm-o-4.5', timeout=180):
        if endpoint not in {'http://127.0.0.1:18649', 'http://localhost:18649'}:
            raise ValueError('Choose the isolated local Omni runtime')
        self.endpoint, self.model, self.timeout = endpoint, model, timeout

    @staticmethod
    def _media(path, *, maximum=16 * 1024 * 1024):
        path = Path(path).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or not 0 < path.stat().st_size <= maximum:
            raise ValueError('Choose a bounded actual media file inside the project')
        raw = path.read_bytes()
        return raw, {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}

    def generate(self, text=None, *, image_path=None, audio_path=None, speak=False,
                 max_tokens=128, additional_information=None, condition=None, output=None,
                 messages=None, stream=False, on_progress=None, request_id=None):
        if request_id is None:
            request_id = uuid.uuid4().hex
        if not isinstance(request_id, str) or not re.fullmatch(r'[a-f0-9]{32}', request_id):
            raise ValueError('Use an explicit 32-character lowercase hex request ID')
        if type(stream) is not bool or (on_progress is not None and (not stream or not callable(on_progress))):
            raise ValueError('Progress requires an explicit stream and a callable consumer')
        if messages is not None and any(value is not None for value in
                                       (text, image_path, audio_path, condition, additional_information)):
            raise ValueError('Explicit role messages cannot be mixed with text/media/latent shortcut parameters')
        if messages is None and (not isinstance(text, str) or not 1 <= len(text) <= 8000):
            raise ValueError('A bounded actual text instruction is required')
        if type(max_tokens) is not int or not 1 <= max_tokens <= 1024:
            raise ValueError('Use 1..1024 generated tokens')
        content, sources = [], {}
        if messages is not None:
            if not isinstance(messages, list) or not 1 <= len(messages) <= 24:
                raise ValueError('Use 1..24 explicit role messages')
            text_chars = 0
            for message in messages:
                if not isinstance(message, dict) or set(message) != {'role', 'content'} or message['role'] not in {'system', 'user', 'assistant'}:
                    raise ValueError('Only explicit system/user/assistant role and content are supported')
                parts = message['content']
                if isinstance(parts, str):
                    text_chars += len(parts)
                elif isinstance(parts, list) and 1 <= len(parts) <= 8 and message['role'] == 'user':
                    for part in parts:
                        if not isinstance(part, dict): raise ValueError('Invalid content part')
                        if part.get('type') == 'text' and set(part) == {'type', 'text'} and isinstance(part['text'], str):
                            text_chars += len(part['text'])
                        elif part.get('type') == 'image_url' and set(part) == {'type', 'image_url'}:
                            image = part['image_url']
                            if not isinstance(image, dict) or set(image) != {'url'} or not isinstance(image['url'], str) or not image['url'].startswith(('data:image/png;base64,', 'data:image/jpeg;base64,', 'data:image/webp;base64,')):
                                raise ValueError('Explicit images must be bounded embedded actual image data')
                            if len(base64.b64decode(image['url'].split(',', 1)[1], validate=True)) > 8 * 1024 * 1024:
                                raise ValueError('Explicit image exceeds 8 MiB')
                        else:
                            raise ValueError('Unsupported explicit content part; use audio_path for audio input')
                else:
                    raise ValueError('Invalid message content')
            serialized_messages = json.dumps(messages, ensure_ascii=False, allow_nan=False).encode()
            if text_chars > 32000 or len(serialized_messages) > 32 * 1024 * 1024:
                raise ValueError('Explicit messages exceed context/media budget')
            # Freeze the exact roles/content sent, without adding another user
            # prefix that would silently move numerical placeholder positions.
            messages = json.loads(serialized_messages)
            sources['messages'] = {'sha256': hashlib.sha256(serialized_messages).hexdigest(),
                                   'roles': [message['role'] for message in messages], 'text_chars': text_chars}
        if condition is not None:
            if additional_information is not None:
                raise ValueError('Pass a numerical condition or explicit metadata, not both')
            from .latent_contract import build_latent_content_prefix
            prefix = build_latent_content_prefix(condition, speak=speak)
            content.append(prefix['content_part'])
            additional_information = prefix['additional_information']
        if image_path:
            raw, sources['image'] = self._media(image_path)
            if raw.startswith(b'\x89PNG\r\n\x1a\n'): mime = 'image/png'
            elif raw.startswith(b'\xff\xd8\xff'): mime = 'image/jpeg'
            else: raise ValueError('Use an actual PNG or JPEG image')
            content.append({'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,' + base64.b64encode(raw).decode()}})
        if audio_path:
            raw, sources['audio'] = self._media(audio_path)
            with wave.open(io.BytesIO(raw)) as wav:
                if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or not 0 < wav.getnframes() <= wav.getframerate() * 30:
                    raise ValueError('Input audio must be an actual <=30 second mono PCM16 WAV')
            content.append({'type': 'input_audio', 'input_audio': {'data': base64.b64encode(raw).decode(), 'format': 'wav'}})
        if messages is None:
            content.append({'type': 'text', 'text': text})
        body = {'model': self.model, 'messages': messages if messages is not None else [{'role': 'user', 'content': content}],
            'modalities': ['text', 'audio'] if speak else ['text'], 'stream': stream,
            'max_tokens': max_tokens, 'temperature': 0, 'seed': 42,
            'chat_template_kwargs': {'enable_thinking': False, 'use_tts_template': bool(speak)},
            'request_id': request_id}
        if additional_information:
            body['additional_information'] = additional_information
        owns_stream_output = False
        if stream and speak:
            output = Path(output or ROOT / 'artifacts/vllm_omni_migration/responses' / request_id).resolve()
            if not output.is_relative_to(ROOT / 'artifacts') or output.exists():
                raise ValueError('Use a fresh project artifact output directory')
            output.mkdir(parents=True)
            owns_stream_output = True
        request = urllib.request.Request(self.endpoint + '/v1/chat/completions',
            data=json.dumps(body, allow_nan=False).encode(), headers={'Content-Type': 'application/json'})
        started = time.monotonic()
        def checked_progress(value):
            if (value.get('request_id') != request_id or
                    value.get('response_id') != 'chatcmpl-' + request_id or
                    value.get('model') != self.model):
                raise ValueError('Stream response identity differs from the submitted request/model')
            if on_progress is not None:
                on_progress(value)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                if stream:
                    from .streaming import consume_stream
                    response = consume_stream(response, request_id=request_id, started=started,
                        speak=speak, audio_directory=output / 'chunks' if owns_stream_output else None,
                        on_progress=checked_progress, timeout=self.timeout)
                else:
                    raw = response.read(40 * 1024 * 1024 + 1)
        except urllib.error.HTTPError as exc:
            raise RuntimeError('vLLM-Omni HTTP failure: ' + exc.read(4000).decode(errors='replace')) from None
        if not stream:
            if len(raw) > 40 * 1024 * 1024: raise ValueError('Oversized Omni response')
            response = json.loads(raw)
        if response.get('error'): raise RuntimeError(str(response['error']))
        if response.get('id') != 'chatcmpl-' + request_id or response.get('model') != self.model:
            raise ValueError('Omni response identity differs from the submitted request/model')
        choices = response.get('choices', [])
        texts = [item.get('message', {}).get('content') for item in choices]
        texts = [item for item in texts if isinstance(item, str) and item.strip()]
        audio = [item.get('message', {}).get('audio') for item in choices if item.get('message', {}).get('audio')]
        if not texts: raise RuntimeError('Omni returned no actual text')
        result = {'schema': 1, 'request_id': request_id, 'response_id': response.get('id'),
            'provider': 'vllm_omni', 'model': response.get('model'), 'text': '\n'.join(dict.fromkeys(texts)),
            'elapsed_s': time.monotonic() - started, 'sources': sources, 'usage': response.get('usage'),
            'finish_reasons': [item.get('finish_reason') for item in choices],
            'requested_audio': speak, 'audio': None, 'native_latent_requested': bool(additional_information),
            'native_latent_verified': False}
        if stream:
            result['streaming'] = response['streaming']
        if speak and not audio: raise RuntimeError('Omni returned no requested native speech; no substitute was generated')
        if len(audio) > 1: raise RuntimeError('Multiple complete audio alternatives require explicit selection')
        if audio:
            raw_audio = base64.b64decode(audio[0]['data'], validate=True)
            with wave.open(io.BytesIO(raw_audio)) as wav:
                channels, width, rate, samples = wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()
                if channels != 1 or width != 2 or rate != 24000 or not 2400 <= samples <= rate * 120:
                    raise ValueError('Expected bounded actual native 24kHz mono PCM16 speech')
            if output is None: output = ROOT / 'artifacts/vllm_omni_migration/responses' / request_id
            output = Path(output).resolve()
            if not output.is_relative_to(ROOT / 'artifacts') or (output.exists() and not owns_stream_output):
                raise ValueError('Use a fresh project artifact output directory')
            output.mkdir(parents=True, exist_ok=owns_stream_output)
            path = output / 'speech.wav'
            path.write_bytes(raw_audio)
            result['audio'] = {'path': str(path), 'sha256': hashlib.sha256(raw_audio).hexdigest(),
                'sample_rate': rate, 'samples': samples, 'seconds': samples / rate,
                'source': 'Actual vLLM-Omni MiniCPM-o Thinker/Talker/Code2Wav response'}
        if output:
            output = Path(output).resolve()
            if not output.is_relative_to(ROOT / 'artifacts'): raise ValueError('Archive in project artifacts')
            output.mkdir(parents=True, exist_ok=True)
            (output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
            # Source content is described by checksums, not duplicated as base64.
            (output / 'response-metadata.json').write_text(json.dumps({k: v for k, v in response.items() if k != 'choices'}, indent=2))
        return result
