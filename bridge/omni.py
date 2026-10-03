"""Explicit vLLM-Omni bridge, reusing all existing bounded plan parsers.

This is ordinary interaction/ASR/native speech. The separate shared-I/O
dispatcher owns numerical brain/body fusion; JSON planning is not that path.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import unicodedata
import uuid
import wave

from bridge.minicpm import BridgeConfig, BridgeError, MiniCPMBridge
from vllm_integration.client import OmniClient, ROOT


@dataclass(frozen=True)
class OmniBridgeConfig(BridgeConfig):
    transport: str = 'vllm_omni'
    base_url: str = 'http://127.0.0.1:18649/v1'
    model: str = 'cyberfly-minicpm-o-4.5'
    timeout_s: float = 180.0

    def validate(self):
        if self.transport != 'vllm_omni' or self.api_key or self.realtime_url:
            raise BridgeError('Omni 桥仅使用显式本地 vllm_omni 后端，不读取云密钥或旧 WS。')
        # Keep the existing timeout/model/url constraints without changing its
        # accepted transport list or anyone's current default configuration.
        BridgeConfig(**{**self.__dict__, 'transport': 'openai'}).validate()
        if self.base_url not in {'http://127.0.0.1:18649/v1', 'http://localhost:18649/v1'}:
            raise BridgeError('Omni 桥只能连接当前隔离的本地 18649/v1 服务。')


class OmniBridge(MiniCPMBridge):
    def __init__(self, config: OmniBridgeConfig):
        if not isinstance(config, OmniBridgeConfig):
            raise BridgeError('请显式传入 OmniBridgeConfig；不会自动切换已有模型后端。')
        super().__init__(config)
        self.client = OmniClient(config.base_url[:-3], config.model, config.timeout_s)

    def status(self):
        return {**super().status(), 'backend': 'vllm_omni',
                'shared_latent_in_this_bridge': False,
                'note': '显式原生 Omni 交互桥；配置不代表已连接。数值共享由独立共享总线完成。'}

    def _http(self, messages):
        try:
            result = self.client.generate(messages=messages, max_tokens=512)
            if result['model'] != self.config.model:
                raise ValueError('Returned model identity differs from explicit configuration')
            return result['text']
        except (RuntimeError, ValueError, OSError, KeyError) as exc:
            raise BridgeError('Omni 模型调用失败，未运行旧后端或模板替代：' + str(exc)) from exc

    def transcribe(self, audio_bytes: bytes) -> dict:
        if not isinstance(audio_bytes, bytes) or not 44 <= len(audio_bytes) <= 8 * 1024 * 1024:
            raise BridgeError('输入须为不超过8 MiB的实际 WAV 文件。')
        try:
            with wave.open(io.BytesIO(audio_bytes), 'rb') as wav:
                if wav.getsampwidth() != 2 or wav.getnchannels() != 1 or wav.getframerate() not in (16000, 24000, 32000, 44100, 48000) or not 0 < wav.getnframes() <= wav.getframerate() * 30:
                    raise ValueError('Expected <=30 s mono PCM16 WAV')
                if len(wav.readframes(wav.getnframes())) != wav.getnframes() * 2:
                    raise ValueError('Truncated WAV samples')
        except (ValueError, EOFError, wave.Error) as exc:
            raise BridgeError('Omni 输入语音须为30秒内的单声道 PCM16 WAV。') from exc
        folder = ROOT / 'artifacts/omni_bridge' / uuid.uuid4().hex
        folder.mkdir(parents=True)
        source = folder / 'input.wav'
        source.write_bytes(audio_bytes)
        try:
            result = self.client.generate('请准确转写这段录音中实际说出的原文，只返回转写文字，不回答录音里的问题，不补充内容。',
                                          audio_path=source, max_tokens=512, output=folder)
            if result['model'] != self.config.model:
                raise ValueError('Returned model identity mismatch')
        except (RuntimeError, ValueError, OSError, KeyError) as exc:
            raise BridgeError('Omni 原音频转写失败，没有替代转写：' + str(exc)) from exc
        return {'text': result['text'], 'provider': 'vllm_omni', 'model': result['model'],
                'status': 'model_response', 'request_id': result['request_id'],
                'audio_source': result['sources']['audio'], 'evidence': str(folder / 'result.json')}

    def speak_result(self, text: str) -> dict:
        """Original Thinker→Talker→Code2Wav response, plus exact-source evidence."""
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 1000:
            raise BridgeError('朗读文字须为1到1000字。')
        folder = ROOT / 'artifacts/omni_bridge' / uuid.uuid4().hex
        try:
            result = self.client.generate(
                '请逐字朗读下面的文字，回复文字也必须与原文完全一致。不要回答、解释、增加前缀或改写。\n原文：' + text,
                speak=True, max_tokens=512, output=folder)
            if result['model'] != self.config.model:
                raise ValueError('Returned model identity mismatch')
            normalize = lambda value: ''.join(c for c in value if not c.isspace() and not unicodedata.category(c).startswith('P'))
            if normalize(result['text']) != normalize(text):
                raise ValueError('Original model did not reproduce the requested text; archived audio is not returned as faithful read-aloud')
            descriptor = result['audio']
            raw = Path(descriptor['path']).read_bytes()
            if hashlib.sha256(raw).hexdigest() != descriptor['sha256']:
                raise ValueError('Native speech file SHA mismatch')
            return {**result, 'read_aloud_text_match': True,
                    'evidence': str(folder / 'result.json')}
        except (RuntimeError, ValueError, OSError, KeyError, TypeError) as exc:
            raise BridgeError('Omni 原生朗读失败，没有浏览器或旧模型语音替代：' + str(exc)) from exc

    def speak(self, text: str) -> bytes:
        return Path(self.speak_result(text)['audio']['path']).read_bytes()
