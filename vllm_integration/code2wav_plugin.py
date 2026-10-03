"""Original MiniCPM Code2Wav with a bounded initial-reference CPU cache.

The superclass owns all actual codec parsing, request epochs, user caches,
original TTS audio generation and output objects. No weight or buffer removed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

from .speech_cache import ReferenceTemplateCache, file_digest

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE = 'CyberFlyMiniCPMO45Code2Wav'


def verify_assets(model_path):
    model_path = Path(model_path).resolve()
    if model_path != (ROOT / 'model_training/models').resolve():
        raise ValueError('Speech extension requires the existing verified original model assets')
    manifest_path = ROOT / 'model_training/native-speech-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    expected = json.loads((ROOT / 'model_training/weights-manifest.json').read_text())['revision']
    if manifest['revision'] != expected:
        raise ValueError('Original speech and language model revisions do not match')
    assets = {}
    for row in manifest['files']:
        if not row['path'].startswith('assets/token2wav/'):
            continue
        path = model_path / row['path']
        actual = file_digest(path)
        if path.stat().st_size != row['size'] or ('lfs' in row and actual != row['lfs']['oid']):
            raise ValueError('Original speech asset failed size/SHA validation: ' + row['path'])
        assets[row['path']] = actual
    if not {'assets/token2wav/flow.pt', 'assets/token2wav/hift.pt'} <= set(assets):
        raise ValueError('Original native speech manifest is incomplete')
    return {'model_revision': expected, 'speech_assets_sha256': assets,
        'speech_manifest_sha256': file_digest(manifest_path),
        'weights_manifest_sha256': file_digest(ROOT / 'model_training/weights-manifest.json')}


def _model_class():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav import MiniCPMO45Code2Wav

    class CyberFlyMiniCPMO45Code2Wav(MiniCPMO45Code2Wav):
        def _build_backend(self):
            if self.backend is not None:
                return
            extra = self._extra_config()
            if self._connector_config != {'codec_chunk_frames': 10, 'codec_left_context_frames': 3}:
                raise ValueError('Verified speech extension requires chunk10 and original left context3')
            if (self._hift_graph_config['enabled'] or not self._cfm_graph_config['enabled'] or
                extra.get('token2wav_float16', False) or extra.get('token2wav_n_timesteps', 10) != 10 or
                extra.get('code2wav_bfloat16_attention_cache', False) or extra.get('token2wav_trt', False) or
                os.environ.get('MINICPMO_TOKEN2WAV_TRT', '') == '1'):
                raise ValueError('Verified extension retains FP32/10Euler/CFM-only original decoder')
            if (not self.vllm_config.model_config.enforce_eager or
                self.vllm_config.scheduler_config.max_num_seqs != 1):
                raise ValueError('First speech-cache deployment requires outer eager and one sequence')
            identity = verify_assets(self.model_path)
            source = ROOT / 'vendor/vllm-omni/vllm_omni/model_executor/models/minicpmo_4_5'
            identity.update(backend_source_sha256=file_digest(source / 'batched_token2wav.py'),
                graph_source_sha256=file_digest(source / 'cuda_graph_wrapper.py'),
                cache_source_sha256=file_digest(Path(__file__).with_name('speech_cache.py')),
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
            import torch
            limit_mib = int(extra.get('code2wav_allocator_limit_mib', 9216))
            if not 1024 <= limit_mib <= 9216:
                raise ValueError('Speech worker allocator limit must be 1..9 GiB')
            device = torch.cuda.current_device()
            torch.cuda.set_per_process_memory_fraction(
                limit_mib * 1024**2 / torch.cuda.get_device_properties(device).total_memory, device)
            super()._build_backend()
            self.requires_grad_(False)
            # S3 is owned by the original adapter rather than registered on
            # this nn.Module. Retain it on its original device, frozen.
            s3 = getattr(getattr(self.backend._token2wav, '_core', None), '_audio_tokenizer', None)
            if s3 is not None and hasattr(s3, 'requires_grad_'):
                s3.requires_grad_(False)
            self.reference_cache = ReferenceTemplateCache(self.backend, model_identity=identity,
                max_entries=int(extra.get('reference_cache_max_entries', 2)),
                max_bytes=int(extra.get('reference_cache_max_bytes', 512 * 1024**2)),
                feature_entries=int(extra.get('reference_feature_cache_entries', 4))).install()
            self._publish_speech_status('loaded_not_yet_real_request')

        def _publish_speech_status(self, status):
            from .latent_contract import write_json
            graph = self.backend._cfm_graph_wrapper
            write_json(ROOT / 'artifacts/vllm_omni_migration/runtime/speech-stage.json', {
                'schema': 1, 'status': status, 'pid': os.getpid(), 'updated_at': time.time(),
                'architecture': ARCHITECTURE, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'original_weights_frozen': not any(p.requires_grad for p in self.parameters()),
                'original_scratch_buffers_retained': True, 'original_s3_retained_on_gpu': True,
                'float16': self.backend.float16, 'n_timesteps': self.backend.n_timesteps,
                'reference_cache': self.reference_cache.snapshot(),
                'cfm_graph_enabled': bool(graph and graph.enabled),
                'cfm_graph_stats': graph.stats_snapshot() if graph else None,
                'actual_audio_request_completed': status == 'actual_audio_request_completed'})

        def on_requests_finished(self, finished_req_ids):
            # Original request cache cleanup still owns epochs and dynamic
            # reference files. Only after it completes, publish small telemetry.
            super().on_requests_finished(finished_req_ids)
            if self.backend is not None:
                self._publish_speech_status('request_finished')

    CyberFlyMiniCPMO45Code2Wav.__qualname__ = ARCHITECTURE
    return CyberFlyMiniCPMO45Code2Wav


def __getattr__(name):
    if name == ARCHITECTURE:
        cls = _model_class()
        globals()[name] = cls
        return cls
    raise AttributeError(name)
