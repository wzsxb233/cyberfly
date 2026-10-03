"""Bounded, immutable CPU templates for original Code2Wav reference prefill.

Only the reference's *initial* state is cached. User-generated codes and HiFT
history always belong to a fresh original request state. No model import at
module import time, so cache/key tests can run without CUDA initialization.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import threading


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def tensor_identity(value):
    raw = value.detach().cpu().contiguous()
    return {'shape': list(raw.shape), 'dtype': str(raw.dtype),
            'sha256': digest_bytes(raw.numpy().tobytes())}


class ReferenceTemplateCache:
    def __init__(self, backend, *, model_identity, max_entries=2,
                 max_bytes=512 * 1024**2, feature_entries=4):
        import torch
        if type(max_entries) is not int or not 1 <= max_entries <= 4:
            raise ValueError('Reference cache entries must be 1..4')
        if type(max_bytes) is not int or not 1 <= max_bytes <= 2 * 1024**3:
            raise ValueError('Reference CPU cache byte budget must be positive and <=2 GiB')
        if type(feature_entries) is not int or not 1 <= feature_entries <= 8:
            raise ValueError('Prompt feature cache entries must be 1..8')
        if backend.float16 or backend.n_timesteps != 10:
            raise ValueError('Verified reference cache requires original FP32 and 10 Euler steps')
        if any(p.requires_grad for p in backend.parameters()):
            raise ValueError('Only frozen original speech parameters can use reference templates')
        self.torch, self.backend = torch, backend
        self.max_entries, self.max_bytes, self.feature_entries = max_entries, max_bytes, feature_entries
        self._prepare = backend.prepare_prompt
        self._setup = backend.setup_batch
        self._evict = backend.evict_prompt
        self._features = OrderedDict()
        self._templates = OrderedDict()
        self._bytes = 0
        self._lock = threading.RLock()
        self.identity = {**model_identity, 'cache_version': 1,
            'float16': False, 'n_timesteps': 10,
            'noise': tensor_identity(backend.flow.decoder.rand_noise),
            'device': str(backend.speech_window.device),
            'attention_cache_dtype': str(backend._estimator_att_cache_dtype),
            'template_storage': 'CPU; independent deep copies for each original request'}
        self.stats = {'hits': 0, 'misses': 0, 'evictions': 0, 'oversize_uncached': 0,
                      'unregistered_features_uncached': 0, 'rng_changed_uncached': 0}

    def install(self):
        self.backend.prepare_prompt = self.prepare_prompt
        self.backend.setup_batch = self.setup_batch
        self.backend.evict_prompt = self.evict_prompt
        return self

    def prepare_prompt(self, prompt_cache_id, prompt_wav):
        with self._lock:
            reference_sha = file_digest(prompt_wav)
            key = (str(prompt_cache_id), str(prompt_wav), reference_sha)
            if key in self._features:
                self._features.move_to_end(key)
                return self._features[key]['features']
            # Content is part of the original feature-cache key, so replacing
            # a path or reusing a caller's cache ID cannot return stale audio.
            internal_id = 'cyberfly-ref-' + digest_bytes(json.dumps(key).encode())
            features = self._prepare(internal_id, str(prompt_wav))
            if file_digest(prompt_wav) != reference_sha:
                self._evict(internal_id, str(prompt_wav))
                raise ValueError('Voice reference changed during feature extraction')
            info = {'reference_sha256': reference_sha,
                'features': {name: tensor_identity(getattr(features, name))
                    for name in ('speech_tokens', 'speaker_embedding', 'mels')}}
            self._features[key] = {'features': features, 'identity': info,
                'internal_id': internal_id, 'path': str(prompt_wav)}
            while len(self._features) > self.feature_entries:
                _, old = self._features.popitem(last=False)
                self._evict(old['internal_id'], old['path'])
            return features

    def evict_prompt(self, prompt_cache_id, prompt_wav):
        with self._lock:
            for key in list(self._features):
                if key[:2] == (str(prompt_cache_id), str(prompt_wav)):
                    old = self._features.pop(key)
                    self._evict(old['internal_id'], old['path'])
            self._evict(prompt_cache_id, prompt_wav)

    def _rng_state(self):
        device = self.backend.speech_window.device
        if device.type == 'cuda':
            return self.torch.cuda.get_rng_state(device).clone()
        return self.torch.get_rng_state().clone()

    def _copy_rows(self, entry):
        device = self.backend.speech_window.device
        # The template remains private CPU memory; no writable tensor is shared
        # between requests, even when the destination itself is CPU in tests.
        return [entry['state_type'](**{group: {name: tensor.to(device=device).clone()
                for name, tensor in arrays.items()} for group, arrays in row.items()})
                for row in entry['rows']]

    def setup_batch(self, features, batch_size):
        with self._lock, self.torch.inference_mode():
            feature = next((row for row in self._features.values()
                            if row['features'] is features), None)
            if feature is None:
                self.stats['unregistered_features_uncached'] += 1
                return self._setup(features, batch_size)
            if type(batch_size) is not int or batch_size < 1:
                raise ValueError('Invalid original Code2Wav batch size')
            key_info = {**self.identity, **feature['identity'], 'batch_size': batch_size}
            key = digest_bytes(json.dumps(key_info, sort_keys=True, separators=(',', ':')).encode())
            if key in self._templates:
                entry = self._templates[key]
                self._templates.move_to_end(key)
                self.stats['hits'] += 1
                return self._copy_rows(entry)
            self.stats['misses'] += 1
            rng_before = self._rng_state()
            states = self._setup(features, batch_size)
            if not self.torch.equal(rng_before, self._rng_state()):
                self.stats['rng_changed_uncached'] += 1
                return states
            total = sum(t.numel() * t.element_size() for state in states
                        for group in ('flow_cache', 'hift_cache')
                        for t in getattr(state, group).values())
            if total > self.max_bytes:
                self.stats['oversize_uncached'] += 1
                return states
            rows = [{group: {name: tensor.detach().cpu().clone()
                    for name, tensor in getattr(state, group).items()}
                    for group in ('flow_cache', 'hift_cache')} for state in states]
            while self._templates and (len(self._templates) >= self.max_entries or
                                       self._bytes + total > self.max_bytes):
                _, old = self._templates.popitem(last=False)
                self._bytes -= old['bytes']
                self.stats['evictions'] += 1
            self._templates[key] = {'rows': rows, 'bytes': total,
                'state_type': type(states[0]), 'identity': key_info}
            self._bytes += total
            return states

    def snapshot(self):
        with self._lock:
            return {'version': 1, **self.stats, 'template_count': len(self._templates),
                'template_bytes': self._bytes, 'max_bytes': self.max_bytes,
                'feature_count': len(self._features),
                'template_keys': list(self._templates),
                'identity': self.identity}
