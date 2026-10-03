"""CPU protocol tests; the fake backend is not MiniCPM performance evidence."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from dataclasses import dataclass
from pathlib import Path
import tempfile
import unittest
import torch

from vllm_integration.speech_cache import ReferenceTemplateCache


@dataclass
class Features:
    speech_tokens: torch.Tensor
    speaker_embedding: torch.Tensor
    mels: torch.Tensor


@dataclass
class State:
    flow_cache: dict
    hift_cache: dict


class FakeBackend(torch.nn.Module):
    def __init__(self, random_setup=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.speech_window = torch.zeros(1)
        self.flow = type('Flow', (), {'decoder': type('Decoder', (), {'rand_noise': torch.ones(3)})()})()
        self.float16 = False
        self.n_timesteps = 10
        self._estimator_att_cache_dtype = torch.float32
        self._prompt_features = {}
        self.calls = 0
        self.random_setup = random_setup

    def prepare_prompt(self, cache_id, path):
        key = (cache_id, path)
        if key not in self._prompt_features:
            value = float(sum(Path(path).read_bytes()))
            self._prompt_features[key] = Features(torch.tensor([int(value)]),
                torch.tensor([value]), torch.tensor([[value]]))
        return self._prompt_features[key]

    def evict_prompt(self, cache_id, path):
        self._prompt_features.pop((cache_id, path), None)

    def setup_batch(self, features, count):
        self.calls += 1
        offset = torch.rand(1) if self.random_setup else torch.zeros(1)
        return [State({'actual_initial_state': features.mels.clone() + offset},
                      {'speech': torch.zeros(0)}) for _ in range(count)]


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / 'reference.wav'
        self.path.write_bytes(b'reference A: protocol fixture, not audio')
        self.backend = FakeBackend()
        self.cache = ReferenceTemplateCache(self.backend,
            model_identity={'revision': 'protocol-test'}, max_entries=2, max_bytes=8,
            feature_entries=2).install()

    def tearDown(self):
        self.directory.cleanup()

    def prepare(self, path=None):
        return self.backend.prepare_prompt('caller-reuses-id', str(path or self.path))

    def test_request_states_are_independent_and_exact(self):
        f = self.prepare()
        first = self.backend.setup_batch(f, 1)[0]
        expected = first.flow_cache['actual_initial_state'].clone()
        with torch.inference_mode():
            first.flow_cache['actual_initial_state'].fill_(999)
        before = torch.get_rng_state().clone()
        second = self.backend.setup_batch(f, 1)[0]
        self.assertTrue(torch.equal(second.flow_cache['actual_initial_state'], expected))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        with torch.inference_mode():
            second.flow_cache['actual_initial_state'].zero_()
        third = self.backend.setup_batch(f, 1)[0]
        self.assertTrue(torch.equal(third.flow_cache['actual_initial_state'], expected))
        self.assertEqual(self.backend.calls, 1)

    def test_same_id_and_path_with_changed_audio_never_returns_old_state(self):
        f = self.prepare()
        a = self.backend.setup_batch(f, 1)[0].flow_cache['actual_initial_state']
        self.path.write_bytes(b'new actual reference content')
        other = self.prepare()
        b = self.backend.setup_batch(other, 1)[0].flow_cache['actual_initial_state']
        self.assertFalse(torch.equal(a, b))
        self.assertEqual(self.backend.calls, 2)

    def test_cache_id_is_not_the_voice_identity_and_eviction_is_bounded(self):
        for i in range(6):
            self.path.write_bytes(bytes([i + 1]))
            self.backend.setup_batch(self.prepare(), 1)
        info = self.cache.snapshot()
        self.assertLessEqual(info['template_bytes'], 8)
        self.assertLessEqual(info['template_count'], 2)
        self.assertLessEqual(len(self.backend._prompt_features), 2)
        self.assertGreater(info['evictions'], 0)

    def test_dynamic_reference_cleanup_keeps_template_private(self):
        f = self.prepare()
        self.backend.setup_batch(f, 1)
        self.backend.evict_prompt('caller-reuses-id', str(self.path))
        self.assertEqual(len(self.backend._prompt_features), 0)
        f2 = self.prepare()
        self.backend.setup_batch(f2, 1)
        self.assertEqual(self.backend.calls, 1)

    def test_oversize_reference_uses_original_path_without_retaining_template(self):
        self.backend.setup_batch(self.prepare(), 3)
        self.assertEqual(self.cache.snapshot()['template_count'], 0)
        self.assertEqual(self.cache.snapshot()['oversize_uncached'], 1)

    def test_rng_consuming_initialization_is_not_cached(self):
        b = FakeBackend(random_setup=True)
        cache = ReferenceTemplateCache(b, model_identity={}).install()
        f = b.prepare_prompt('voice', str(self.path))
        b.setup_batch(f, 1)
        b.setup_batch(f, 1)
        self.assertEqual(b.calls, 2)
        self.assertEqual(cache.snapshot()['template_count'], 0)
        self.assertEqual(cache.snapshot()['rng_changed_uncached'], 2)


if __name__ == '__main__':
    unittest.main()
