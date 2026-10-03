"""CPU protocol tests only: injected mock pool/flow never request a model/GPU."""
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid
import wave

from vllm_integration.fleet_controller import FleetController, ROOT, acceptance_status


class MockPool:
    def __init__(self, configs, *, directory):
        self.configs, self.directory = configs, Path(directory)
        self.workers = {c['env_id']: {'env_id': c['env_id'], 'seed': c['seed'], 'step': 0,
            'episode': 1, 'frame_seq': 1, 'simulation_ms': 0, 'phase': 'running', 'alive': True} for c in configs}
        self.selected = configs[0]['env_id']
        self.started = self.closed = False
        self.calls = []
        self.fail_save = None

    def start(self):
        self.directory.mkdir(parents=True)
        self.started = True
        self.calls.append('start')

    def status(self):
        return {'workers': deepcopy(self.workers), 'selected_env_id': self.selected}

    def render_selected(self):
        self.calls.append('render:' + self.selected)
        raw = b'mock JPEG bytes for protocol caching, not a rendered simulator image'
        return {'jpeg_base64': base64.b64encode(raw).decode(), 'sha256': hashlib.sha256(raw).hexdigest()}

    def select(self, env_id):
        self.calls.append('select:' + env_id)
        self.selected = env_id

    def save(self, env_id):
        self.calls.append('save:' + env_id)
        if self.fail_save == env_id:
            raise OSError('injected save failure')
        return {'checkpoint': {'path': str(self.directory / env_id), 'mock': True}}

    def close(self):
        self.calls.append('close')
        self.closed = True


class MockFlow:
    def __init__(self, pool, **kwargs):
        self.pool, self.kwargs = pool, kwargs
        self.entered = threading.Event()
        self.release = threading.Event()
        self.closed = False
        self.ticks = 0

    def tick(self):
        self.entered.set()
        if not self.release.wait(5):
            raise TimeoutError('test did not release simulated model stall')
        time.sleep(.01)
        for worker in self.pool.workers.values():
            worker['step'] += 1
            worker['frame_seq'] += 1
            worker['simulation_ms'] += 28.6
        self.ticks += 1
        return {}

    def status(self):
        return {'pending_model_environments': list(self.pool.workers), 'errors': {}, 'mock': True}

    def close(self):
        self.closed = True


class FleetControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = ROOT / 'artifacts/fleet_controller_tests' / uuid.uuid4().hex[:12]
        cls.output.mkdir(parents=True)
        cls.results = {'scope': 'Mock pool and flow lifecycle/HTTP-facing cache protocol only; no actual brain, model, GPU, or live Lab mutation', 'checks': {}, 'test_outcomes': {}}

    @classmethod
    def tearDownClass(cls):
        cls.results['status'] = 'passed' if all(cls.results['test_outcomes'].values()) else 'failed'
        (cls.output / 'evidence.json').write_text(json.dumps(cls.results, ensure_ascii=False, indent=2))
        print('Evidence:', cls.output)

    def setUp(self):
        self.folder = self.output / self._testMethodName
        self.folder.mkdir()
        self.marker = self.folder / 'mock-acceptance.json'
        self.marker.write_text(json.dumps({'status': 'passed', 'model_bidirectional_verified': True, 'native_speech_verified': True}))
        checkpoint = self.folder / 'mock-checkpoints'
        generation = checkpoint / ('a' * 32)
        generation.mkdir(parents=True)
        (checkpoint / 'latest.json').write_text(json.dumps({'generation': 'a' * 32}))
        (generation / 'adapter.json').write_text(json.dumps({'generation': 'a' * 32, 'memory_sha256': 'mock-not-real-brain'}))
        (generation / 'brain.npz').write_bytes(b'mock checkpoint only; injected MockPool does not load a brain')
        self.pools, self.flows = [], []
        self.provider_calls = 0
        def provider():
            self.provider_calls += 1
            return {'session_id': 'mock-live-lab', 'scenario': {'name': 'fly_neural_link', 'env_kwargs': {'max_episode_steps': 50}},
                    'spec': {'name': 'fly_neural_link', 'neural_input': True, 'embodiment': 'physical_3d'},
                    'brain_learning': False, 'info': {'neural_drive': {'retina_left': .2, 'retina_right': .3, 'sugar': .1},
                        'general_stimulation': [{'macro_group_currents_mv': [1.] * 47}, {'neuron_currents_file': {'do_not_replay': True}}]}}
        def pool(*args, **kwargs):
            value = MockPool(*args, **kwargs); self.pools.append(value); return value
        def flow(*args, **kwargs):
            value = MockFlow(*args, **kwargs); self.flows.append(value); return value
        self.controller = FleetController(provider, pool_factory=pool, flow_factory=flow,
            acceptance_path=self.marker, output_root=self.folder / 'runs', min_free_bytes=0,
            migration_seed={'fleet_seed_brain_checkpoint': str(checkpoint)})

    def tearDown(self):
        result = self._outcome.result
        self.results['test_outcomes'][self._testMethodName] = not any(test.id() == self.id() for test, _ in result.errors + result.failures)
        for flow in self.flows:
            flow.release.set()
        for pool in self.pools:
            pool.fail_save = None
        self.controller.close(timeout=5)

    def wait_for(self, condition, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if condition():
                return
            time.sleep(.01)
        self.fail('Condition did not become true')

    def test_acceptance_fails_closed_before_start(self):
        self.marker.unlink()
        with self.assertRaises(ValueError):
            self.controller.start({'count': 2})
        self.marker.write_text(json.dumps({'status': 'passed', 'model_bidirectional_verified': True, 'native_speech_verified': 'true'}))
        self.assertFalse(acceptance_status(self.marker)['ready'])
        with self.assertRaises(ValueError):
            self.controller.start({'count': 2})
        self.assertEqual(self.pools, [])
        self.assertEqual(self.provider_calls, 0)
        self.results['checks']['fail_closed_before_any_environment_or_lab_access'] = True

    def test_async_cache_selection_and_checkpointed_stop(self):
        started = time.perf_counter()
        reply = self.controller.start({'count': 2})
        launch_ms = (time.perf_counter() - started) * 1000
        self.assertTrue(reply['accepted'])
        self.assertLess(launch_ms, 1000)
        self.wait_for(lambda: bool(self.flows) and self.flows[0].entered.is_set())
        delays = []
        for _ in range(50):
            t = time.perf_counter(); self.controller.status(); self.controller.frame('fly0'); delays.append((time.perf_counter()-t)*1000)
        self.assertLess(max(delays), 50)
        with self.assertRaises(ValueError): self.controller.frame('fly1')
        self.controller.select('fly1')
        self.assertEqual(self.controller.status()['selected_id'], 'fly0')
        self.flows[0].release.set()
        self.wait_for(lambda: self.controller.status()['selected_id'] == 'fly1')
        self.controller.frame('fly1')
        self.controller.select('fly1')
        self.wait_for(lambda: self.controller.status()['pending_selected_id'] is None)
        pool = self.pools[0]
        self.assertIn('select:fly1', pool.calls)
        self.assertFalse(any('reset' in call for call in pool.calls))
        self.assertNotEqual(pool.configs[0]['seed'], pool.configs[1]['seed'])
        kwargs = pool.configs[0]['scenario']['env_kwargs']
        self.assertEqual(len(kwargs['general_stimulation']), 1)
        self.assertEqual(kwargs['semantic_drive']['sugar'], .1)
        self.assertTrue(kwargs['brain_checkpoint'].endswith('a' * 32))
        self.controller.stop(); self.assertTrue(self.controller.stopped.wait(5))
        self.assertFalse(self.controller.status()['enabled'])
        self.assertLess(pool.calls.index('save:fly0'), pool.calls.index('close'))
        self.assertLess(pool.calls.index('save:fly1'), pool.calls.index('close'))
        self.assertEqual(sum(call.startswith('render:') for call in pool.calls), self.flows[0].ticks + 2)
        self.assertEqual(self.provider_calls, 1)
        self.results['checks']['async_start_cached_reads_select_no_reset_save_before_close'] = {'passed': True, 'start_ms': launch_ms, 'cached_read_max_ms': max(delays), 'calls': pool.calls}

    def test_save_failure_retains_owned_state_for_retry(self):
        self.controller.start({'count': 2})
        self.wait_for(lambda: bool(self.flows) and self.flows[0].entered.is_set())
        pool = self.pools[0]; pool.fail_save = 'fly1'
        self.flows[0].release.set(); self.controller.stop()
        self.wait_for(lambda: self.controller.status()['state'] == 'save_failed')
        self.assertFalse(pool.closed)
        self.assertTrue(self.controller.status()['enabled'])
        pool.fail_save = None
        self.controller.stop(); self.assertTrue(self.controller.stopped.wait(5))
        self.assertTrue(pool.closed)
        self.assertEqual(pool.calls.count('save:fly0'), 1)
        self.results['checks']['save_failure_preserves_then_retry_closes'] = True

    def test_low_disk_pauses_owned_loop_only(self):
        self.controller.min_free_bytes = 1024
        with patch('vllm_integration.fleet_controller.shutil.disk_usage', return_value=SimpleNamespace(free=0)):
            self.controller.start({'count': 2})
            self.wait_for(lambda: self.controller.status()['state'] == 'paused_low_disk')
            self.assertEqual(self.flows[0].ticks, 0)
            self.assertFalse(self.pools[0].closed)
            self.controller.stop(); self.assertTrue(self.controller.stopped.wait(5))
        self.results['checks']['low_disk_pauses_without_gpu_operations'] = True

    def test_audio_only_selected_owned_flow_with_verified_sha(self):
        # Synthetic non-speech PCM exercises serving restrictions only.
        flow_dir = ROOT / 'artifacts/vllm_omni_migration/flows' / ('audio-protocol-' + uuid.uuid4().hex[:12])
        flow_dir.mkdir(parents=True)
        path = flow_dir / 'fixture.wav'
        with wave.open(str(path), 'wb') as output:
            output.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
            output.writeframes(b'\0\0' * 24)
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        with self.controller.lock:
            self.controller.cached['selected_id'] = 'fly0'
            self.controller.audio_root = flow_dir
            self.controller.audio_descriptors = {'fly0': {'path': str(path), 'sha256': digest}}
        try:
            self.assertEqual(self.controller.audio('fly0', digest), raw)
            with self.assertRaises(ValueError): self.controller.audio('fly1', digest)
            with self.assertRaises(ValueError): self.controller.audio('fly0', 'f' * 64)
            path.write_bytes(raw + b'tampered')
            with self.assertRaises(ValueError): self.controller.audio('fly0', digest)
            path.write_bytes(raw)
            unrelated = self.folder / 'outside-owned-flow.wav'
            unrelated.write_bytes(raw)
            with self.controller.lock:
                self.controller.audio_descriptors['fly0']['path'] = str(unrelated)
            with self.assertRaises(ValueError): self.controller.audio('fly0', digest)
            self.results['checks']['audio_selected_owned_flow_sha_and_nonempty_pcm_guard'] = True
        finally:
            path.unlink()
            flow_dir.rmdir()

    def test_new_body_never_inherits_legacy_motor(self):
        previous = self.controller.state_provider
        def old_body():
            value = previous()
            value['info']['motor_readout'] = {'checkpoint': 'old-body-checkpoint-must-not-be-read'}
            return value
        self.controller.state_provider = old_body
        result = self.controller._configuration({'scenario': {'name': 'flybody_neural_link', 'env_kwargs': {}}})
        self.assertIsNone(result['motor_checkpoint'])
        self.assertNotIn('motor_readout_checkpoint', result['configs'][0]['scenario']['env_kwargs'])
        with self.assertRaisesRegex(ValueError, '两维'):
            self.controller._configuration({'scenario': {'name': 'flybody_neural_link', 'env_kwargs': {}},
                'motor_checkpoint': 'old-body-checkpoint-must-not-be-read'})
        self.controller.state_provider = lambda: {}
        empty = self.controller._configuration({})
        self.assertEqual(empty['configs'][0]['scenario']['name'], 'flybody_neural_link')
        self.results['checks']['cross_body_motor_not_inherited_empty_lab_defaults_unified'] = True


if __name__ == '__main__':
    unittest.main(verbosity=2)
