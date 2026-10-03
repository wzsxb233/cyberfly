"""Private-filesystem configuration tests; no service, GPU or model requests."""
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import yaml

from vllm_integration import manage


class ManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.runtime = Path(self.temp.name)
        original = json.loads((manage.RUNTIME / 'config.json').read_text())
        profile = self.runtime / 'original.yaml'
        profile.write_bytes(Path(original['deploy_config']).read_bytes())
        original['deploy_config'] = str(profile)
        original['vllm'] = sys.executable
        (self.runtime / 'config.json').write_text(json.dumps(original))
        self.original = (self.runtime / 'config.json').read_bytes()
        self.profile = profile
        self.original_profile = profile.read_bytes()
        self.stack = ExitStack()
        self.stack.enter_context(patch.object(manage, 'RUNTIME', self.runtime))
        self.stack.enter_context(patch.object(manage, 'inspect_resources', return_value={'fixture': True}))

    def tearDown(self):
        self.stack.close()
        self.temp.cleanup()

    def test_prepare_keeps_exact_active_bytes_and_stage0_stage1(self):
        prepared = manage.prepare_speech()
        config = json.loads(Path(prepared['candidate_config']).read_text())
        self.assertEqual((self.runtime / 'config.json').read_bytes(), self.original)
        self.assertEqual(self.profile.read_bytes(), self.original_profile)
        old = yaml.safe_load(self.original_profile)
        new = yaml.safe_load(Path(config['deploy_config']).read_text())
        self.assertEqual(old['stages'][:2], new['stages'][:2])
        self.assertEqual(new['stages'][2]['devices'], '1')
        self.assertEqual(config['gpu_uuids'], [manage.GPU, manage.SPEECH_GPU])
        self.assertEqual(config['speech_resource_budget']['maximum_added_mib'], 10240)

    def test_cannot_activate_over_running_service_or_changed_candidate(self):
        prepared = manage.prepare_speech()
        with patch.object(manage, 'status', return_value={'supervisor_running': True}):
            with self.assertRaises(RuntimeError):
                manage.activate_speech()
        Path(prepared['candidate_config']).write_text('{}')
        with self.assertRaises(ValueError):
            manage._candidate()
        self.assertEqual((self.runtime / 'config.json').read_bytes(), self.original)

    def test_activation_then_rollback_restores_exact_bytes_without_start(self):
        manage.prepare_speech()
        with patch.object(manage, 'status', return_value={'supervisor_running': False}), \
             patch.object(manage, 'preflight'):
            self.assertEqual(manage.activate_speech()['status'], 'activated_not_started')
            self.assertEqual(manage.rollback()['status'], 'baseline_restored_not_started')
        self.assertEqual((self.runtime / 'config.json').read_bytes(), self.original)
        self.assertEqual(self.profile.read_bytes(), self.original_profile)

    def test_preflight_requires_real_headroom_and_resident_training_guard(self):
        prepared = manage.prepare_speech()
        config = json.loads(Path(prepared['candidate_config']).read_text())
        def memory(gpu):
            return {'gpu_uuid': gpu, 'free_mib': 23000 if gpu == manage.GPU else 13000}
        with patch.object(manage, 'gpu_snapshot', side_effect=memory), \
             patch.object(manage, '_gpu1_guard', return_value={'ready': False}), \
             patch.object(manage.socket, 'socket'):
            with self.assertRaisesRegex(RuntimeError, '容量保护'):
                manage.preflight(config)
        def low_memory(gpu):
            return {'gpu_uuid': gpu, 'free_mib': 23000 if gpu == manage.GPU else 11000}
        with patch.object(manage, 'gpu_snapshot', side_effect=low_memory), \
             patch.object(manage, '_gpu1_guard', return_value={'ready': True}), \
             patch.object(manage.socket, 'socket'):
            with self.assertRaisesRegex(RuntimeError, '12GiB'):
                manage.preflight(config)


if __name__ == '__main__':
    unittest.main()
