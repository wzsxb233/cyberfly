"""Receipt and stale-input guards only; these mocks do not prove GPU inference."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from vllm_integration.native_flow import NativeOmniFlow


class NativeReceiptTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        keys = ('model_revision', 'weights_manifest_sha256', 'vllm_omni_commit',
                'io_scope', 'codec_sha256', 'coupler_sha256', 'coupler_generation',
                'read_ids_sha256', 'write_ids_sha256', 'soft_tokens_sha256')
        self.condition = {key: 'test-' + key for key in keys}
        self.condition['sha256'] = 'condition-file-sha'
        self.descriptor = {**self.condition, 'event_id': 'event',
            'request_id': 'chatcmpl-' + 'a' * 32 + '-1234abcd',
            'sequence': 1, 'path': str(Path(self.folder.name)/'mutable-worker.npz'), 'sha256': 'worker-file-sha',
            'condition_file_sha256': self.condition['sha256'],
            'same_original_latent_returned_to_native_tts': True, 'original_model_trainable_parameters': 0}
        self.calls = []
        def currents(hidden, **kwargs):
            self.calls.append((hidden.copy(), kwargs))
            return {'test_only': True}
        self.flow = NativeOmniFlow.__new__(NativeOmniFlow)
        self.flow.ttl = 5
        self.flow.turns = {'fly': {'event_id': 'event', 'condition': self.condition,
            'client_request_id': 'a' * 32, 'expected_model': 'cyberfly-minicpm-o-4.5',
            'last_applied_sequence': -1, 'directory': self.folder.name}}
        self.flow.couplers = {'fly': SimpleNamespace(currents=currents)}

    def test_pending_and_expired_do_not_apply(self):
        for error in (FileNotFoundError(), ValueError('Latest model hidden expired')):
            with patch('vllm_integration.latent_contract.read_latest_hidden', side_effect=error):
                self.assertIsNone(self.flow._latest('fly'))
        self.assertEqual(self.calls, [])

    def test_same_event_foreign_request_fails_before_any_progress_or_projection(self):
        descriptor = {**self.descriptor, 'request_id': 'chatcmpl-' + 'b' * 32 + '-1234abcd'}
        with patch('vllm_integration.latent_contract.read_latest_hidden',
                   return_value=(np.zeros(4096, dtype=np.float32), descriptor)):
            with self.assertRaisesRegex(ValueError, 'submitted model request'):
                self.flow._latest('fly')
        self.assertEqual(self.calls, [])

    def test_wrong_condition_fails_before_projection(self):
        descriptor = {**self.descriptor, 'coupler_sha256': 'wrong-head'}
        with patch('vllm_integration.latent_contract.read_latest_hidden', return_value=(np.zeros(4096,dtype=np.float32),descriptor)):
            with self.assertRaisesRegex(ValueError, 'coupler_sha256'):
                self.flow._latest('fly')
        self.assertEqual(self.calls, [])

    def test_selected_hidden_survives_worker_replacement_and_deduplicates(self):
        hidden = np.arange(4096,dtype=np.float32)
        with patch('vllm_integration.latent_contract.read_latest_hidden', return_value=(hidden,self.descriptor)):
            _, selected = self.flow._latest('fly')
            self.flow.turns['fly']['last_applied_sequence'] = selected['sequence']
            self.assertIsNone(self.flow._latest('fly'))
        Path(self.descriptor['path']).write_bytes(b'newer worker data')
        np.testing.assert_array_equal(np.load(selected['path'],allow_pickle=False),hidden)
        self.assertEqual(len(self.calls),1)
        self.assertEqual(self.calls[0][1]['source']['selected_hidden']['path'],selected['path'])


if __name__ == '__main__': unittest.main()
