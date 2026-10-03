"""CPU-only train/deploy checks using archived real tensors and explicit fixtures.

Fixture event metadata is deliberately mocked: this verifies training plumbing,
not a newly completed live shared interaction. Production pointers stay untouched.
"""
from copy import deepcopy
import json
from pathlib import Path
import threading
import unittest
from unittest.mock import patch
import uuid

import numpy as np
import torch

from shared_io import sensory_sources
from shared_io.sensory_training import (ROOT, _sha, completed_event, prepare_sensory_training,
    run_sensory_training, prepare_body_embedding_training, write_json)


class SensoryTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.output = ROOT / 'artifacts/sensory_training_checks' / uuid.uuid4().hex[:12]
        cls.output.mkdir(parents=True)
        cls.body = json.loads((ROOT / 'artifacts/body_sense_checks/e1251663e5f2/evidence.json').read_text())
        cls.modal = json.loads((ROOT / 'artifacts/modal_currents_checks/db2eddb03e65/evidence.json').read_text())
        cls.production_pointers = {str(p): p.read_bytes() for p in sensory_sources.BASE.glob('**/current.json')}
        cls.base_patch = patch.object(sensory_sources, 'BASE', cls.output / 'isolated-pointers')
        cls.base_patch.start()
        cls.results = {}
        cls.fixture_id = 'a' * 32
        cls.fixture = {'event_id': cls.fixture_id, 'phase': 'complete', 'shared_input_verified': True,
            'body_features_file': cls.body['untrained_encoder']['body_features_file'],
            'model_evidence': {key: cls.modal['modal_embeddings_file'][key]
                               for key in ('model_revision', 'weights_manifest_sha256')}}
        packet = json.loads(Path(cls.fixture['body_features_file']['packet_path']).read_text())
        cls.fixture['image_rgb_sha256'] = packet['vision']['rgb_sha256']
        cls.fixture['model_evidence']['modal_embeddings_file'] = {
            **cls.modal['modal_embeddings_file'], 'event_id': cls.fixture_id}
        # Only fixture metadata gets the test event ID. Real archived token bytes
        # and their tensor/file SHA checks remain unchanged.
        original = json.loads(Path(cls.modal['source_json']).read_text())
        hidden = np.asarray(original['hidden'], dtype=np.float32)
        np.save(cls.output / 'model_hidden.npy', hidden, allow_pickle=False)
        cls.fixture['model_hidden_sha256'] = _sha(hidden.tobytes())

    @classmethod
    def tearDownClass(cls):
        cls.base_patch.stop()
        after = {str(p): p.read_bytes() for p in sensory_sources.BASE.glob('**/current.json')}
        if after != cls.production_pointers:
            raise AssertionError('Offline tests modified production sensory pointers')
        write_json(cls.output / 'evidence.json', {'status': 'passed',
            'isolation': 'CPU only; no GPU/model/lab calls and no production pointer changes',
            'fixture_event_metadata': True,
            'source_body_evidence': str(ROOT / 'artifacts/body_sense_checks/e1251663e5f2/evidence.json'),
            'source_modal_evidence': str(ROOT / 'artifacts/modal_currents_checks/db2eddb03e65/evidence.json'),
            'source_token_bytes_unchanged': cls.modal['modal_embeddings_file']['sha256'],
            'results': cls.results})
        print('EVIDENCE', cls.output / 'evidence.json')

    def source(self):
        return patch('shared_io.sensory_training.completed_event',
                     return_value=(deepcopy(self.fixture), self.output, 'offline-fixture-metadata'))

    def test_01_completed_event_guard(self):
        with self.assertRaises(ValueError):
            completed_event('../not-an-event')
        with self.assertRaises(ValueError):
            prepare_sensory_training({'event_id': 'fbdae1ab6f95465b8c9cc185587061c8',
                'modality': 'body', 'label_source': 'Explicit test', 'target_current_mv': .08})

    def test_02_targets_and_source_validation(self):
        request = {'event_id': self.fixture_id, 'modality': 'body', 'label_source': 'Explicit test current labels',
                   'target_current_mv': .08, 'steps': 2}
        with self.source():
            for mutation in ({'label_source': ''}, {'steps': True}, {'target_current_mv': 2},
                             {'learning_rate': float('nan')}, {'target_currents_mv': [0]}):
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    prepare_sensory_training({**request, **mutation})
            full = np.zeros(166700, dtype=np.float32)
            full[0] = .01
            plan = prepare_sensory_training(request)
            outside = np.setdiff1d(np.arange(166700), plan['adapter'].indices)
            full[outside[0]] = .02
            with self.assertRaises(ValueError):
                prepare_sensory_training({key: value for key, value in request.items()
                    if key != 'target_current_mv'} | {'target_currents_mv': full.tolist()})
        corrupted = deepcopy(self.fixture)
        corrupted['image_rgb_sha256'] = 'changed'
        with patch('shared_io.sensory_training.completed_event', return_value=(corrupted, self.output, 'fixture')):
            with self.assertRaises(ValueError):
                prepare_sensory_training(request)

    def test_03_all_three_heads_actual_training_and_pointer_deploy(self):
        for modality in ('vision', 'audio', 'body'):
            with self.subTest(modality=modality), self.source():
                request = {'event_id': self.fixture_id, 'modality': modality, 'steps': 2,
                    'label_source': 'Explicit constant +0.08 mV engineering unit-test label; not natural receptor recordings',
                    'target_current_mv': .08}
                plan = prepare_sensory_training(request)
                output = ROOT / 'artifacts/model-training' / ('offline-sensory-' + modality + '-' + self.output.name)
                result = run_sensory_training(plan, output)
                self.assertGreater(result['changed_parameters'], 0)
                self.assertTrue(result['reload_verified'])
                self.assertFalse(result['base_model_weights_changed'])
                self.assertFalse(result['connectome_weights_changed'])
                rows = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]
                self.assertEqual([row['step'] for row in rows], [1, 2])
                self.assertTrue(all(row['changed_parameters'] > 0 and row['parameter_l2_change'] > 0 for row in rows))
                fresh = sensory_sources.SharedSensorySources()
                active = fresh.body_adapter(plan['packet']) if modality == 'body' else fresh.modal_adapter(modality)
                if modality == 'body':
                    self.assertEqual(active.adapter_sha256(), result['final_adapter_sha256'])
                else:
                    self.assertEqual(active._weight_sha(), result['weights_sha256_after'])
                self.results[modality] = {'output': str(output), 'changed_parameters': result['changed_parameters'],
                    'reload_verified': result['reload_verified'], 'deployed_checkpoint': result['deployed_checkpoint'],
                    'losses': result.get('losses', result.get('step_training_mse')),
                    'final_loss': result.get('loss_after', result.get('final_training_mse'))}

    def test_04_body_embedding_record_is_exact_same_source(self):
        with self.source():
            request = prepare_body_embedding_training({'event_id': self.fixture_id, 'steps': 2}, self.output / 'gpu-job-unused')
        record = request['records'][0]
        self.assertEqual(record['body_features_file'], self.fixture['body_features_file'])
        self.assertEqual(record['target_hidden_file_sha256'], _sha((self.output / 'model_hidden.npy').read_bytes()))
        self.assertIn('self-supervised', record['label_source'])
        bad = deepcopy(self.fixture)
        bad['model_hidden_sha256'] = 'corrupted'
        with patch('shared_io.sensory_training.completed_event', return_value=(bad, self.output, 'fixture')):
            with self.assertRaises(ValueError):
                prepare_body_embedding_training({'event_id': self.fixture_id}, self.output / 'unused')

    def test_05_body_endpoint_transport_contract_without_gpu(self):
        from cyberfly_core.server import Jobs
        finished, calls = threading.Event(), []
        def fake_call(client, operation, request):
            try:
                self.assertEqual(operation, 'train_body_embedding')
                self.assertFalse((Path(request['output']) / 'progress.json').exists())
                calls.append(request)
                return {'status': 'offline_transport_mock', 'gpu_called': False}
            finally:
                finished.set()
        with self.source(), patch('shared_io.bus.SharedLinkClient.call', new=fake_call):
            response = Jobs().launch_body_embedding({'event_id': self.fixture_id, 'steps': 2})
            self.assertTrue(finished.wait(5))
        self.assertEqual(response['state'], 'starting')
        self.assertEqual(calls[0]['records'][0]['body_features_file'], self.fixture['body_features_file'])
        self.results['body_embedding_transport'] = {'gpu_called': False, 'records': calls[0]['records']}

    def test_06_stop_saves_partial_without_deployment(self):
        from shared_io import sensory_training
        for modality in ('body', 'vision'):
            with self.subTest(modality=modality), self.source():
                plan = prepare_sensory_training({'event_id': self.fixture_id, 'modality': modality,
                    'steps': 10, 'target_current_mv': -.03, 'label_source': 'Explicit interruption-test teaching target'})
                before = {str(p): p.read_bytes() for p in sensory_sources.BASE.glob('**/current.json')}
                output = ROOT / 'artifacts/model-training' / ('offline-stop-' + modality + '-' + self.output.name)
                def write_then_stop(path, value):
                    write_json(path, value)
                    if Path(path) == output / 'progress.json' and value.get('step') == 1:
                        (output / 'STOP').touch()
                with patch.object(sensory_training, 'write_json', side_effect=write_then_stop):
                    result = run_sensory_training(plan, output)
                self.assertEqual(result['status'], 'interrupted')
                self.assertEqual(result['step'], 1)
                self.assertTrue(result['reload_verified'])
                self.assertTrue(result['deployment_skipped'])
                self.assertIsNone(result['deployed_checkpoint'])
                self.assertEqual(before, {str(p): p.read_bytes() for p in sensory_sources.BASE.glob('**/current.json')})
                self.results['interrupted_' + modality] = {'step': result['step'], 'status': result['status'],
                    'checkpoint': result['checkpoint'], 'deployed': False}


if __name__ == '__main__':
    unittest.main()
