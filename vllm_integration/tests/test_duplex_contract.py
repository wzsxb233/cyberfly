"""Offline protocol checks. Fake audio/transport are NOT inference evidence."""
import asyncio
import base64
import copy
import json
import unittest

import numpy as np

from vllm_integration.duplex_client import NativeLatentDuplexClient
from vllm_integration.duplex_contract import (
    EXTENSION, UnitLedger, fuse_current_unit, require_native_capabilities,
    reserve_append_slots, scheduled_slice,
)
from vllm_integration.latent_contract import ROOT, load_condition


def actual_condition():
    folder = ROOT / 'artifacts/vllm_latent/73b76825873118ae8d35ccff'
    return json.loads(sorted(folder.glob('soft-*.json'))[0].read_text())


def native_caps(*, extended=False):
    # Exact stock capability payload; importing this module does not load a model.
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.capabilities import minicpmo45_native_capabilities
    result = minicpmo45_native_capabilities().as_dict()
    if extended:
        result[EXTENSION] = dict(schema=1, placement='before_sensory', append_only=True,
                                 facts_in_model_input=False, generation_hidden_export=True,
                                 unit_samples=16000)
    return result


class ContractTests(unittest.TestCase):
    def test_stock_native_passes_native_check_but_refuses_uninstalled_extension(self):
        session = {'capabilities': native_caps()}
        require_native_capabilities(session, require_latent=False)
        with self.assertRaisesRegex(ValueError, 'not installed'):
            require_native_capabilities(session)
        session['capabilities']['implementation_level'] = 'serving_session_adapter'
        with self.assertRaisesRegex(ValueError, 'not native'):
            require_native_capabilities(session, require_latent=False)

    def test_real_archived_numeric_file_packet_and_no_hidden_text(self):
        condition = actual_condition()
        ledger = UnitLedger('offline-protocol', 0)
        pcm = bytes(32000)  # Explicit synthetic silence, protocol test only.
        event, audit = ledger.prepare(event_id=condition['event_id'], pcm16=pcm, condition=condition)
        self.assertEqual(base64.b64decode(event['audio']), pcm)
        self.assertEqual(audit['runtime_seq'], None)
        self.assertEqual(audit['unit_index'], 0)
        self.assertEqual(event[EXTENSION]['soft_token_count'], 6)
        self.assertNotIn('text', event)
        self.assertNotIn('force_speak', event)
        self.assertNotIn('force_listen', event)
        ledger.submitted(audit)
        with self.assertRaisesRegex(ValueError, 'replayed'):
            ledger.prepare(event_id=condition['event_id'], pcm16=pcm, condition=condition)

    def test_epoch_pin_and_audio_boundary(self):
        condition = actual_condition()
        ledger = UnitLedger('offline-protocol', 0)
        ledger.identity = {"wrong": "identity"}
        with self.assertRaisesRegex(ValueError, 'changed'):
            ledger.prepare(event_id=condition['event_id'], pcm16=bytes(32000), condition=condition)
        with self.assertRaisesRegex(ValueError, 'exactly 1 s'):
            UnitLedger('offline-protocol', 0).prepare(event_id=condition['event_id'], pcm16=bytes(64000), condition=condition)

    def test_current_unit_insertion_preserves_native_boundary_and_scheduler_chunks(self):
        soft = load_condition(actual_condition())
        # Artificial original embeddings solely exercise the indexing contract.
        original = np.arange(9 * 4096, dtype=np.float32).reshape(9, 4096) / 40000
        ids = [1, 2, 151707, 4, 5, 6, 7, 8, 9]
        fused, token_ids, info = fuse_current_unit(original, ids, soft, unit_token_id=151707)
        np.testing.assert_array_equal(fused[3:9], soft)
        np.testing.assert_array_equal(fused[-1], original[-1])
        np.testing.assert_array_equal(np.concatenate((fused[:3], fused[9:])), original)
        self.assertEqual(info['soft_start_in_actual_append'], 3)
        pieces = [scheduled_slice(fused, token_ids, prompt_len=15, token_offset=i, span=n)[0]
                  for i, n in [(0, 5), (5, 7), (12, 3)]]
        np.testing.assert_array_equal(np.concatenate(pieces), fused)
        with self.assertRaisesRegex(ValueError, 'Generation tokens'):
            scheduled_slice(fused, token_ids, prompt_len=15, token_offset=15, span=1)
        with self.assertRaisesRegex(ValueError, 'budget mismatch'):
            scheduled_slice(fused, token_ids, prompt_len=14, token_offset=0, span=4)
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            fuse_current_unit(original, [151707] * 9, soft, unit_token_id=151707)

    def test_official_plan_reserve_is_only_new_append(self):
        from vllm_omni.engine.duplex.messages import DuplexFence
        from vllm_omni.engine.duplex.runtime import DuplexInputMode
        from vllm_omni.model_executor.models.minicpmo_4_5.duplex.runtime import build_duplex_data_plane_prompt

        payload = dict(audio=base64.b64encode(bytes(64000)).decode(), format='pcm_f32le', sample_rate_hz=16000)
        original = build_duplex_data_plane_prompt(
            request_id='offline-plan', fence=DuplexFence('offline-session', epoch=0),
            session_config={}, runtime_config={}, seq=2, turn_seq=2,
            mode=DuplexInputMode.APPEND_AUDIO_CHUNK, payload=payload, final=False)
        before = copy.deepcopy(original)
        reserved = reserve_append_slots(original, soft_token_count=6)
        self.assertEqual(original, before)
        self.assertEqual(len(reserved['prompt_token_ids']), len(original['prompt_token_ids']) + 6)
        self.assertEqual(reserved['model_intermediate_buffer']['duplex']['payload'], payload)
        with self.assertRaises(ValueError):
            reserve_append_slots(reserved, soft_token_count=6)


class FakeSocket:
    def __init__(self, caps):
        self.caps = caps
        self.sent = []
        self.queue = asyncio.Queue()
        self.closed = False

    async def send(self, raw):
        event = json.loads(raw)
        self.sent.append(event)
        if event['type'] == 'session.update':
            self.queue.put_nowait({'type': 'session.created', 'session': {
                'session_id': 'offline-session', 'epoch': 0, 'capabilities': self.caps}})
        elif event['type'] == 'session.close':
            self.queue.put_nowait({'type': 'session.closed', 'session_id': 'offline-session'})

    async def recv(self):
        return json.dumps(await self.queue.get())

    async def close(self):
        self.closed = True


class OfficialClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_stock_handshake_refuses_latent_and_cleans_session(self):
        socket = FakeSocket(native_caps())
        async def connect(url):
            return socket
        with self.assertRaisesRegex(ValueError, 'not installed'):
            async with NativeLatentDuplexClient('ws://127.0.0.1:18649', model='test',
                    ref_audio='data:audio/wav;base64,test-only', session_id='offline-session', connect=connect):
                self.fail('Stock server silently accepted latent extension')
        self.assertTrue(socket.closed)
        self.assertIn('session.close', [event['type'] for event in socket.sent])

    async def test_official_client_serializes_extension_without_enabling_a_server(self):
        socket = FakeSocket(native_caps(extended=True))
        async def connect(url):
            self.assertIn('duplex=1', url)
            self.assertIn('autostart=0', url)
            return socket
        condition = actual_condition()
        async with NativeLatentDuplexClient('ws://127.0.0.1:18649', model='test',
                ref_audio='data:audio/wav;base64,test-only', session_id='offline-session', connect=connect) as client:
            audit = await client.append_unit(event_id=condition['event_id'], pcm16=bytes(32000), condition=condition)
            self.assertEqual(audit['status'], 'sent_not_yet_worker_verified')
            self.assertEqual(socket.sent[-1][EXTENSION]['condition']['sha256'], condition['sha256'])
        self.assertTrue(socket.closed)


if __name__ == '__main__':
    unittest.main()
