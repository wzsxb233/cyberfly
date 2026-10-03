"""Stream boundary regressions; synthetic protocol data, no model/body calls."""
from concurrent.futures import Future
from copy import deepcopy
import io
import json
import unittest
from unittest.mock import patch

from vllm_integration.client import OmniClient
from vllm_integration.native_flow import NativeOmniFlow
from vllm_integration.streaming import LatestProgress


class StreamingRequestBindingTests(unittest.TestCase):
    def make_flow(self, *, hidden_request, client_request='a' * 32,
                  response_id=None, failed=False):
        progress = LatestProgress()
        progress.publish({'schema': 1, 'event_id': '1' * 32, 'sequence': 2,
                          'request_id': client_request,
                          'response_id': response_id or 'chatcmpl-' + client_request,
                          'model': 'cyberfly-minicpm-o-4.5', 'text': 'protocol only',
                          'audio_chunks': [], 'complete': False})
        future = Future()
        if failed:
            future.set_exception(RuntimeError('SSE ended before completion marker'))
        flow = NativeOmniFlow.__new__(NativeOmniFlow)
        flow.turns = {'lab': {'event_id': '1' * 32, 'request_id': hidden_request,
                             'client_request_id': client_request,
                             'expected_model': 'cyberfly-minicpm-o-4.5',
                             'verified_applications': ['synthetic-receipt'],
                             'progress': progress, 'future': future}}
        flow.responses, flow.errors = {}, {}
        return flow

    def assert_not_verified(self, flow):
        try:
            result = flow.stream_progress('lab')
        except (RuntimeError, ValueError):
            return
        self.assertFalse((result or {}).get('numerical_feedback_verified'), result)

    def test_partial_audio_cannot_borrow_another_request_receipt(self):
        flow = self.make_flow(hidden_request='chatcmpl-' + 'b' * 32 + '-1234abcd')
        self.assert_not_verified(flow)

    def test_foreign_external_sse_identity_is_not_verified(self):
        flow = self.make_flow(hidden_request='chatcmpl-' + 'a' * 32 + '-1234abcd',
                              response_id='chatcmpl-' + 'b' * 32)
        self.assert_not_verified(flow)

    def test_completed_error_revokes_partial_without_waiting_for_body_tick(self):
        flow = self.make_flow(hidden_request='chatcmpl-' + 'a' * 32 + '-1234abcd', failed=True)
        try:
            result = flow.stream_progress('lab')
        except (RuntimeError, ValueError):
            return
        self.assertTrue((result or {}).get('error') or
                        not (result or {}).get('numerical_feedback_verified'), result)

    def test_matching_active_request_keeps_verified_progress(self):
        flow = self.make_flow(hidden_request='chatcmpl-' + 'a' * 32 + '-1234abcd')
        before = deepcopy(flow.turns['lab']['progress'].snapshot())
        result = flow.stream_progress('lab')
        self.assertTrue(result['numerical_feedback_verified'])
        self.assertEqual(result['request_id'], 'a' * 32)
        self.assertEqual(before, flow.turns['lab']['progress'].snapshot())


class ClientRequestBindingTests(unittest.TestCase):
    def response(self, *, request='a' * 32, model='cyberfly-minicpm-o-4.5'):
        row = {'id': 'chatcmpl-' + request, 'model': model, 'modality': 'text',
               'choices': [{'index': 0, 'delta': {'content': 'protocol test'}}]}
        return io.BytesIO(b'data: ' + json.dumps(row).encode() + b'\n\ndata: [DONE]\n\n')

    def test_explicit_request_matches_wire_and_completed_response(self):
        published = []
        with patch('urllib.request.urlopen', return_value=self.response()) as opener:
            result = OmniClient().generate('test', request_id='a' * 32,
                stream=True, on_progress=published.append)
        wire = json.loads(opener.call_args.args[0].data)
        self.assertEqual(wire['request_id'], 'a' * 32)
        self.assertEqual(result['response_id'], 'chatcmpl-' + 'a' * 32)
        self.assertEqual(published[0]['response_id'], result['response_id'])

    def test_foreign_first_stream_chunk_never_reaches_progress_consumer(self):
        for options in ({'request': 'b' * 32}, {'model': 'foreign-model'}):
            published = []
            with patch('urllib.request.urlopen', return_value=self.response(**options)):
                with self.assertRaisesRegex(ValueError, 'identity'):
                    OmniClient().generate('test', request_id='a' * 32,
                        stream=True, on_progress=published.append)
            self.assertEqual(published, [])

    def test_invalid_explicit_id_rejected_before_http(self):
        for request_id in ('', 'a' * 31, 'A' * 32, 'a' * 33, 12):
            with patch('urllib.request.urlopen') as opener:
                with self.assertRaises(ValueError):
                    OmniClient().generate('test', request_id=request_id)
                opener.assert_not_called()

    def test_foreign_completed_nonstream_reply_rejected(self):
        response = {'id': 'chatcmpl-' + 'b' * 32, 'model': 'cyberfly-minicpm-o-4.5',
                    'choices': [{'message': {'content': 'foreign'}}]}
        with patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(response).encode())):
            with self.assertRaisesRegex(ValueError, 'identity'):
                OmniClient().generate('test', request_id='a' * 32)


if __name__ == '__main__':
    unittest.main()
