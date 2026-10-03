"""Concurrent file publication checks; no GPU or model request is performed."""
import copy
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

import vllm_integration.latent_contract as contract


class PublicationTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='cyberfly-hidden-publication-')
        self.addCleanup(temporary.cleanup)
        replacement=patch.object(contract,'ROOT',Path(temporary.name));replacement.start();self.addCleanup(replacement.stop)
        self.event='publication-fixture';self.request='worker-request-fixture'
        self.folder=contract.ROOT/'artifacts/vllm_latent'/contract.sha(self.event.encode())[:24]/contract.sha(self.request.encode())[:24]
        self.folder.mkdir(parents=True)
        self.file=self.folder/'latest-hidden.npz';self.pointer=self.folder/'latest.json'
        with contract._hidden_publication_lock:contract._hidden_publications.clear()

    def make(self,sequence=0,value=0.):
        hidden=np.full((1,4096),value,dtype=np.float32);tokens=np.array([42],dtype=np.int64)
        stream=io.BytesIO();np.savez(stream,hidden=hidden,input_token_ids=tokens);raw=stream.getvalue()
        descriptor={'schema':1,'event_id':self.event,'request_id':self.request,'sequence':sequence,
                    'model_revision':contract.REVISION,'vllm_omni_commit':contract.OMNI_COMMIT,
                    'published_at':time.time(),'path':str(self.file),'sha256':contract.sha(raw),
                    'hidden_sha256':contract.sha(hidden.tobytes()),'condition_file_sha256':'1'*64,
                    'soft_tokens_sha256':'2'*64,'input_token_ids':[42],'tokens_processed_this_step':1}
        return raw,descriptor

    def publish(self,raw,descriptor,parent=True):
        temporary=self.file.with_suffix('.tmp');temporary.write_bytes(raw);temporary.replace(self.file)
        contract.write_json(self.pointer,descriptor)
        if parent:contract.write_json(self.folder.parent/'latest.json',descriptor)

    def test_stale_parent_uses_new_authoritative_child(self):
        raw,old=self.make(0,1);self.publish(raw,old)
        raw,new=self.make(1,2);self.publish(raw,new,parent=False)
        hidden,actual=contract.read_latest_hidden(self.event)
        self.assertEqual(actual['sequence'],1);np.testing.assert_array_equal(hidden,np.full(4096,2,np.float32))

    def test_delayed_child_commit_yields_then_recovers_without_sleeping_actor(self):
        raw,old=self.make(0,1);self.publish(raw,old)
        raw,new=self.make(1,2);self.file.write_bytes(raw)
        with patch.object(contract.time,'sleep',side_effect=AssertionError('Actor must not sleep')):
            with self.assertRaises(contract.LatestHiddenPending) as caught:
                contract.read_latest_hidden(self.event,request_id=self.request)
        self.assertEqual(caught.exception.reason,'uncommitted_pair_grace_not_yet_classified')
        contract.write_json(self.pointer,new)
        self.assertEqual(contract.read_latest_hidden(self.event,request_id=self.request)[1]['sequence'],1)

    def test_stable_corruption_is_not_swallowed_as_permanent_pending(self):
        raw,descriptor=self.make();self.publish(raw,descriptor);self.file.write_bytes(b'corrupted')
        with patch.object(contract.time,'monotonic',return_value=100.):
            with self.assertRaises(contract.LatestHiddenPending): contract.read_latest_hidden(self.event)
        with patch.object(contract.time,'monotonic',return_value=100.36):
            with self.assertRaisesRegex(ValueError,'Stable latest hidden artifact SHA mismatch'):
                contract.read_latest_hidden(self.event)

    def test_changing_bad_files_also_have_absolute_deadline(self):
        raw,descriptor=self.make();self.publish(raw,descriptor)
        for number,clock in enumerate([10.,10.3,10.6,10.9,11.2,11.5,11.8,12.1]):
            self.file.write_bytes(bytes([number]))
            with patch.object(contract.time,'monotonic',return_value=clock):
                if number<7:
                    with self.assertRaises(contract.LatestHiddenPending):contract.read_latest_hidden(self.event)
                else:
                    with self.assertRaisesRegex(ValueError,'across changing publications'):contract.read_latest_hidden(self.event)

    def test_committed_tensor_identity_and_request_tampering_fail_immediately(self):
        raw,descriptor=self.make();self.publish(raw,{**descriptor,'hidden_sha256':'0'*64})
        with self.assertRaisesRegex(ValueError,'tensor identity'):contract.read_latest_hidden(self.event)
        self.publish(raw,{**descriptor,'request_id':'other'})
        with self.assertRaisesRegex(ValueError,'event/request'):contract.read_latest_hidden(self.event,request_id=self.request)
        self.publish(raw,{**descriptor,'path':str(self.folder.parent/'elsewhere.npz')})
        with self.assertRaisesRegex(ValueError,'exact event/request'):contract.read_latest_hidden(self.event,request_id=self.request)

    def test_actual_concurrent_atomic_replacements_preserve_verified_pairs(self):
        raw,descriptor=self.make();self.publish(raw,descriptor)
        def writer():
            for sequence in range(1,7):
                raw,new=self.make(sequence,float(sequence))
                temporary=self.file.with_suffix('.writer');temporary.write_bytes(raw);temporary.replace(self.file)
                time.sleep(.012)
                contract.write_json(self.pointer,new)
                time.sleep(.008)
                contract.write_json(self.folder.parent/'latest.json',new)
        thread=threading.Thread(target=writer);thread.start()
        verified=[];pending=0
        try:
            while thread.is_alive():
                try:
                    hidden,desc=contract.read_latest_hidden(self.event)
                    np.testing.assert_array_equal(hidden,np.full(4096,desc['sequence'],np.float32))
                    verified.append(desc['sequence'])
                except contract.LatestHiddenPending:pending+=1
                time.sleep(.003)
        finally:thread.join()
        self.assertEqual(contract.read_latest_hidden(self.event)[1]['sequence'],6)
        self.assertGreater(pending,0)
        self.assertGreater(len(verified),0)


if __name__=='__main__':unittest.main()
