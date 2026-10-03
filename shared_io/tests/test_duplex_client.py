"""Lifecycle regression checks at the HTTP boundary; no generated media claims."""
from unittest import TestCase,main
from unittest.mock import patch
from shared_io.bus import SharedLinkClient
from shared_io.duplex_client import DuplexLinkClient

class DuplexClientLifecycle(TestCase):
 def test_keep_same_epoch_and_increment_only_after_generate(self):
  calls=[]
  def rpc(client,op,req):
   calls.append((op,dict(req)))
   return {'epoch':'actual-test-epoch'} if op=='duplex/open' else {'status':'accepted'}
  c=DuplexLinkClient();request={'io_scope':'sensory_motor','brain_state':{'sha256':'fixture'},'event_id':'unit'}
  with patch.object(SharedLinkClient,'call',rpc):
   c.call('encode',request);c.call('decode',request);c.call('encode',{**request,'event_id':'unit2'});c.close()
  self.assertEqual([op for op,_ in calls],['duplex/open','duplex/prefill','duplex/generate','duplex/prefill','duplex/close'])
  self.assertEqual(calls[1][1]['sequence'],0);self.assertEqual(calls[3][1]['sequence'],1)
  self.assertEqual(calls[3][1]['epoch'],'actual-test-epoch');self.assertIsNone(c.epoch)
 def test_failed_close_invalidates_local_epoch_and_retries_same_owner(self):
  c=DuplexLinkClient();c.epoch='previous-epoch';c.sequence=8;timeout=c.timeout
  with patch.object(SharedLinkClient,'call',side_effect=RuntimeError('connection lost')):
   with self.assertRaises(RuntimeError):c.close()
  self.assertIsNone(c.epoch);self.assertEqual(c.timeout,timeout);self.assertEqual(c.failed_close['epoch'],'previous-epoch')
  calls=[]
  def recovered(client,op,req):
   calls.append((op,dict(req)));return {'epoch':'next-epoch'} if op=='duplex/open' else {'status':'accepted'}
  with patch.object(SharedLinkClient,'call',recovered):c.call('encode',{'io_scope':'sensory_motor','brain_state':{},'event_id':'next'})
  self.assertEqual([x[0]for x in calls],['duplex/close','duplex/open','duplex/prefill'])
  self.assertEqual(calls[0][1]['epoch'],'previous-epoch');self.assertIsNone(c.failed_close)
 def test_failed_generate_does_not_advance_sequence(self):
  c=DuplexLinkClient();c.epoch='epoch';c.sequence=5
  with patch.object(SharedLinkClient,'call',side_effect=RuntimeError('generation failed')):
   with self.assertRaises(RuntimeError):c.call('decode',{'event_id':'unit'})
  self.assertEqual(c.sequence,5)

if __name__=='__main__':main()
