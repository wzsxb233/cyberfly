"""Offline protocol tests only: no model weights, fake speech or GPU inference."""
import unittest
from shared_io.duplex_link import DuplexProtocol

class Ownership(unittest.TestCase):
    def setUp(self):
        self.p=DuplexProtocol('owner');self.req={'session_id':'owner','epoch':self.p.epoch,'sequence':0,'event_id':'unit-0'}
    def test_order_and_matching_pending(self):
        self.p.before_prefill(self.req)
        self.p.pending={'event_id':'unit-0'}
        with self.assertRaises(ValueError):self.p.before_prefill(self.req)
        with self.assertRaises(ValueError):self.p.before_generate({**self.req,'event_id':'other'})
        self.p.before_generate(self.req);self.p.completed()
        with self.assertRaises(ValueError):self.p.before_prefill(self.req)
        self.p.before_prefill({**self.req,'sequence':1,'event_id':'unit-1'})
    def test_cancel_epoch_cannot_own_reopened_session(self):
        newer=DuplexProtocol('owner')
        self.assertNotEqual(self.p.epoch,newer.epoch)
        with self.assertRaises(ValueError):newer.owner(self.req)
        self.p.closed=True
        with self.assertRaises(ValueError):self.p.owner(self.req)
    def test_reject_missing_sequence_or_owner(self):
        for update in ({'epoch':'wrong'},{'session_id':'other'},{'sequence':True},{'sequence':1},{'event_id':''}):
            with self.assertRaises(ValueError):self.p.before_prefill({**self.req,**update})
    def test_no_generate_without_prefill(self):
        with self.assertRaises(ValueError):self.p.before_generate(self.req)

if __name__=='__main__':unittest.main()
