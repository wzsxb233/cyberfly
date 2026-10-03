"""Ownership/cancellation unit tests; real physics evidence is separate."""
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import unittest

from shared_io.async_loop import AsyncSharedLoop,SnapshotBrain


class OwnershipChecks(unittest.TestCase):
    def test_snapshot_reads_do_not_expose_mutable_actor_snapshot(self):
        original={'sha256':'recorded','arrays':{'ids':{'shape':[166700]}}}
        frozen=SnapshotBrain({}, {}, original)
        original['sha256']='later'
        first=frozen.read_state();first['arrays']['ids']['shape'][0]=1
        self.assertEqual(frozen.read_state()['sha256'],'recorded')
        self.assertEqual(frozen.read_state()['arrays']['ids']['shape'],[166700])

    def test_worker_cannot_use_actor_entrypoint(self):
        loop=object.__new__(AsyncSharedLoop);loop.owner=threading.get_ident();loop.closed=False
        with ThreadPoolExecutor(max_workers=1) as pool:
            with self.assertRaisesRegex(RuntimeError,'Only the body actor'):
                pool.submit(loop._actor).result()

    def test_close_does_not_wait_for_inflight_http(self):
        entered=threading.Event();release=threading.Event()
        def blocked():entered.set();release.wait(5)
        loop=object.__new__(AsyncSharedLoop);loop.owner=threading.get_ident();loop.closed=False
        loop.cancelled=threading.Event();loop.executor=ThreadPoolExecutor(max_workers=1)
        loop.future=loop.executor.submit(blocked);self.assertTrue(entered.wait(1))
        try:
            start=time.monotonic();loop.close()
            self.assertLess(time.monotonic()-start,0.1)
            self.assertTrue(loop.closed);self.assertTrue(loop.cancelled.is_set())
            loop.close()  # Idempotent ownership-preserving cleanup.
        finally:
            release.set();loop.future.result(timeout=1)


if __name__=='__main__':unittest.main()
