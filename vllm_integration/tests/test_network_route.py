"""Fake routing table only; never mutates the host rules."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from vllm_integration.network_route import ensure_local_route, inspect_local_route, rollback_local_route


class FakeRoute:
    def __init__(self, directions=(), fail_sport=False):
        self.rules=[self.rule(d) for d in directions]
        self.commands=[]
        self.fail_sport=fail_sport
    @staticmethod
    def rule(direction):
        return dict(priority=0,src='127.0.0.1',dst='127.0.0.1',ipproto='tcp',table='local',**{direction:18649})
    def __call__(self, args):
        self.commands.append(list(args))
        if args == ['ip','-j','rule','show']: return json.dumps(self.rules)
        direction='dport' if 'dport' in args else 'sport'
        if args[2] == 'add':
            if direction == 'sport' and self.fail_sport: raise RuntimeError('simulated add failure')
            self.rules.append(self.rule(direction))
        elif args[2] == 'del': self.rules.remove(self.rule(direction))
        else: raise AssertionError(args)
        return ''


class RouteTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='cyberfly-route-fixture-');self.addCleanup(temporary.cleanup)
        patcher=patch('vllm_integration.network_route.ROOT',Path(temporary.name));patcher.start();self.addCleanup(patcher.stop)
    def test_idempotent_existing_rules_never_owned_or_removed(self):
        runner=FakeRoute(('dport','sport'));before=copy.deepcopy(runner.rules)
        result=ensure_local_route(runner=runner)
        self.assertEqual(result['status'],'present_unchanged');self.assertEqual(result['added_directions'],[])
        rollback_local_route(result['receipt_path'],runner=runner)
        self.assertEqual(before,runner.rules)
    def test_new_rules_and_exact_owned_rollback(self):
        runner=FakeRoute(('dport',))
        result=ensure_local_route(runner=runner)
        self.assertEqual(result['added_directions'],['sport'])
        restored=rollback_local_route(result['receipt_path'],runner=runner)
        self.assertEqual(restored['removed_directions'],['sport'])
        self.assertEqual(runner.rules,[runner.rule('dport')])
        self.assertEqual(rollback_local_route(result['receipt_path'],runner=runner)['status'],'rolled_back')
    def test_partial_creation_failure_removes_only_new_rules(self):
        runner=FakeRoute(fail_sport=True)
        with self.assertRaises(RuntimeError): ensure_local_route(runner=runner)
        self.assertEqual(runner.rules,[])
    def test_duplicate_rule_rejected_and_dryrun_never_adds(self):
        runner=FakeRoute(('dport','dport'))
        with self.assertRaises(RuntimeError): inspect_local_route(runner=runner)
        runner=FakeRoute()
        self.assertEqual(len(ensure_local_route(dry_run=True,runner=runner)['would_add']),2)
        self.assertTrue(all(args == ['ip','-j','rule','show'] for args in runner.commands))


if __name__ == '__main__': unittest.main()
