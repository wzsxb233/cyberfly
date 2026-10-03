"""Idempotent, port-specific WSL mirrored-loopback workaround.

No firewall, service, Windows setting or model request is touched. Rollback
accepts only a receipt for rules this invocation actually added in this boot.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
PORT = 18649
DIRECTIONS = ('dport', 'sport')


def _run(argv):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=5)
    if result.returncode:
        raise RuntimeError('Exact loopback route command failed: ' + result.stderr.strip())
    return result.stdout


def _arguments(direction):
    if direction not in DIRECTIONS:
        raise ValueError('Only the fixed service source/destination ports are supported')
    return ['pref','0','from','127.0.0.1/32','to','127.0.0.1/32',
            'ipproto','tcp',direction,str(PORT),'lookup','local']


def _matching(rule, direction):
    allowed = {'priority','src','dst','ipproto','table',direction,'protocol'}
    return (set(rule) <= allowed and rule.get('priority') == 0
            and rule.get('src') in ('127.0.0.1','127.0.0.1/32')
            and rule.get('dst') in ('127.0.0.1','127.0.0.1/32')
            and rule.get('ipproto') in ('tcp',6) and rule.get(direction) == PORT
            and rule.get('table') in ('local',255))


def inspect_local_route(*, runner=_run):
    rules = json.loads(runner(['ip','-j','rule','show']))
    if not isinstance(rules,list):
        raise RuntimeError('Unexpected policy routing response')
    counts = {direction:sum(_matching(rule,direction) for rule in rules) for direction in DIRECTIONS}
    if any(count > 1 for count in counts.values()):
        raise RuntimeError('Duplicate matching rules require inspection; refusing automatic cleanup')
    return {'schema':1,'port':PORT,'present':all(count == 1 for count in counts.values()),
            'directions':counts,'rules_sha256':hashlib.sha256(json.dumps(rules,sort_keys=True).encode()).hexdigest()}


def _boot_identity():
    return {'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'net_namespace':os.readlink('/proc/self/ns/net')}


def _write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(value,indent=2))
    temporary.replace(path)


def ensure_local_route(*, dry_run=False, receipt_path=None, runner=_run):
    """Return present/added rule receipt; existing rules are never claimed."""
    before = inspect_local_route(runner=runner)
    planned = [direction for direction,count in before['directions'].items() if count == 0]
    if dry_run:
        return {'status':'dry_run','before':before,
                'would_add':[['ip','rule','add',*_arguments(direction)] for direction in planned]}
    folder = ROOT/'artifacts/vllm_omni_migration/network_routes'
    folder.mkdir(parents=True,exist_ok=True)
    receipt_path = Path(receipt_path).resolve() if receipt_path else folder/(uuid.uuid4().hex+'.json')
    if not receipt_path.is_relative_to(folder.resolve()) or receipt_path.exists():
        raise ValueError('Use a new receipt inside the dedicated network_routes directory')
    with (folder/'lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        before = inspect_local_route(runner=runner)
        record = {'schema':1,'component':'cyberfly_loopback_route','port':PORT,
                  **_boot_identity(),'created_at':time.time(),'before':before,
                  'added_directions':[],'receipt_path':str(receipt_path)}
        try:
            for direction,count in before['directions'].items():
                if count: continue
                runner(['ip','rule','add',*_arguments(direction)])
                record['added_directions'].append(direction)
            record['after'] = inspect_local_route(runner=runner)
            if not record['after']['present']:
                raise RuntimeError('Both precise rules did not appear after creation')
            record['status'] = 'present_added' if record['added_directions'] else 'present_unchanged'
        except BaseException as exc:
            record['status'] = 'failed_rollback_attempted';record['error'] = str(exc)
            record['rollback_errors'] = []
            for direction in reversed(record['added_directions']):
                try: runner(['ip','rule','del',*_arguments(direction)])
                except Exception as rollback_exc: record['rollback_errors'].append(str(rollback_exc))
            raise
        finally:
            _write(receipt_path,record)
        return record


def rollback_local_route(receipt_path, *, runner=_run):
    path = Path(receipt_path).resolve()
    root = (ROOT/'artifacts/vllm_omni_migration/network_routes').resolve()
    if not path.is_relative_to(root) or path.stat().st_size > 65536:
        raise ValueError('Use an owned bounded network route receipt')
    with (root/'lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        record = json.loads(path.read_text())
        if (record.get('component') != 'cyberfly_loopback_route' or record.get('port') != PORT
                or any(record.get(k) != value for k,value in _boot_identity().items())):
            raise ValueError('Receipt does not own this service/boot/network namespace')
        added = record.get('added_directions')
        if not isinstance(added,list) or len(set(added)) != len(added) or any(v not in DIRECTIONS for v in added):
            raise ValueError('Invalid owned-rule list')
        if record.get('status') == 'rolled_back': return record
        if record.get('status') not in ('present_added','present_unchanged'):
            raise ValueError('Only a completed ensure receipt can be rolled back')
        current = inspect_local_route(runner=runner)
        record['removed_directions'] = []
        for direction in reversed(added):
            if not current['directions'][direction]: continue
            runner(['ip','rule','del',*_arguments(direction)])
            record['removed_directions'].append(direction)
            _write(path,record)
        record['status']='rolled_back';record['after_rollback']=inspect_local_route(runner=runner)
        _write(path,record)
        return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=['status','ensure','rollback'])
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--receipt')
    args=parser.parse_args()
    if args.operation == 'status': result=inspect_local_route()
    elif args.operation == 'ensure': result=ensure_local_route(dry_run=args.dry_run,receipt_path=args.receipt)
    else:
        if not args.receipt: parser.error('rollback requires its own --receipt')
        if args.dry_run: parser.error('rollback does not accept --dry-run')
        result=rollback_local_route(args.receipt)
    print(json.dumps(result,indent=2))


if __name__ == '__main__': main()
