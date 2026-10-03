"""CPU-only capacity admission; never acquires the resident-model training lock.

The caller that actually allocates training memory must hold training_capacity,
not merely call the preflight probe. A busy request is retryable, not consumed.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
LEASE_DIRECTORY = ROOT / 'model_training'
GPU_UUIDS = (
    'GPU-7c819eea-a864-1b0a-90b8-64854f446945',
    'GPU-ece2affe-2484-b8cb-1c7d-e3c6ceb9ec9c',
    'GPU-702d73d7-3075-1d30-a7fa-f899d3de0744',
    'GPU-551886a8-c863-1f65-3122-33ea29ca4097',
)
VERSION = 1


def _paths(gpu_uuid):
    if gpu_uuid not in GPU_UUIDS:
        raise ValueError('Capacity lease requires one explicit project GPU UUID')
    index = GPU_UUIDS.index(gpu_uuid)
    return (LEASE_DIRECTORY / f'gpu{index}-capacity.lock',
            LEASE_DIRECTORY / f'gpu{index}-capacity-owner.json')


def _read_owner(path):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def _write_owner(path, value):
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


class GPUCapacityBusy(RuntimeError):
    retryable = True

    def __init__(self, status):
        self.status = status
        role = (status.get('owner') or {}).get('purpose', 'another capacity owner')
        card = GPU_UUIDS.index(status['gpu_uuid']) if status.get('gpu_uuid') in GPU_UUIDS else '?'
        if role == 'native_code2wav':
            message = f'显卡{card}语音服务占用训练容量，样本保留排队；释放语音服务后继续。'
        elif role == 'language_training':
            message = f'显卡{card}正在训练，样本保留排队；当前训练结束后继续。'
        else:
            message = f'显卡{card}容量被其他任务占用，样本保留排队；释放容量后继续。'
        super().__init__(message)

    def to_dict(self):
        return {'status': 'capacity_busy', 'code': 'gpu_capacity_busy',
                'retryable': True, 'request_consumed': False, **self.status,
                'message': str(self)}


def inspect_gpu_capacity(gpu_uuid):
    lock_path, owner_path = _paths(gpu_uuid)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {'schema': VERSION, 'gpu_uuid': gpu_uuid, 'busy': True,
                    'owner': _read_owner(owner_path), 'lock_path': str(lock_path)}
        else:
            fcntl.flock(lock, fcntl.LOCK_UN)
            return {'schema': VERSION, 'gpu_uuid': gpu_uuid, 'busy': False,
                    'owner': None, 'lock_path': str(lock_path)}


def require_gpu_training_available(gpu_uuid):
    """Nonholding preflight only. The allocator must also use the context below."""
    status = inspect_gpu_capacity(gpu_uuid)
    if status['busy']:
        raise GPUCapacityBusy(status)
    return status


@contextmanager
def capacity_lease(gpu_uuid, *, purpose, reserve_mib=0):
    if purpose not in ('language_training', 'native_code2wav'):
        raise ValueError('Unknown project GPU capacity purpose')
    if type(reserve_mib) is not int or not 0 <= reserve_mib <= 24576:
        raise ValueError('Invalid declared capacity reservation')
    lock_path, owner_path = _paths(gpu_uuid)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise GPUCapacityBusy({'gpu_uuid': gpu_uuid, 'busy': True,
                'owner': _read_owner(owner_path), 'lock_path': str(lock_path)}) from None
        owner = {'schema': VERSION, 'token': uuid.uuid4().hex, 'pid': os.getpid(),
                 'gpu_uuid': gpu_uuid, 'purpose': purpose, 'reserve_mib': reserve_mib,
                 'started_at': time.time(), 'status': 'held',
                 'existing_resident_model_lock_modified': False}
        try:
            _write_owner(owner_path, owner)
            yield owner
        finally:
            try:
                if (_read_owner(owner_path) or {}).get('token') == owner['token']:
                    _write_owner(owner_path, {**owner, 'status': 'released', 'released_at': time.time()})
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)


def training_capacity(gpu_uuid):
    return capacity_lease(gpu_uuid, purpose='language_training')


def vocoder_capacity(gpu_uuid, reserve_mib=10240):
    return capacity_lease(gpu_uuid, purpose='native_code2wav', reserve_mib=reserve_mib)
