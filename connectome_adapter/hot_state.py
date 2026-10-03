"""Owned Linux-local snapshots used only while budgeting one neural input.

This is not an archive root or a general model-input root. Full observations,
training examples and permanent exports still use their existing locations.
Only this module's exact, process-owned files are eligible for cleanup.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
HOT_ROOT = Path('/root/.cache/cyberfly/brain-state')
_UUID = re.compile(r'[0-9a-f]{32}')
_STATE = re.compile(r'state-[0-9a-f]{32}\.npz')
_owners = {}
_lock = threading.Lock()


def _private_directory(path, *, create=False):
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode) or path.resolve(strict=True) != path or
            info.st_uid != os.getuid() or info.st_mode & 0o077):
        raise ValueError('Hot brain state requires an owned private ordinary directory')
    return path


def _owner(directory):
    directory = Path(directory).resolve(strict=True)
    if not directory.is_dir() or not directory.is_relative_to(ROOT / 'artifacts'):
        raise ValueError('Budget owner must be an existing project artifact directory')
    key = (os.getpid(), str(directory))
    with _lock:
        if key not in _owners:
            _private_directory(HOT_ROOT, create=True)
            identity = uuid.uuid4().hex
            folder = HOT_ROOT / identity
            folder.mkdir(mode=0o700, exist_ok=False)
            budget = folder / 'budget'
            budget.mkdir(mode=0o700)
            _owners[key] = {'owner_id': identity, 'owner_directory': directory,
                            'folder': folder, 'budget': budget, 'active': {}}
        return _owners[key]


def _hot_path(descriptor):
    storage = descriptor.get('budget_storage')
    if (not isinstance(storage, dict) or storage.get('schema') != 1 or
            storage.get('kind') != 'linux_hot_budget' or storage.get('archive') is not False or
            storage.get('root') != str(HOT_ROOT)):
        raise ValueError('Hot snapshot requires its explicit temporary budget identity')
    owner_id, lease_id = storage.get('owner_id'), storage.get('lease_id')
    if not isinstance(owner_id, str) or not _UUID.fullmatch(owner_id) or not isinstance(lease_id, str) or not _UUID.fullmatch(lease_id):
        raise ValueError('Invalid hot snapshot owner or lease identity')
    path = Path(descriptor.get('path', ''))
    expected = HOT_ROOT / owner_id / 'budget' / lease_id
    if not path.is_absolute() or path.parent != expected or not _STATE.fullmatch(path.name):
        raise ValueError('Hot snapshot path differs from its exact owner and lease')
    for directory in (HOT_ROOT, HOT_ROOT / owner_id, HOT_ROOT / owner_id / 'budget', expected):
        _private_directory(directory)
    return path


def read_budget_snapshot(descriptor):
    """Complete SHA validation; exactly two roots, only for input budgeting."""
    from vllm_integration.checked_artifacts import read_checked_artifact
    if descriptor.get('hot_storage') is not None:
        if descriptor.get('budget_storage') is not None:
            raise ValueError('Ambiguous hot snapshot storage identity')
        return read_observation_snapshot(descriptor)
    path = Path(descriptor.get('path', ''))
    if descriptor.get('budget_storage') is not None:
        path = _hot_path(descriptor)
        root = HOT_ROOT
    else:
        root = ROOT / 'artifacts'
    if path.suffix != '.npz':
        raise ValueError('Neural input budget requires a complete NPZ snapshot')
    return read_checked_artifact(descriptor, root=root, limit=64 * 1024 * 1024)


def _observation_owner(directory):
    directory = Path(directory).resolve(strict=True)
    roots = (ROOT / 'artifacts/environment_pool', ROOT / 'artifacts/vllm_omni_migration/session-pools')
    if not any(directory.is_relative_to(root) for root in roots):
        raise ValueError('Hot observations require an explicitly owned environment pool directory')
    owner = _owner(directory)
    if 'registration' not in owner:
        observations = owner['folder'] / 'observations'
        observations.mkdir(mode=0o700)
        for kind in ('input', 'feedback'):
            (observations / kind).mkdir(mode=0o700)
        registration = {'schema': 1, 'kind': 'linux_hot_observation_owner',
            'owner_id': owner['owner_id'], 'owner_artifact_directory': str(directory),
            'root': str(HOT_ROOT), 'producer_pid': os.getpid(), 'uid': os.getuid()}
        path = directory / ('hot-owner-' + owner['owner_id'] + '.json')
        raw = json.dumps(registration, sort_keys=True).encode()
        with path.open('xb') as stream: stream.write(raw)
        owner['registration'] = {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}
        owner['observations'] = observations
    return owner


def capture_observation_snapshot(brain, owner_directory, *, kind, record_id, env_id):
    """Write a real unadvanced snapshot into one explicitly issued observation lease."""
    if kind not in ('input', 'feedback') or not isinstance(record_id, str) or not _UUID.fullmatch(record_id):
        raise ValueError('Use an exact input/feedback record identity')
    if not isinstance(env_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', env_id):
        raise ValueError('Invalid observation environment identity')
    directory = Path(owner_directory).resolve(strict=True)
    roots = (ROOT / 'artifacts/environment_pool', ROOT / 'artifacts/vllm_omni_migration/session-pools')
    if not any(directory.is_relative_to(root) for root in roots):
        # Explicit legacy/custom artifact pools keep their original permanent
        # exports. They do not expand the trusted Linux observation roots.
        if not directory.is_relative_to(ROOT / 'artifacts'):
            raise ValueError('Observation owner is outside project artifacts')
        return brain.snapshot(directory / ('inputs' if kind == 'input' else 'feedback') / record_id,
                              include_graph=False)
    owner = _observation_owner(owner_directory)
    if owner['owner_directory'].name != env_id:
        raise ValueError('Observation environment differs from its owner directory')
    folder = owner['observations'] / kind / record_id
    folder.mkdir(mode=0o700, exist_ok=False)
    descriptor = brain.snapshot(folder, include_graph=False)
    path = Path(descriptor['path'])
    if path.parent != folder or not _STATE.fullmatch(path.name):
        raise ValueError('Snapshot producer returned a foreign observation file')
    lease = {'schema': 1, 'owner_id': owner['owner_id'], 'record_id': record_id,
             'env_id': env_id, 'kind': kind, 'path': str(path), 'sha256': descriptor['sha256']}
    raw = json.dumps(lease, sort_keys=True).encode()
    lease_path = folder / 'lease.json'
    with lease_path.open('xb') as stream: stream.write(raw)
    descriptor['hot_storage'] = {'schema': 1, 'kind': 'linux_hot_observation',
        'root': str(HOT_ROOT), 'owner_id': owner['owner_id'], 'record_id': record_id,
        'observation_kind': kind, 'env_id': env_id, 'archive': False,
        'owner_artifact_directory': str(owner['owner_directory']),
        'registration': dict(owner['registration']),
        'lease': {'path': str(lease_path), 'sha256': hashlib.sha256(raw).hexdigest()},
        'retention': 'Only exact pool/record leases; persist before training, never reuse after release'}
    descriptor['temporary'] = True
    return descriptor


def observation_snapshot_path(descriptor, *, expected_owner=None, expected_kind=None, expected_record_id=None):
    """Authenticate the exact registered owner and issued file; no global root relaxation."""
    from vllm_integration.checked_artifacts import read_checked_artifact
    storage = descriptor.get('hot_storage')
    if (not isinstance(storage, dict) or storage.get('schema') != 1 or
            storage.get('kind') != 'linux_hot_observation' or storage.get('archive') is not False or
            storage.get('root') != str(HOT_ROOT) or descriptor.get('budget_storage') is not None):
        raise ValueError('Expected explicit registered hot observation storage')
    owner_id, record_id, kind = storage.get('owner_id'), storage.get('record_id'), storage.get('observation_kind')
    if (not isinstance(owner_id, str) or not _UUID.fullmatch(owner_id) or
            not isinstance(record_id, str) or not _UUID.fullmatch(record_id) or kind not in ('input', 'feedback')):
        raise ValueError('Invalid hot observation owner/record identity')
    owner_directory = Path(storage.get('owner_artifact_directory', ''))
    roots = (ROOT / 'artifacts/environment_pool', ROOT / 'artifacts/vllm_omni_migration/session-pools')
    if (not owner_directory.is_absolute() or owner_directory.resolve() != owner_directory or
            not any(owner_directory.is_relative_to(root) for root in roots) or
            owner_directory.name != storage.get('env_id')):
        raise ValueError('Hot observation producer is outside registered pool roots')
    if expected_owner is not None and owner_directory != Path(expected_owner).absolute():
        raise ValueError('Hot observation belongs to another pool/environment owner')
    if expected_kind is not None and kind != expected_kind or expected_record_id is not None and record_id != expected_record_id:
        raise ValueError('Hot observation belongs to another actual capture')
    registration = storage.get('registration', {})
    if registration.get('path') != str(owner_directory / ('hot-owner-' + owner_id + '.json')):
        raise ValueError('Hot observation registration path changed')
    _, raw = read_checked_artifact(registration, root=ROOT / 'artifacts', limit=8192)
    registered = json.loads(raw)
    for key, value in {'schema': 1, 'kind': 'linux_hot_observation_owner', 'owner_id': owner_id,
                       'root': str(HOT_ROOT), 'owner_artifact_directory': str(owner_directory), 'uid': os.getuid()}.items():
        if registered.get(key) != value:
            raise ValueError('Hot observation owner registration differs: ' + key)
    folder = HOT_ROOT / owner_id / 'observations' / kind / record_id
    for directory in (HOT_ROOT, HOT_ROOT / owner_id, HOT_ROOT / owner_id / 'observations', folder.parent, folder):
        _private_directory(directory)
    path = Path(descriptor.get('path', ''))
    if path.parent != folder or not _STATE.fullmatch(path.name):
        raise ValueError('Hot observation file differs from its exact owner/record')
    lease = storage.get('lease', {})
    if lease.get('path') != str(folder / 'lease.json'):
        raise ValueError('Hot observation lease path differs')
    _, raw = read_checked_artifact(lease, root=HOT_ROOT / owner_id, limit=8192)
    expected = {'schema': 1, 'owner_id': owner_id, 'record_id': record_id,
                'env_id': storage['env_id'], 'kind': kind, 'path': str(path), 'sha256': descriptor['sha256']}
    if json.loads(raw) != expected:
        raise ValueError('Hot observation file does not match its issued lease')
    return path


def read_observation_snapshot(descriptor):
    """Full SHA on every read, either old artifacts or an exact registered Linux lease."""
    from vllm_integration.checked_artifacts import read_checked_artifact
    if descriptor.get('hot_storage') is not None:
        path = observation_snapshot_path(descriptor)
        root = HOT_ROOT / descriptor['hot_storage']['owner_id']
    else:
        if descriptor.get('budget_storage') is not None:
            raise ValueError('Temporary budgeting snapshots are not shared observations')
        path = Path(descriptor.get('path', ''))
        root = ROOT / 'artifacts'
    if path.suffix != '.npz':
        raise ValueError('Expected a complete neural observation NPZ')
    return read_checked_artifact(descriptor, root=root, limit=64 * 1024 * 1024)


def remove_observation_snapshot(descriptor, *, expected_owner):
    """Retention calls only after all model/learner leases have been released."""
    path = observation_snapshot_path(descriptor, expected_owner=expected_owner)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.stat(path.name, dir_fd=fd, follow_symlinks=False)
        read_observation_snapshot(descriptor)
        after = os.stat(path.name, dir_fd=fd, follow_symlinks=False)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError('Hot observation changed before release; retained for diagnosis')
        os.unlink(path.name, dir_fd=fd)
        os.unlink('lease.json', dir_fd=fd)
    finally:
        os.close(fd)
    path.parent.rmdir()


def _release(owner, descriptor):
    path = _hot_path(descriptor)
    lease_id = descriptor['budget_storage']['lease_id']
    if owner['active'].get(lease_id) is not descriptor or path.parent.parent != owner['budget']:
        raise ValueError('Refused to release a snapshot not issued to this process owner')
    _private_directory(path.parent)
    # Pin the exact directory during validation and unlink. No recursive scan,
    # adoption of old files, or deletion through an untrusted descriptor.
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
        read_budget_snapshot(descriptor)
        after = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError('Hot snapshot changed before release; retained for diagnosis')
        os.unlink(path.name, dir_fd=directory_fd)
    finally:
        os.close(directory_fd)
    path.parent.rmdir()
    owner['active'].pop(lease_id)


def prepare_temporary_stimulation(brain, owner_directory, model_currents_descriptor,
                                  legacy_general, actual_ports):
    """Budget one input synchronously, release raw data before physics starts.

    The receipt explicitly describes the deleted temporary source. A later
    actual-current comparison must still verify delivery in the ordinary caller.
    Neither this function nor the snapshot operation advances the brain.
    """
    from shared_io.parallel_inputs import prepare_parallel_stimulation
    owner = _owner(owner_directory)
    lease_id = uuid.uuid4().hex
    directory = owner['budget'] / lease_id
    directory.mkdir(mode=0o700, exist_ok=False)
    descriptor = None
    try:
        descriptor = brain.snapshot(directory, include_graph=False)
        descriptor['budget_storage'] = {
            'schema': 1, 'kind': 'linux_hot_budget', 'root': str(HOT_ROOT),
            'owner_id': owner['owner_id'], 'lease_id': lease_id,
            'owner_artifact_directory': str(owner['owner_directory']),
            'producer_pid': os.getpid(), 'created_at': time.time(), 'archive': False}
        descriptor['temporary'] = True
        _hot_path(descriptor)
        owner['active'][lease_id] = descriptor
        result = prepare_parallel_stimulation(brain, descriptor, model_currents_descriptor,
                                              legacy_general, actual_ports)
        source = {key: descriptor[key] for key in ('path', 'sha256', 'ids_sha256',
            'simulation_ms', 'duration_ms', 'neurons')}
        source.update(temporary=True, budget_storage=dict(descriptor['budget_storage']),
                      raw_snapshot_archived=False)
        result['budget_source'] = source
    except BaseException as error:
        if descriptor is not None and owner['active'].get(lease_id) is descriptor:
            try:
                _release(owner, descriptor)
            except BaseException as cleanup_error:
                error.add_note('Hot snapshot retained after cleanup refusal: ' + str(cleanup_error))
        elif not any(directory.iterdir()):
            directory.rmdir()
        raise
    _release(owner, descriptor)
    source.update(removed_after_budget_consumer_completed=True,
                  removed_before_physical_step=True)
    return result


def _close_empty_owners():
    # Active/failed snapshots are intentionally never guessed safe on exit.
    for owner in list(_owners.values()):
        if owner['active']:
            continue
        try:
            owner['budget'].rmdir()
            owner['folder'].rmdir()
        except OSError:
            pass


atexit.register(_close_empty_owners)
