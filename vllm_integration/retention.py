"""Bounded raw evidence for one explicitly owned CPU pool/native flow.

Only exact files returned by that flow's producers are registered. There is no
directory scan, recursive deletion, model/GPU access, or training invocation.
Call ``protect`` before handing raw files to an asynchronous external consumer
(including an online learner), and release that lease only when it is finished.
Used hidden arrays and application/request receipts are never eligible.
"""
from __future__ import annotations

from collections import defaultdict, deque
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
_HEX = re.compile(r"[0-9a-f]{64}")
_UUID = re.compile(r"[0-9a-f]{32}")
_NAME = re.compile(r"[A-Za-z0-9_-]{1,128}")


def _sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def _plain_path(value):
    path = Path(value).absolute()
    # Do not follow symlinks even when their targets happen to be in artifacts.
    if path.resolve() != path or not path.is_relative_to(ROOT / 'artifacts'):
        raise ValueError('Retention requires an ordinary path inside project artifacts')
    return path


def _name(value):
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise ValueError('Use a bounded environment/event/application identity')
    return value


def _references(value):
    """Protect nested references without claiming ownership of those files."""
    paths, hot = set(), set()
    def collect(item):
        if isinstance(item, dict):
            if item.get('hot_storage') is not None:
                from connectome_adapter.hot_state import observation_snapshot_path
                hot.add(str(observation_snapshot_path(item)))
            for key, child in item.items():
                if key in ('path', 'packet_path', 'layout_path') and isinstance(child, str):
                    paths.add(Path(child).absolute())
                elif isinstance(child, (dict, list, tuple)):
                    collect(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                collect(child)
    collect(value)
    found = set(hot)
    # A condition may repeat a source in several descriptors. Resolve each
    # distinct spelling once, preserving both original and canonical leases.
    from .path_identity import resolve_path
    for path in paths:
        for candidate in (path, resolve_path(path, root=ROOT / 'artifacts')):
            if candidate.is_relative_to(ROOT / 'artifacts'):
                found.add(str(candidate))
    return found


class NativeRawRetention:
    """Keep 12 successful applications and two completed model turns per fly.

    Pending and retired turns stay live until ``complete_turn`` is explicitly
    called after their Future has actually finished. ``close`` is deliberately
    absent: cancellation/closing a flow does not prove a model reader stopped.
    This object is process-local and never adopts old files after restart.
    """
    def __init__(self, flow_directory, pool_directory, *, applications_per_env=12,
                 completed_turns_per_env=2):
        self.directory = _plain_path(flow_directory)
        self.pool_directory = _plain_path(pool_directory)
        if not self.directory.is_relative_to(ROOT / 'artifacts/vllm_omni_migration/flows'):
            raise ValueError('Retention must belong to an explicitly owned flow directory')
        if not self.directory.is_dir() or not self.pool_directory.is_dir():
            raise ValueError('Create the owned flow and pool before retention')
        for value in (applications_per_env, completed_turns_per_env):
            if type(value) is not int or not 1 <= value <= 1000:
                raise ValueError('Retention limits must be integers in [1,1000]')
        self.application_limit = applications_per_env
        self.turn_limit = completed_turns_per_env
        self._lock = threading.RLock()
        self._files, self._groups, self._pins = {}, {}, {}
        self._turns = {}
        self._applications = defaultdict(deque)
        self._completed_turns = defaultdict(deque)
        self._deleted_files = self._deleted_bytes = 0
        self._errors = []
        self._closed_hot = False
        self.journal = self.directory / 'raw-retention-receipts.jsonl'
        # Readers never acquire the transaction lock that protects file I/O.
        # Published dictionaries are replaced, never mutated after publication.
        self._status_lock = threading.Lock()
        self._status_sequence = 0
        self._published_status = None
        self._publish_status_locked()

    @contextmanager
    def _mutation(self, *, blocking=True):
        acquired = self._lock.acquire(blocking=blocking)
        if not acquired:
            yield False
            return
        try:
            yield True
        finally:
            try:
                # Also publish failure pins and errors; a rejected transaction
                # must not leave its new protection invisible to diagnostics.
                self._publish_status_locked()
            finally:
                self._lock.release()

    def _publish_status_locked(self):
        """Build under the mutation lock; exchange only a reference for readers."""
        self._status_sequence += 1
        value = {'applications_per_env': self.application_limit, 'completed_turns_per_env': self.turn_limit,
            'retained_applications': {key: len(value) for key, value in self._applications.items()},
            'retained_completed_turns': {key: len(value) for key, value in self._completed_turns.items()},
            'pending_turns': sum(value == 'pending' for value in self._turns.values()),
            'external_leases': len(self._pins), 'registered_raw_files': len(self._files),
            'registered_raw_bytes': sum(x['size'] for x in self._files.values()),
            'deleted_files': self._deleted_files, 'deleted_bytes': self._deleted_bytes,
            'errors': deepcopy(self._errors), 'audit_journal': str(self.journal),
            'retention_boundary': 'Only explicit producer-owned raw files; used hidden and small receipts remain permanent',
            'status_sequence': self._status_sequence,
            'status_semantics': 'Last published mutation outcome; may lag active disk work; never deletion authorization'}
        with self._status_lock:
            self._published_status = value

    def _receipt(self, value):
        with self.journal.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'schema': 1, 'at': time.time(), **value},
                                    ensure_ascii=False, allow_nan=False) + '\n')

    def _register(self, descriptor, *, allowed, kind):
        # Deliberately inspect only this descriptor's own path. A condition's
        # brain_state/body fields can point at another still-live turn.
        if not isinstance(descriptor, dict) or not _HEX.fullmatch(str(descriptor.get('sha256', ''))):
            raise ValueError('Owned raw files require their actual file SHA256')
        from .checked_artifacts import read_checked_artifact
        path = Path(descriptor['path']).absolute()
        hot = descriptor.get('hot_storage') is not None
        if str(path) != os.path.normpath(str(path)) or (not hot and not path.is_relative_to(ROOT / 'artifacts')):
            raise ValueError('Retention requires an ordinary path inside project artifacts')
        if not allowed(path):
            raise ValueError('Raw file is outside this producer\'s exact ownership scope: ' + str(path))
        if path.name.startswith('used-hidden-') or path.name.startswith(('application-', 'request-source')):
            raise ValueError('Used hidden and audit receipts are permanently retained')
        # Kernel-confined open, bounded regular-file read and complete SHA are
        # one operation. No file bytes or digests are cached across consumers.
        if hot:
            from connectome_adapter.hot_state import read_observation_snapshot
            resolved, raw = read_observation_snapshot(descriptor)
        else:
            resolved, raw = read_checked_artifact(descriptor, root=ROOT / 'artifacts', limit=128 * 1024 * 1024)
        if resolved != path:
            raise ValueError('Retention requires a canonical path without symbolic links')
        digest = descriptor['sha256']
        key = str(path)
        previous = self._files.get(key)
        if previous and previous['sha256'] != digest:
            raise ValueError('Previously owned immutable evidence changed')
        self._files[key] = {'path': key, 'sha256': digest, 'size': len(raw), 'kind': kind}
        if hot:
            self._files[key]['hot_descriptor'] = deepcopy(descriptor)
        return key

    def _observation(self, env_id, record, *, kind):
        if not isinstance(record, dict) or record.get('env_id') != env_id or record.get('kind') != kind:
            raise ValueError('Observation does not belong to this environment/producer')
        tag = record.get('record_id')
        if not isinstance(tag, str) or not _UUID.fullmatch(tag):
            raise ValueError('Expected the pool\'s original immutable observation ID')
        folder = self.pool_directory / env_id / ('inputs' if kind == 'input' else 'feedback') / tag
        paths = set()
        expected = {'rgb_file': 'rgb.npy', 'body_packet_file': 'body_packet.json'}
        for key, filename in expected.items():
            paths.add(self._register(record[key], allowed=lambda p, f=filename: p == folder / f, kind=kind))
        if record.get('neural_state_file'):
            descriptor = record['neural_state_file']
            if descriptor.get('hot_storage') is not None:
                from connectome_adapter.hot_state import observation_snapshot_path
                expected = observation_snapshot_path(descriptor, expected_owner=self.pool_directory / env_id,
                    expected_kind=kind, expected_record_id=tag)
                paths.add(self._register(descriptor, allowed=lambda p: p == expected, kind=kind))
            else:
                paths.add(self._register(descriptor, allowed=lambda p:
                    p.parent == folder and re.fullmatch(r'state-[0-9a-f]{32}\.npz', p.name), kind=kind))
        return paths

    def register_turn(self, env_id, event_id, observation, *, created_descriptors=(), created_files=()):
        """Register actual capture + explicitly created files; protects pending reads.

        Pass ``condition`` among created_descriptors: nested sources become
        *references*, never owned deletion targets. Body export packet/layout
        sidecars can be passed explicitly in created_files.
        """
        env_id, event_id = _name(env_id), _name(event_id)
        key = ('turn', env_id, event_id)
        folder = self.directory / env_id / event_id
        condition_folder = ROOT / 'artifacts/vllm_latent' / hashlib.sha256(event_id.encode()).hexdigest()[:24]
        with self._mutation():
            if key in self._turns:
                raise ValueError('Turn was already registered')
            before = set(self._files)
            paths = set()
            try:
                paths.update(self._observation(env_id, observation, kind='input'))
                for descriptor in created_descriptors:
                    paths.add(self._register(descriptor, allowed=lambda p:
                        (p.is_relative_to(folder) and p.name == 'features.npz') or
                        (p.parent == condition_folder and re.fullmatch(r'soft-[0-9a-f]{32}\.npz', p.name)),
                        kind='model_observation'))
                for filename in created_files:
                    from .checked_artifacts import read_bounded_artifact
                    path = Path(filename).absolute()
                    allowed = lambda p: p.is_relative_to(folder) and p.name in (
                        'input.png', 'body_packet.json', 'feature_layout.json')
                    if (str(path) != os.path.normpath(str(path)) or
                            not path.is_relative_to(ROOT / 'artifacts') or not allowed(path)):
                        raise ValueError('Expected this producer exact ordinary artifact')
                    resolved, raw = read_bounded_artifact(path, root=ROOT / 'artifacts', limit=128 * 1024 * 1024)
                    if resolved != path:
                        raise ValueError('Expected canonical path without symbolic links')
                    # Independent second full read/SHA check is intentionally
                    # retained, including mutations between the two reads.
                    paths.add(self._register({'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()},
                        allowed=allowed, kind='model_observation'))
                references = _references(created_descriptors)
            except Exception:
                # Registration failure cannot make partially registered inputs
                # eligible for deletion while their consumer may be running.
                self._pins['registration-failed:' + env_id + ':' + event_id] = paths | (set(self._files) - before)
                raise
            self._groups[key] = paths | references
            self._turns[key] = 'pending'
            self._receipt({'operation': 'register_turn', 'env_id': env_id, 'event_id': event_id,
                           'owned_files': [self._files[p] for p in sorted(paths)]})

    def complete_application(self, env_id, event_id, application_id, *, current_descriptors,
                             feedback, actual_sum_verified):
        """Register only after a verified body/brain step; never register budget snapshots.

        The original model vector and budgeted applied vector are explicit
        producer returns. Temporary budget snapshots are worker-owned and are
        already removed by the worker; this method does not inspect them.
        """
        env_id, event_id, application_id = _name(env_id), _name(event_id), _name(str(application_id))
        if actual_sum_verified is not True:
            raise ValueError('Raw application retention requires verified actual current delivery')
        key = ('application', env_id, application_id)
        with self._mutation():
            if ('turn', env_id, event_id) not in self._turns:
                raise ValueError('Application must belong to a registered turn')
            if key in self._groups:
                raise ValueError('Application was already completed')
            before = set(self._files)
            paths = set()
            try:
                for descriptor in current_descriptors:
                    paths.add(self._register(descriptor, allowed=lambda p:
                        p.is_relative_to(ROOT / 'artifacts/neuron_currents') and
                        re.fullmatch(r'currents-[0-9a-f]{32}\.npz', p.name), kind='applied_current'))
                if not paths:
                    raise ValueError('An applied current descriptor is required')
                paths.update(self._observation(env_id, feedback, kind='feedback'))
            except Exception:
                self._pins['application-registration-failed:' + env_id + ':' + application_id] = paths | (set(self._files) - before)
                raise
            self._groups[key] = paths
            self._applications[env_id].append(key)
            while len(self._applications[env_id]) > self.application_limit:
                self._groups.pop(self._applications[env_id].popleft())
            self._receipt({'operation': 'complete_application', 'env_id': env_id,
                'event_id': event_id, 'application_id': application_id,
                'actual_sum_verified': True, 'owned_files': [self._files[p] for p in sorted(paths)]})

    def complete_turn(self, env_id, event_id, *, future_done):
        """Only after actual completion, including a retired/cancelled Future."""
        key = ('turn', _name(env_id), _name(event_id))
        if future_done is not True:
            raise ValueError('Pending/retired model readers cannot be released')
        with self._mutation():
            if key not in self._turns:
                raise ValueError('Unknown model turn')
            if self._turns[key] == 'completed':
                return
            self._turns[key] = 'completed'
            self._completed_turns[env_id].append(key)
            while len(self._completed_turns[env_id]) > self.turn_limit:
                stale = self._completed_turns[env_id].popleft()
                self._groups.pop(stale)
                self._turns.pop(stale)
            self._receipt({'operation': 'complete_turn', 'env_id': env_id, 'event_id': event_id})

    def protect(self, owner, descriptors):
        """Replace one lease; call before publishing raw paths to other readers."""
        if not isinstance(owner, str) or not owner or len(owner) > 256:
            raise ValueError('A bounded named consumer lease is required')
        paths = _references(descriptors)
        with self._mutation():
            self._pins[owner] = paths

    def register_observation_copy(self, env_id, observation, *, consumer):
        """Own an extra pool capture only for a verified learner-copy lease.

        Return the lease key. After a successful independent copy, ``release``
        it and call ``cleanup``. Other model/application references still win.
        Caller must establish that this capture was created by its own pool
        operation, rather than adopting an arbitrary observation archive.
        """
        env_id = _name(env_id)
        if not isinstance(consumer, str) or not consumer or len(consumer) > 220:
            raise ValueError('A bounded unique observation-copy consumer is required')
        lease = 'observer_copy:' + consumer
        kind = observation.get('kind') if isinstance(observation, dict) else None
        if kind not in ('input', 'feedback'):
            raise ValueError('Expected an actual input/feedback observation')
        with self._mutation():
            if lease in self._pins:
                raise ValueError('Observation-copy lease is already active')
            before = set(self._files)
            try:
                paths = self._observation(env_id, observation, kind=kind)
            except Exception:
                self._pins[lease] = set(self._files) - before
                raise
            self._pins[lease] = paths
            self._receipt({'operation': 'register_observation_copy', 'env_id': env_id,
                           'consumer': consumer, 'lease': lease,
                           'owned_files': [self._files[p] for p in sorted(paths)]})
        return lease

    def release(self, owner):
        with self._mutation():
            self._pins.pop(owner, None)
            closed = self._closed_hot
        if closed:
            self.cleanup()

    def discard_unsubmitted_observation(self, env_id, observation):
        """Only when the caller proves no model Future was submitted for it."""
        lease = self.register_observation_copy(env_id, observation,
            consumer='unsubmitted-' + observation['record_id'])
        with self._mutation():
            paths = self._pins.pop(lease, set())
            for key in list(self._turns):
                if key[1] == env_id and self._turns[key] == 'pending' and paths & self._groups.get(key, set()):
                    self._groups.pop(key, None)
                    self._turns.pop(key, None)
            # _prepare registers the turn before submitting HTTP. If that
            # registration failed, no asynchronous consumer owns these files.
            for key in list(self._pins):
                if key.startswith('registration-failed:' + env_id + ':'):
                    self._pins.pop(key)
        return self.cleanup()

    def close_after_futures(self, futures):
        """Release ephemeral hot snapshots only after all old HTTP readers end.

        The callback does no physics/model work. Persistent artifact retention
        remains unchanged; external learner-copy leases still protect raw data.
        """
        futures = tuple(futures)
        def finish(_=None):
            if not all(future.done() for future in futures):
                return
            with self._mutation():
                if self._closed_hot:
                    return
                self._closed_hot = True
                hot = {key for key, entry in self._files.items() if entry.get('hot_descriptor')}
                for values in self._groups.values():
                    values.difference_update(hot)
                for name, values in self._pins.items():
                    if name.startswith(('previous_body:', 'registration-failed:', 'application-registration-failed:')):
                        values.difference_update(hot)
                self._receipt({'operation': 'close_hot_after_model_readers_finished',
                               'hot_paths': sorted(hot), 'external_copy_leases_preserved': True})
            self.cleanup()
        def dispatch(_=None):
            # Future.add_done_callback executes synchronously if already done.
            # Always hand cleanup to its own bounded CPU callback thread.
            threading.Thread(target=finish, name='cyberfly-hot-retention-close', daemon=True).start()
        for future in futures:
            future.add_done_callback(dispatch)
        if not futures:
            dispatch()

    def cleanup(self, *, blocking=True):
        """Prune unreferenced registered files only; leave all directories intact."""
        if type(blocking) is not bool:
            raise ValueError('Cleanup blocking must be an explicit boolean')
        with self._mutation(blocking=blocking) as acquired:
            if not acquired:
                return {'skipped': True, 'reason': 'retention_mutation_in_progress',
                        'deleted_files': 0, 'deleted_bytes': 0, 'errors': []}
            protected = set().union(*self._groups.values(), *self._pins.values())
            removed, errors = [], []
            for key, entry in list(self._files.items()):
                if key in protected:
                    continue
                try:
                    if entry.get('hot_descriptor'):
                        from connectome_adapter.hot_state import remove_observation_snapshot
                        descriptor = entry['hot_descriptor']
                        remove_observation_snapshot(descriptor,
                            expected_owner=self.pool_directory / descriptor['hot_storage']['env_id'])
                        self._files.pop(key)
                        self._deleted_files += 1
                        self._deleted_bytes += entry['size']
                        removed.append(entry)
                        continue
                    path = _plain_path(key)
                    if not path.exists():
                        errors.append({'path': key, 'error': 'Registered raw evidence is already missing'})
                        self._files.pop(key)
                        continue
                    if not path.is_file() or path.stat().st_size != entry['size'] or _sha(path) != entry['sha256']:
                        raise ValueError('Raw file changed since registration; refused to delete')
                    path.unlink()
                    self._files.pop(key)
                    self._deleted_files += 1
                    self._deleted_bytes += entry['size']
                    removed.append(entry)
                except (OSError, ValueError) as exc:
                    errors.append({'path': key, 'error': str(exc)})
            if removed or errors:
                self._receipt({'operation': 'cleanup', 'removed': removed, 'errors': errors,
                               'used_hidden_and_receipts_preserved': True})
            self._errors = errors[-20:]
            return {'deleted_files': len(removed), 'deleted_bytes': sum(x['size'] for x in removed), 'errors': errors}

    def status(self):
        with self._status_lock:
            value = self._published_status
        return deepcopy(value)
