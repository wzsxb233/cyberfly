"""Omni's pool protocol over the *existing* Lab actor's body and brain.

There are no worker processes, resets, model requests or new environments on
construction. Every physical operation is confined to the Lab owner thread.
NativeOmniFlow owns background HTTP futures; this facade only applies actual
selected currents, alongside the original senses, through LabSession._step.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import threading
import time
import uuid
import wave

import numpy as np

from .environment_pool import ROOT, _artifact_path, _digest, _json


class SessionEnvironmentPool:
    """Actor-only single-body implementation of NativeOmniFlow's pool API."""

    def __init__(self, session, *, directory=None):
        if session.thread.ident != threading.get_ident():
            raise RuntimeError('Attach to the Lab from its simulation owner thread.')
        env = session.env
        if (env is None or not env.scenario_spec.get('neural_input') or
                env.scenario_spec.get('embodiment') != 'physical_3d' or
                env.action_space.shape != (3,) or env.brain is None):
            raise ValueError('Omni Lab flow currently supports a physical neural body with three original sensory ports.')
        if session.done:
            raise ValueError('Finish the normal Lab episode reset before attaching a flow.')
        self.session, self.env, self.brain = session, env, env.brain
        self.owner_id = threading.get_ident()
        self.env_id = 'lab'
        self.workers = {self.env_id: None}  # Protocol identity; never a second worker.
        self.directory = _artifact_path(directory or ROOT / 'artifacts/vllm_omni_migration/session-pools' / uuid.uuid4().hex[:12])
        self.directory.mkdir(parents=True, exist_ok=False)
        self.worker_directory = self.directory / self.env_id
        self.worker_directory.mkdir()
        self.started, self.closed = True, False
        self.input_record = None
        self._cached = {}
        self._refresh_status()

    def _owner(self):
        if threading.get_ident() != self.owner_id:
            raise RuntimeError('Physics and brain access must stay on the Lab owner thread.')
        if self.closed:
            raise RuntimeError('This Lab facade has detached.')
        if self.session.env is not self.env or self.env.brain is not self.brain:
            raise RuntimeError('The Lab environment or brain changed; detach the old Omni flow first.')

    def _ids(self, env_ids):
        self._owner()
        ids = [self.env_id] if env_ids is None else list(env_ids)
        if len(set(ids)) != len(ids) or any(key != self.env_id for key in ids):
            raise ValueError('This facade addresses only the current Lab body.')
        return ids

    def _refresh_status(self):
        info = self.env.task_state()
        neural = info.get('neural') or {}
        telemetry = neural.get('neural', {})
        self._cached = {'env_id': self.env_id, 'pid': os.getpid(),
            'brain_pid': self.brain.process.pid, 'seed': 42 + self.session.episode,
            'episode': self.session.episode, 'phase': info.get('phase'),
            'scenario': info.get('scenario'), 'step': info.get('step'),
            'frame_seq': self.session.frame_seq, 'brain_neurons': info.get('brain_neurons'),
            'brain_edges': info.get('brain_edges'), 'simulation_ms': telemetry.get('simulation_ms', 0),
            'spikes_this_step': telemetry.get('spikes_this_step', 0),
            'total_spikes': telemetry.get('total_spikes', 0),
            'body_interval_ms': info.get('body_interval_ms'), 'neural_interval_ms': info.get('neural_interval_ms'),
            'body_action': info.get('body_action'), 'brain_learning': info.get('brain_learning'),
            'visual_input_enabled': info.get('visual_input_enabled'),
            'input_rgb_sha256': neural.get('input_rgb_sha256'),
            'pending_input_id': self.input_record['input_id'] if self.input_record else None,
            'scope': 'existing_lab_environment', 'owns_environment': False, 'alive': not self.closed}

    def status(self):
        """A read never touches MuJoCo or the brain process."""
        return {'schema': 1, 'started': self.started, 'closed': self.closed,
                'selected_env_id': self.env_id, 'directory': str(self.directory),
                'model_invocation': False, 'owns_environment': False,
                'workers': {self.env_id: deepcopy(self._cached)}}

    def _attach_neural_state(self, record):
        if 'neural_state_file' not in record:
            from connectome_adapter.hot_state import capture_observation_snapshot
            record['neural_state_file'] = capture_observation_snapshot(self.brain, self.worker_directory,
                kind=record['kind'], record_id=record['record_id'], env_id=self.env_id)
        return record

    def _save_observation(self, kind, include_neural_state):
        if kind == 'input':
            rgb = self.env.shared_input_rgb()
            packet = self.env.shared_body_senses()
        else:
            # Rendering an after-state must not pin the next neural input.
            rgb = self.env._frame.copy()
            packet = self.env.body.unwrapped.body_sensory_packet(rgb=rgb)
        tag = uuid.uuid4().hex
        folder = self.worker_directory / ('inputs' if kind == 'input' else 'feedback') / tag
        folder.mkdir(parents=True)
        np.save(folder / 'rgb.npy', rgb, allow_pickle=False)
        (folder / 'body_packet.json').write_text(_json(packet))
        raw_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
        if packet['vision']['rgb_sha256'] != raw_sha:
            raise RuntimeError('Actual body packet and camera identity differ.')
        record = {'schema': 1, 'kind': kind, 'record_id': tag, 'env_id': self.env_id,
            'session_id': self.session.session_id, 'episode': self.session.episode,
            'frame_seq': self.session.frame_seq, 'body_simulation_time_s': packet['simulation_time_s'],
            'rgb_file': {'path': str(folder / 'rgb.npy'), 'sha256': _digest(folder / 'rgb.npy'),
                'raw_rgb_sha256': raw_sha, 'shape': list(rgb.shape), 'dtype': str(rgb.dtype)},
            'body_packet_file': {'path': str(folder / 'body_packet.json'),
                'sha256': _digest(folder / 'body_packet.json'), 'packet_sha256': packet['packet_sha256']},
            'model_invocation': False, 'scope': 'existing_lab_environment'}
        if include_neural_state:
            self._attach_neural_state(record)
        return record

    def capture_inputs(self, env_ids=None, *, include_neural_state=False):
        ids = self._ids(env_ids)
        if type(include_neural_state) is not bool:
            raise ValueError('include_neural_state must be bool.')
        if not ids:
            return {}
        if self.session.done:
            raise ValueError('Reset the completed episode before capturing its next input.')
        if self.input_record is None:
            self.input_record = self._save_observation('input', include_neural_state)
            self.input_record['input_id'] = self.input_record['record_id']
        elif include_neural_state:
            self._attach_neural_state(self.input_record)
        self._refresh_status()
        return {self.env_id: deepcopy(self.input_record)}

    def capture_feedback(self, env_ids=None, *, include_neural_state=False):
        ids = self._ids(env_ids)
        if type(include_neural_state) is not bool:
            raise ValueError('include_neural_state must be bool.')
        return {key: self._save_observation('feedback', include_neural_state) for key in ids}

    def step_many(self, commands):
        self._owner()
        if not isinstance(commands, dict) or set(commands) != {self.env_id}:
            raise ValueError('Submit exactly one command for the current Lab body.')
        return {self.env_id: self._step(commands[self.env_id])}

    def _step(self, command):
        allowed = {'action', 'input_id', 'general_stimulation', 'neuron_currents_file',
                   'model_currents_file', 'aversive', 'capture_neural_state'}
        if not isinstance(command, dict) or set(command) - allowed:
            raise ValueError(f'Step supports only {sorted(allowed)}.')
        if self.session.done:
            raise ValueError('The completed Lab episode must reset before another step.')
        record = self.input_record
        if 'input_id' in command and (record is None or command['input_id'] != record['input_id']):
            raise ValueError('Stale or foreign input_id; no brain/body advance occurred.')
        if record is not None:
            if 'input_id' not in command:
                raise ValueError('Submit the pinned input_id before advancing this body.')
            if (record['episode'] != self.session.episode or
                    record['body_simulation_time_s'] != float(self.env.body.unwrapped.sim.mj_data.time)):
                raise ValueError('Body advanced outside the pinned interval; discard the old flow.')
        action = np.asarray(command.get('action', self.env.shared_passthrough_action()), dtype=np.float32)
        if not self.env.action_space.contains(action):
            raise ValueError('Action is outside the existing neural input space.')
        for field in ('aversive', 'capture_neural_state'):
            if type(command.get(field, False)) is not bool:
                raise ValueError(field + ' must be bool.')
        if 'model_currents_file' in command and 'neuron_currents_file' in command:
            raise ValueError('Choose budgeted model input or explicit experimental neuron currents, not both.')
        from connectome_adapter.ports import validate_general_stimulation
        original = validate_general_stimulation(command.get('general_stimulation', self.env.general_stimulation))
        parallel = None
        if 'model_currents_file' in command:
            from shared_io.parallel_inputs import prepare_parallel_stimulation
            ports = dict(zip(('retina_left', 'retina_right', 'sugar'),
                             ((action.astype(np.float64) + 1.0) / 2.0).tolist()))
            if record is not None:
                raw = self._attach_neural_state(record)['neural_state_file']
                parallel = prepare_parallel_stimulation(self.brain, raw, command['model_currents_file'], original, ports)
                parallel['budget_source'] = {key: raw[key] for key in
                    ('sha256', 'ids_sha256', 'simulation_ms', 'duration_ms', 'neurons')}
                parallel['budget_source']['temporary'] = False
            else:
                from connectome_adapter.hot_state import prepare_temporary_stimulation
                parallel = prepare_temporary_stimulation(self.brain, self.worker_directory,
                    command['model_currents_file'], original, ports)
        if 'general_stimulation' in command:
            self.env.set_general_stimulation(original)
        if 'neuron_currents_file' in command:
            self.env.append_neuron_currents(command['neuron_currents_file'])
        if command.get('aversive'):
            self.env.queue_aversive()
        if parallel:
            self.env.set_general_stimulation([*original, {'neuron_currents_file': parallel['applied_currents_file']}])
        started = time.perf_counter()
        try:
            # Exactly the ordinary Lab path: recorder, online capture, actual
            # PPL delivery and its 3D feedback, then the display frame.
            self.session._step(action)
        finally:
            self.input_record = None
            self.env.set_general_stimulation([item for item in original if 'neuron_currents_file' not in item])
        info = deepcopy(self.session.info)
        if record and info['neural']['input_rgb_sha256'] != record['rgb_file']['raw_rgb_sha256']:
            raise RuntimeError('Actual brain pixels differ from the dispatched Lab observation.')
        if parallel:
            actual = info['neural']['general_stimulation']['current_vector_sha256']
            if actual != parallel['expected_general_currents_sha256']:
                raise RuntimeError('Actual full-brain current sum differs from the preserved-input budget.')
            parallel.update(actual_general_currents_sha256=actual, actual_sum_verified=True)
        self._refresh_status()
        result = {'status': deepcopy(self._cached), 'observation': self.session.obs.tolist(),
            'reward': info['last_reward'], 'terminated': info['terminated'], 'truncated': info['truncated'],
            'info': info, 'input_id': record['input_id'] if record else None,
            'wall_seconds': time.perf_counter() - started, 'model_invocation': False,
            'scope': 'existing_lab_environment'}
        if parallel:
            result['parallel_inputs'] = parallel
        if command.get('capture_neural_state'):
            result['feedback'] = self._save_observation('feedback', True)
            result['feedback']['source_input_id'] = result['input_id']
        return result

    def reset(self, env_id, *, seed=None):
        self._ids([env_id])
        expected = 42 + self.session.episode + 1
        if seed is not None and (type(seed) is not int or seed != expected):
            raise ValueError('A Lab episode uses its existing deterministic episode seed progression.')
        self.session._reset(preserve_native_flow=True)
        self.input_record = None
        self._refresh_status()
        return {'status': deepcopy(self._cached), 'observation': self.session.obs.tolist(),
                'info': deepcopy(self.session.info), 'brain_weights_preserved': True}

    def close(self):
        if self.closed:
            return
        self._owner()
        # A pending render is just a pin, not a physics state. Do not discard
        # any user sensory configuration, apply a current, or reset the fly.
        if self.input_record is not None:
            packet = getattr(self.env, '_shared_body_packet', None)
            if packet and packet.get('packet_sha256') == self.input_record['body_packet_file']['packet_sha256']:
                self.env._shared_input_rgb = self.env._shared_body_packet = None
        self.input_record = None
        self.closed = True
        self._cached = {**self._cached, 'pending_input_id': None, 'alive': False}


class SessionNativeOmniFlow:
    """Small Lab lifecycle adapter, with cached state and actual audio receipts."""

    def __init__(self, session, *, text, speak=False, max_tokens=32,
                 model_interval_s=2, hidden_ttl_s=5, io_scope='sensory_motor', audio_path=None):
        self.pool = SessionEnvironmentPool(session)
        from .session_cache import SessionInferenceCache
        if getattr(session, '_omni_cache', None) is None:
            session._omni_cache = SessionInferenceCache()
        session._omni_cache.bind_environment(session.env, session.env.brain)
        from .native_flow import NativeOmniFlow
        self.flow = NativeOmniFlow(self.pool, text=text, speak=speak, max_tokens=max_tokens,
                                   model_interval_s=model_interval_s, hidden_ttl_s=hidden_ttl_s,
                                   io_scope=io_scope, audio_path=audio_path,
                                   inference_cache=session._omni_cache)
        self.last = None
        self.closed = False
        self._archived = set()
        self._cached = self._build_status()

    def _build_status(self):
        value = self.flow.status()
        return {**value, 'busy': bool(value['pending_model_environments']),
                'scope': 'existing_lab_environment', 'pool': self.pool.status()}

    def _publish(self):
        key = self.pool.env_id
        value = self.flow.status()
        response = value['responses'].get(key)
        turn = self.flow.turns.get(key)
        event_id = turn['event_id'] if turn else (response or {}).get('event_id')
        if not event_id:
            return
        complete = response is not None and response['event_id'] == event_id
        progress = self.flow.stream_progress(key)
        event = {'schema': 1, 'event_id': event_id, 'phase': 'complete' if complete else 'error' if value['errors'].get(key) else 'running',
            'model_backend': 'vllm_omni', 'transport': 'simplex', 'connection_mode': 'neuron_direct',
            'io_scope': self.flow.io_scope, 'scope': 'existing_lab_environment', 'fusion_mode': 'latent_only',
            'asynchronous': True, 'same_observation_and_application_frame_claimed': False,
            'original_inputs_preserved': True,
            'actual_model_applications': value['actual_model_applications'].get(key, 0),
            'body_steps_while_model_pending': value['body_steps_while_model_pending'].get(key, 0),
            'recent_applications': [row for row in value['recent_applications'] if row['event_id'] == event_id],
            'brain_body': self.pool.status()['workers'][key],
            'say': response['text'] if complete else (progress or {}).get('text', ''),
            'streaming': progress, 'sequence': (progress or {}).get('sequence', 0),
            'updated_at': time.time(), 'error': value['errors'].get(key)}
        if complete:
            event.update(application_receipts=response['application_receipts'],
                         native_latent_verified=response['native_latent_verified'],
                         request_id=response['request_id'], model_evidence=response)
            if event_id not in self._archived:
                self._archive_response(event, response)
                self._archived.add(event_id)
            else:
                saved = json.loads((ROOT / 'artifacts/shared_io' / event_id / 'event.json').read_text())
                for field in ('audio_output_path', 'audio_output_sha256', 'native_speech'):
                    if field in saved:
                        event[field] = saved[field]
        self.last = event

    def _archive_response(self, event, response):
        """Only a verified actual response gets the existing audio-route index."""
        if not response.get('native_latent_verified') or not response.get('application_receipts'):
            raise ValueError('Cannot publish an unverified model/brain response.')
        audio = response.get('audio')
        if audio:
            path = Path(audio['path']).resolve()
            if not path.is_relative_to(self.flow.directory) or not path.is_file() or path.stat().st_size > 32_000_000:
                raise ValueError('Native audio must belong to this exact flow.')
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != audio['sha256'] or not response['text'].strip():
                raise ValueError('Native speech identity or actual text is missing.')
            with wave.open(io.BytesIO(raw)) as wav:
                if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (24000, 1, 2) or wav.getnframes() != audio['samples']:
                    raise ValueError('Native speech format or actual sample count differs.')
            event.update(audio_output_path=str(path), audio_output_sha256=audio['sha256'],
                native_speech={**audio, 'event_id': event['event_id'], 'request_id': response['request_id'],
                    'source': 'Actual vLLM-Omni native speech from this numerically coupled Lab turn'})
        folder = ROOT / 'artifacts/shared_io' / event['event_id']
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / 'event.json'
        if target.exists():
            previous = json.loads(target.read_text())
            if previous.get('model_backend') != 'vllm_omni' or previous.get('request_id') != event['request_id']:
                raise ValueError('Shared event ID is already owned by another response.')
        temporary = folder / ('event-' + uuid.uuid4().hex + '.tmp')
        temporary.write_text(_json(event))
        temporary.replace(target)

    def tick(self):
        self.pool._owner()
        results = self.flow.tick()
        self._publish()
        self._cached = self._build_status()
        return results

    def status(self):
        return deepcopy(self._cached)

    def stream_progress(self):
        return self.flow.stream_progress(self.pool.env_id)

    def close(self):
        if self.closed:
            return
        self.pool._owner()
        self.flow.close()
        self.pool.close()
        self.closed = True
        self._cached = self._build_status()
