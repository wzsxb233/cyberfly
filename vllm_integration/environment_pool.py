"""Independent CPU bodies/brains for an external inference dispatcher.

Nothing starts on import or construction. ``start()`` owns two isolated process
trees at most; closing the pool never touches another service. Off-screen flies
still render their real sensory camera at every neural step. Only the selected
fly produces a display JPEG. This module does not call or impersonate an LLM.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = Path('/root/.cache/cyberfly/rl-env/bin/python')
CPU_ENV = {
    'CUDA_VISIBLE_DEVICES': '', 'MUJOCO_GL': 'egl',
    'LIBGL_ALWAYS_SOFTWARE': '1', 'GALLIUM_DRIVER': 'llvmpipe',
    '__EGL_VENDOR_LIBRARY_FILENAMES': '/usr/share/glvnd/egl_vendor.d/50_mesa.json',
    'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
}


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _artifact_path(value):
    path = Path(value).resolve()
    if not path.is_relative_to((ROOT / 'artifacts').resolve()):
        raise ValueError('Pool output must be inside project artifacts.')
    return path


def _configurations(configs):
    if configs is None:
        configs = [{'env_id': f'fly-{i}', 'seed': 20260913 + i,
                    'scenario': {'name': 'fly_neural_link', 'env_kwargs': {
                        'learning': False, 'visual_input_enabled': True, 'max_episode_steps': 150}}}
                   for i in range(2)]
    if not isinstance(configs, list) or not 1 <= len(configs) <= 2:
        raise ValueError('The initial CPU pool supports one or two environments.')
    result = deepcopy(configs)
    seen = set()
    for config in result:
        if not isinstance(config, dict) or set(config) != {'env_id', 'seed', 'scenario'}:
            raise ValueError('Each environment requires exactly env_id, seed and scenario.')
        name = config['env_id']
        if not isinstance(name, str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,48}', name) or name in seen:
            raise ValueError('Environment IDs must be distinct safe names.')
        seen.add(name)
        if type(config['seed']) is not int or not 0 <= config['seed'] < 2**32:
            raise ValueError('Each seed must be an unsigned 32-bit integer.')
        scenario = config['scenario']
        if not isinstance(scenario, dict) or not isinstance(scenario.get('env_kwargs', {}), dict):
            raise ValueError('A standard scenario configuration is required.')
        if scenario.get('plugins'):
            raise ValueError('The isolated pool currently supports built-in scenarios only.')
        kwargs = scenario.setdefault('env_kwargs', {})
        if kwargs.get('visual_input_enabled', True) is not True:
            raise ValueError('Pool environments preserve the existing visual input.')
        kwargs['visual_input_enabled'] = True
    return result


class _OwnedWorker:
    def __init__(self, config, directory, python, timeout):
        self.config, self.directory, self.timeout = config, directory, timeout
        self.lock = threading.Lock()
        self.responses = queue.Queue()
        self.sequence = 0
        self.cached = {'env_id': config['env_id'], 'state': 'starting'}
        self.closed = False
        self.unusable = False
        self.stderr = (directory / 'worker.log').open('w')
        self.process = subprocess.Popen(
            [str(python), '-u', str(Path(__file__).resolve()), '--worker'],
            cwd=ROOT, env={**os.environ, **CPU_ENV}, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.stderr, text=True,
            encoding='utf-8', bufsize=1, start_new_session=True)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        try:
            for line in self.process.stdout:
                try:
                    self.responses.put(json.loads(line))
                except ValueError:
                    self.responses.put(RuntimeError('Malformed environment worker response.'))
        finally:
            self.responses.put(RuntimeError(f'Environment worker exited; inspect {self.directory / "worker.log"}'))

    def request(self, operation, **data):
        with self.lock:
            if self.closed or self.unusable or self.process.poll() is not None:
                raise RuntimeError('Environment worker is closed.')
            self.sequence += 1
            identity = self.sequence
            self.process.stdin.write(_json({'id': identity, 'op': operation, **data}) + '\n')
            self.process.stdin.flush()
            try:
                response = self.responses.get(timeout=self.timeout)
            except queue.Empty as exc:
                self.unusable = True
                raise TimeoutError(f'{self.config["env_id"]}: {operation} exceeded {self.timeout} seconds; outcome unknown, do not retry a step.') from exc
            if isinstance(response, BaseException):
                self.unusable = True
                raise response
            if response.get('id') != identity:
                self.unusable = True
                raise RuntimeError('Environment response identity mismatch; outcome unknown.')
            if response.get('ok') is not True:
                raise RuntimeError(response.get('error', 'Environment worker failed.'))
            result = response['result']
            if 'status' in result:
                self.cached = result['status']
            return result

    def close(self):
        if self.closed:
            return
        try:
            if self.process.poll() is None:
                self.request('close')
                self.process.wait(timeout=10)
        except BaseException:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            # A brain subprocess has its own session. Only this worker's
            # recorded direct child is eligible for emergency cleanup.
            child = self.cached.get('brain_pid')
            if child:
                try:
                    os.kill(child, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        finally:
            self.closed = True
            self.stderr.close()
            for stream in (self.process.stdin, self.process.stdout):
                if stream:
                    stream.close()


class EnvironmentPool:
    """Explicitly started, selected-view CPU environment collection.

    ``capture_inputs`` pins actual RGB/body data for a batch dispatcher. Submit
    the returned ``input_id`` with ``step_many`` to reject stale model outputs.
    Actions enter neural input ports, never body actuators. A missing action
    preserves the scenario's configured semantic/port input. No automatic reset,
    training, model invocation, checkpoint deployment, or playback is performed.
    """
    def __init__(self, configs=None, *, directory=None, python=DEFAULT_PYTHON, timeout=240):
        self.configs = _configurations(configs)
        self.directory = _artifact_path(directory or ROOT / 'artifacts/environment_pool' / uuid.uuid4().hex[:12])
        # Resolving a venv's executable symlink would discard its packages.
        self.python = Path(python).absolute()
        if not self.python.is_file() or type(timeout) not in (int, float) or not 0 < timeout <= 600:
            raise ValueError('Existing worker interpreter and timeout in (0,600] required.')
        self.timeout = float(timeout)
        self.workers = {}
        self.selected = None
        self.started = False
        self.closed = False

    def start(self, configs=None):
        if self.started or self.closed:
            raise RuntimeError('Create a new pool instead of starting it twice.')
        if configs is not None:
            self.configs = _configurations(configs)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.started = True
        try:
            for config in self.configs:
                folder = self.directory / config['env_id']
                folder.mkdir()
                self.workers[config['env_id']] = _OwnedWorker(config, folder, self.python, self.timeout)
            self._parallel({name: ('init', {'config': worker.config, 'directory': str(worker.directory)})
                            for name, worker in self.workers.items()})
            self.selected = self.configs[0]['env_id']
            (self.directory / 'configuration.json').write_text(_json({'schema': 1, 'configs': self.configs,
                'cpu_environment': CPU_ENV, 'model_invocation': False, 'display_policy': 'selected JPEG only; all sensory cameras remain active'}))
            return self.status()
        except BaseException:
            self.close()
            raise

    def _worker(self, env_id):
        if not self.started or self.closed:
            raise RuntimeError('Call start() on an open pool first.')
        if env_id not in self.workers:
            raise ValueError(f'Unknown environment: {env_id}')
        return self.workers[env_id]

    def _parallel(self, commands):
        for name in commands:
            self._worker(name)
        with ThreadPoolExecutor(max_workers=max(1, len(commands))) as executor:
            futures = {name: executor.submit(self.workers[name].request, operation, **payload)
                       for name, (operation, payload) in commands.items()}
            results, failures = {}, {}
            for name, future in futures.items():
                try:
                    results[name] = future.result()
                except Exception as exc:
                    failures[name] = str(exc)
            if failures:
                # Partial success is not rolled back or silently replayed.
                error = RuntimeError(f'Environment batch failed: {failures}; completed={list(results)}')
                error.completed_results, error.failures = results, failures
                raise error
            return results

    def status(self):
        return {'schema': 1, 'started': self.started, 'closed': self.closed,
                'selected_env_id': self.selected, 'directory': str(self.directory),
                'model_invocation': False, 'workers': {name: {**deepcopy(worker.cached),
                    'alive': worker.process.poll() is None} for name, worker in self.workers.items()}}

    def select(self, env_id):
        self._worker(env_id)
        self.selected = env_id
        return self.status()

    def capture_inputs(self, env_ids=None, *, include_neural_state=False):
        if type(include_neural_state) is not bool:
            raise ValueError('include_neural_state must be bool.')
        ids = list(self.workers) if env_ids is None else list(env_ids)
        return self._parallel({name: ('capture', {'include_neural_state': include_neural_state}) for name in ids})

    def capture_feedback(self, env_ids=None, *, include_neural_state=False):
        if type(include_neural_state) is not bool:
            raise ValueError('include_neural_state must be bool.')
        ids = list(self.workers) if env_ids is None else list(env_ids)
        return self._parallel({name: ('feedback', {'include_neural_state': include_neural_state}) for name in ids})

    def step_many(self, commands):
        if not isinstance(commands, dict) or not commands:
            raise ValueError('Provide env_id -> step command dictionaries.')
        return self._parallel({name: ('step', {'command': command}) for name, command in commands.items()})

    def step(self, env_id, **command):
        return self.step_many({env_id: command})[env_id]

    def reset(self, env_id, *, seed=None):
        return self._worker(env_id).request('reset', seed=seed)

    def set_task(self, env_id, task):
        return self._worker(env_id).request('task', task=task)

    def render_selected(self, *, quality=85):
        if type(quality) is not int or not 30 <= quality <= 95:
            raise ValueError('JPEG quality must be an integer in [30,95].')
        return self._worker(self.selected).request('render', quality=quality)

    def save(self, env_id):
        return self._worker(env_id).request('save')

    def close(self):
        for worker in self.workers.values():
            worker.close()
        self.closed = True

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.close()


def _worker_main():
    """Protocol worker; imported heavy dependencies only in the CPU process."""
    from contextlib import redirect_stdout
    def finish_on_termination(*_):
        raise SystemExit('Owned environment worker asked to stop.')
    signal.signal(signal.SIGTERM, finish_on_termination)
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from PIL import Image
    env = None
    config = None
    directory = None
    episode = 0
    input_record = None
    frame_seq = 0
    jpeg_count = 0
    gl_renderer = None

    def status():
        info = env.task_state()
        neural = info.get('neural') or {}
        telemetry = neural.get('neural', {})
        return {'env_id': config['env_id'], 'pid': os.getpid(), 'brain_pid': env.brain.process.pid,
                'seed': config['seed'], 'episode': episode, 'phase': info.get('phase'),
                'scenario': info.get('scenario'), 'step': info.get('step'), 'frame_seq': frame_seq,
                'brain_neurons': info.get('brain_neurons'), 'brain_edges': info.get('brain_edges'),
                'simulation_ms': telemetry.get('simulation_ms', 0),
                'spikes_this_step': telemetry.get('spikes_this_step', 0),
                'total_spikes': telemetry.get('total_spikes', 0),
                'body_interval_ms': info.get('body_interval_ms'), 'neural_interval_ms': info.get('neural_interval_ms'),
                'body_action': info.get('body_action'), 'brain_learning': info.get('brain_learning'),
                'visual_input_enabled': info.get('visual_input_enabled'),
                'input_rgb_sha256': neural.get('input_rgb_sha256'),
                'jpeg_encodes': jpeg_count, 'renderer': gl_renderer,
                'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'pending_input_id': input_record.get('input_id') if input_record else None}

    def attach_neural_state(record):
        if 'neural_state_file' not in record:
            from connectome_adapter.hot_state import capture_observation_snapshot
            record['neural_state_file'] = capture_observation_snapshot(env.brain, directory,
                kind=record['kind'], record_id=record['record_id'], env_id=record['env_id'])
        return record

    def save_observation(kind, include_neural_state):
        if kind == 'input':
            rgb = env.shared_input_rgb()
            packet = env.shared_body_senses()
        else:
            # After-step sampling does not create a pending next-step input.
            rgb = env._frame.copy()
            packet = env.body.unwrapped.body_sensory_packet(rgb=rgb)
        tag = uuid.uuid4().hex
        folder = directory / ('inputs' if kind == 'input' else 'feedback') / tag
        folder.mkdir(parents=True)
        np.save(folder / 'rgb.npy', rgb, allow_pickle=False)
        (folder / 'body_packet.json').write_text(_json(packet))
        raw_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
        if packet['vision']['rgb_sha256'] != raw_sha:
            raise RuntimeError('Body and camera identity mismatch.')
        record = {'schema': 1, 'kind': kind, 'record_id': tag, 'env_id': config['env_id'], 'episode': episode,
            'frame_seq': frame_seq, 'body_simulation_time_s': packet['simulation_time_s'],
            'rgb_file': {'path': str(folder / 'rgb.npy'), 'sha256': _digest(folder / 'rgb.npy'),
                         'raw_rgb_sha256': raw_sha, 'shape': list(rgb.shape), 'dtype': str(rgb.dtype)},
            'body_packet_file': {'path': str(folder / 'body_packet.json'),
                                 'sha256': _digest(folder / 'body_packet.json'), 'packet_sha256': packet['packet_sha256']},
            'model_invocation': False}
        if include_neural_state:
            attach_neural_state(record)
        return record

    def capture(include_neural_state=False):
        nonlocal input_record
        if input_record is not None:
            if include_neural_state:
                attach_neural_state(input_record)
            return deepcopy(input_record)
        input_record = save_observation('input', include_neural_state)
        input_record['input_id'] = input_record['record_id']
        return deepcopy(input_record)

    def operate(request):
        nonlocal env, config, directory, episode, input_record, frame_seq, jpeg_count, gl_renderer
        op = request['op']
        if op == 'init':
            if env is not None:
                raise RuntimeError('Worker already initialized.')
            from scenarios import create_scenario
            config = request['config']
            directory = _artifact_path(request['directory'])
            env = create_scenario(config['scenario'])
            if not env.scenario_spec.get('neural_input') or env.scenario_spec.get('embodiment') != 'physical_3d':
                raise ValueError('This pool currently requires an embodied physical neural scenario.')
            env.reset(seed=config['seed'])
            episode = 1
            frame_seq = 1
            from OpenGL import GL
            gl_renderer = GL.glGetString(GL.GL_RENDERER).decode()
            if 'llvmpipe' not in gl_renderer.lower() and 'softpipe' not in gl_renderer.lower():
                raise RuntimeError(f'CPU software renderer required; received {gl_renderer}')
            return {'status': status(), 'spec': env.scenario_spec}
        if env is None:
            raise RuntimeError('Initialize the worker first.')
        if op == 'capture':
            return {**capture(request.get('include_neural_state', False)), 'status': status()}
        if op == 'feedback':
            return {**save_observation('feedback', request.get('include_neural_state', False)), 'status': status()}
        if op == 'step':
            command = request['command']
            allowed = {'action', 'input_id', 'general_stimulation', 'neuron_currents_file',
                       'model_currents_file', 'aversive', 'capture_neural_state'}
            if not isinstance(command, dict) or set(command) - allowed:
                raise ValueError(f'Step supports only {sorted(allowed)}.')
            if 'input_id' in command and (input_record is None or command['input_id'] != input_record['input_id']):
                raise ValueError('Stale or foreign input_id; no neural/body step executed.')
            if input_record is not None and 'input_id' not in command:
                raise ValueError('A captured input is pinned: submit its input_id before stepping.')
            action = np.asarray(command.get('action', env.shared_passthrough_action()), dtype=np.float32)
            if not env.action_space.contains(action):
                raise ValueError('Action is outside this neural input space.')
            if 'aversive' in command and type(command['aversive']) is not bool:
                raise ValueError('aversive must be bool.')
            if type(command.get('capture_neural_state', False)) is not bool:
                raise ValueError('capture_neural_state must be bool.')
            if 'model_currents_file' in command and 'neuron_currents_file' in command:
                raise ValueError('Use model_currents_file for budgeted model input or neuron_currents_file for direct experiments, not both.')
            from connectome_adapter.ports import validate_general_stimulation
            legacy_general = validate_general_stimulation(command.get('general_stimulation', env.general_stimulation))
            parallel_inputs = None
            if 'model_currents_file' in command:
                from shared_io.parallel_inputs import prepare_parallel_stimulation
                normalized = (action.astype(np.float64) + 1.0) / 2.0
                actual_ports = dict(zip(('retina_left', 'retina_right', 'sugar'), normalized.tolist()))
                if input_record is not None:
                    raw_state = attach_neural_state(input_record)['neural_state_file']
                    parallel_inputs = prepare_parallel_stimulation(env.brain, raw_state, command['model_currents_file'],
                                                                   legacy_general, actual_ports)
                    parallel_inputs['budget_source'] = {key: raw_state[key] for key in
                        ('sha256', 'ids_sha256', 'simulation_ms', 'duration_ms', 'neurons')}
                    parallel_inputs['budget_source']['temporary'] = False
                else:
                    from connectome_adapter.hot_state import prepare_temporary_stimulation
                    parallel_inputs = prepare_temporary_stimulation(env.brain, directory,
                        command['model_currents_file'], legacy_general, actual_ports)
            if 'general_stimulation' in command:
                env.set_general_stimulation(legacy_general)
            if 'neuron_currents_file' in command:
                env.append_neuron_currents(command['neuron_currents_file'])
            if command.get('aversive'):
                env.queue_aversive()
            if parallel_inputs:
                env.set_general_stimulation([*legacy_general, {'neuron_currents_file': parallel_inputs['applied_currents_file']}])
            record = input_record
            started = time.perf_counter()
            try:
                obs, reward, terminated, truncated, info = env.step(action)
            finally:
                input_record = None
                if parallel_inputs:
                    # Retain all persistent old inputs; every file is one
                    # interval and is never automatically replayed on retry.
                    env.set_general_stimulation([item for item in legacy_general if 'neuron_currents_file' not in item])
            elapsed = time.perf_counter() - started
            if record and info['neural']['input_rgb_sha256'] != record['rgb_file']['raw_rgb_sha256']:
                raise RuntimeError('Actual neural image differs from the dispatched input.')
            if parallel_inputs:
                actual = info['neural']['general_stimulation']['current_vector_sha256']
                if actual != parallel_inputs['expected_general_currents_sha256']:
                    raise RuntimeError('Actual full-brain input sum differs from the preserved-input budget.')
                parallel_inputs['actual_general_currents_sha256'] = actual
                parallel_inputs['actual_sum_verified'] = True
            frame_seq += 1
            result = {'status': status(), 'observation': obs.tolist(), 'reward': reward,
                    'terminated': terminated, 'truncated': truncated, 'info': info,
                    'input_id': record['input_id'] if record else None, 'wall_seconds': elapsed,
                    'model_invocation': False}
            if parallel_inputs:
                result['parallel_inputs'] = parallel_inputs
            if command.get('capture_neural_state'):
                result['feedback'] = save_observation('feedback', True)
                result['feedback']['source_input_id'] = result['input_id']
            return result
        if op == 'reset':
            seed = request.get('seed')
            if seed is not None:
                if type(seed) is not int or not 0 <= seed < 2**32:
                    raise ValueError('seed must be an unsigned 32-bit integer.')
                config['seed'] = seed
            obs, info = env.reset(seed=config['seed'])
            episode += 1
            frame_seq += 1
            input_record = None
            return {'status': status(), 'observation': obs.tolist(), 'info': info,
                    'brain_weights_preserved': True}
        if op == 'task':
            if input_record is not None:
                raise ValueError('Finish the pinned input before changing the task.')
            return {'applied': env.apply_task(request['task']), 'status': status()}
        if op == 'render':
            # Use the last actual frame without creating a new sensory capture.
            rgb = env._frame.copy()
            stream = io.BytesIO()
            Image.fromarray(rgb).save(stream, format='JPEG', quality=request['quality'])
            encoded = stream.getvalue()
            jpeg_count += 1
            return {'env_id': config['env_id'], 'episode': episode, 'frame_seq': frame_seq,
                    'jpeg_base64': base64.b64encode(encoded).decode(), 'mime_type': 'image/jpeg',
                    'sha256': hashlib.sha256(encoded).hexdigest(), 'raw_rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
                    'status': status()}
        if op == 'save':
            result = env.brain.save(directory / 'checkpoints')
            return {'checkpoint': result, 'status': status()}
        if op == 'close':
            env.close()
            return {'closed': True}
        raise ValueError(f'Unsupported operation: {op}')

    try:
        for line in sys.stdin:
            request = None
            try:
                request = json.loads(line)
                with redirect_stdout(sys.stderr):
                    result = operate(request)
                response = {'id': request['id'], 'ok': True, 'result': result}
            except Exception as exc:
                response = {'id': request.get('id') if isinstance(request, dict) else None,
                            'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
            sys.stdout.write(_json(response) + '\n')
            sys.stdout.flush()
            if isinstance(request, dict) and request.get('op') == 'close':
                break
    finally:
        if env is not None:
            with redirect_stdout(sys.stderr):
                env.close()


if __name__ == '__main__':
    if sys.argv[1:] != ['--worker']:
        raise SystemExit('Use EnvironmentPool from Python; this CLI only serves owned workers.')
    _worker_main()
