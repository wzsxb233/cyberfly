"""One owner for an opt-in, independently checkpointed native Omni fleet.

Public status/frame reads only copy cached data. No handler calls model
inference, physics, live Lab saving, or a GPU-service start/stop operation.
"""
from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import queue
import re
import shutil
import threading
import time
import uuid
import wave

from .environment_pool import EnvironmentPool

ROOT = Path(__file__).resolve().parents[1]
ACCEPTANCE_PATH = ROOT / 'artifacts/vllm_omni_migration/acceptance.json'
DEFAULT_INSTRUCTION = '观察眼前环境，自由探索，用一句简短中文描述变化。'


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def acceptance_status(path=ACCEPTANCE_PATH):
    """Fail closed; this reads the explicit acceptance marker, never a GPU."""
    path = Path(path)
    result = {'ready': False, 'readiness_error': '等待模型双向连接与原生语音的实际验收。',
              'acceptance_path': str(path), 'acceptance_sha256': None}
    try:
        if not path.is_file():
            return result
        if path.stat().st_size > 2_000_000:
            raise ValueError('迁移验收记录过大')
        raw = path.read_bytes()
        value = json.loads(raw)
        result['acceptance_sha256'] = hashlib.sha256(raw).hexdigest()
        if not isinstance(value, dict) or value.get('status') != 'passed' or value.get('model_bidirectional_verified') is not True or value.get('native_speech_verified') is not True:
            raise ValueError('迁移验收尚未同时确认真实模型双向连接和原生语音')
        result.update(ready=True, readiness_error=None)
    except (OSError, ValueError) as exc:
        result['readiness_error'] = str(exc)
    return result


def _local_directory(value):
    if not isinstance(value, str) or not value:
        raise ValueError('请选择已存在的本地存档路径')
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT) or not path.is_dir():
        raise ValueError('存档必须是当前项目内已有的目录')
    return path


def _brain_generation(value):
    path = _local_directory(value)
    if (path / 'latest.json').is_file():
        generation = json.loads((path / 'latest.json').read_text()).get('generation')
        if not isinstance(generation, str) or not re.fullmatch(r'[a-f0-9]{32}', generation):
            raise ValueError('大脑存档版本指针无效')
        path = path / generation
    if not (path / 'adapter.json').is_file() or not (path / 'brain.npz').is_file():
        raise ValueError('请选择包含完整大脑状态的既有存档')
    metadata = json.loads((path / 'adapter.json').read_text())
    return str(path), {'requested_path': value, 'immutable_path': str(path),
                       'memory_sha256': metadata.get('memory_sha256'),
                       'generation': metadata.get('generation')}


class FleetController:
    """Asynchronous start/stop/select commands; simulation has one owner.

    Factories and acceptance_path can be injected for CPU protocol tests. The
    production defaults use actual EnvironmentPool and NativeOmniFlow.
    """
    def __init__(self, state_provider, *, migration_seed=None, acceptance_path=ACCEPTANCE_PATH,
                 output_root=ROOT / 'artifacts/vllm_omni_migration/fleets',
                 pool_factory=EnvironmentPool, flow_factory=None, min_free_bytes=2 * 1024**3,
                 training_jobs=None):
        if not callable(state_provider):
            raise TypeError('A read-only cached Lab state provider is required.')
        if type(min_free_bytes) is not int or min_free_bytes < 0:
            raise ValueError('min_free_bytes must be a nonnegative integer.')
        self.state_provider = state_provider
        self.migration_seed = deepcopy(migration_seed or {})
        self.acceptance_path = Path(acceptance_path)
        self.output_root = Path(output_root).resolve()
        if not self.output_root.is_relative_to(ROOT / 'artifacts'):
            raise ValueError('Fleet artifacts must remain in this project.')
        self.pool_factory, self.flow_factory = pool_factory, flow_factory
        self.min_free_bytes = min_free_bytes
        self.training_jobs = training_jobs
        self.online_loop = self.online_observer = self.online_dispatcher = None
        self.audio_descriptors = {}
        self.audio_root = None
        self.lock = threading.Lock()
        self.commands = queue.Queue()
        self.stop_requested = threading.Event()
        self.shutdown_requested = threading.Event()
        self.stopped = threading.Event()
        self.stopped.set()
        self.pool = self.flow = self.directory = None
        self.jpeg = None
        self.jpeg_identity = None
        self.saved = {}
        self._closed = False
        self.cached = {'schema': 1, 'enabled': False, 'state': 'disabled', 'selected_id': None,
                       'environments': [], 'loop_paused': True, 'model_backend': 'vllm_omni',
                       'stop_requested': False, 'error': None, 'last_checkpoints': {},
                       'online_learning': {'enabled': False, 'data_source': 'independent_fleet_actual_transitions'},
                       'online_error': None, 'online_requested': False,
                       'migration_seed': deepcopy(self.migration_seed)}
        self.thread = threading.Thread(target=self._owner, name='cyberfly-fleet-owner', daemon=True)
        self.thread.start()

    def status(self):
        with self.lock:
            result = deepcopy(self.cached)
        return {**result, **acceptance_status(self.acceptance_path)}

    def frame(self, env_id):
        with self.lock:
            if env_id != self.cached['selected_id']:
                raise ValueError('仅可读取当前选中的果蝇画面')
            if not self.jpeg:
                raise ValueError('选中果蝇的真实画面尚未准备完成')
            return self.jpeg

    def audio(self, env_id, sha256):
        with self.lock:
            if env_id != self.cached['selected_id']:
                raise ValueError('仅可读取当前选中果蝇的原生语音')
            descriptor = deepcopy(self.audio_descriptors.get(env_id))
            audio_root = self.audio_root
        if not descriptor or descriptor.get('sha256') != sha256 or not re.fullmatch(r'[a-f0-9]{64}', sha256 or ''):
            raise ValueError('这句话没有已验证的原生语音，或所选响应已变化')
        path = Path(descriptor.get('path', '')).resolve()
        allowed = ROOT / 'artifacts/vllm_omni_migration/flows'
        if not audio_root or not path.is_relative_to(audio_root) or not path.is_relative_to(allowed) or path.suffix != '.wav' or not path.is_file() or not 0 < path.stat().st_size <= 32_000_000:
            raise ValueError('语音必须来自本环境池已完成的模型响应目录')
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != sha256:
            raise ValueError('原生语音文件校验失败')
        with wave.open(io.BytesIO(data)) as source:
            if source.getframerate() != 24000 or source.getnchannels() != 1 or source.getsampwidth() != 2 or not source.getnframes():
                raise ValueError('原生语音格式无效')
        return data

    def _configuration(self, payload):
        allowed = {'count', 'scenario', 'brain_checkpoint', 'motor_checkpoint', 'seed', 'text', 'speak'}
        if not isinstance(payload, dict) or set(payload) - allowed:
            raise ValueError('不支持的环境池启动参数')
        count, seed = payload.get('count', 2), payload.get('seed', 20260913)
        if type(count) is not int or not 1 <= count <= 2:
            raise ValueError('目前可启动一只或两只独立果蝇')
        if type(seed) is not int or not 0 <= seed <= 2**32 - count:
            raise ValueError('随机种子超出范围')
        if type(payload.get('speak', False)) is not bool:
            raise ValueError('speak 必须是布尔值')
        text = payload.get('text', DEFAULT_INSTRUCTION)
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 1000:
            raise ValueError('请输入1–1000字的实际指令')
        lab = deepcopy(self.state_provider())
        scenario = deepcopy(payload.get('scenario') or lab.get('scenario') or {'name': 'flybody_neural_link', 'env_kwargs': {}})
        if not isinstance(scenario, dict):
            raise ValueError('当前场景配置无效')
        # The active Lab already published an authoritative scenario spec.
        # Reusing it avoids cold registry imports in the HTTP start handler.
        spec = lab.get('spec')
        if not isinstance(spec, dict) or spec.get('name') != scenario.get('name'):
            from scenarios import list_scenarios
            spec = next((item for item in list_scenarios() if item['name'] == scenario.get('name')), None)
        if not spec or not spec.get('neural_input') or spec.get('embodiment') != 'physical_3d':
            raise ValueError('环境池需要带真实大脑的三维果蝇场景')
        kwargs = scenario.setdefault('env_kwargs', {})
        if not isinstance(kwargs, dict):
            raise ValueError('场景参数必须是对象')
        info, shared = lab.get('info') or {}, lab.get('shared_io') or {}
        if 'scenario' not in payload:
            if type(lab.get('brain_learning')) is bool:
                kwargs['learning'] = lab['brain_learning']
            if isinstance(info.get('neural_drive'), dict):
                kwargs['semantic_drive'] = deepcopy(info['neural_drive'])
            if isinstance(info.get('general_stimulation'), list):
                kwargs['general_stimulation'] = [deepcopy(item) for item in info['general_stimulation']
                                                 if 'neuron_currents_file' not in item]
        if kwargs.get('visual_input_enabled', True) is not True:
            raise ValueError('此运行器保留原始视觉，请选择开启视觉的场景')
        kwargs['visual_input_enabled'] = True
        source = payload.get('brain_checkpoint') or lab.get('loaded_brain_checkpoint') or kwargs.get('brain_checkpoint') or self.migration_seed.get('fleet_seed_brain_checkpoint')
        seed_source = {'kind': 'new_full_graph_initial_state', 'current_live_brain_copied': False}
        if source:
            kwargs['brain_checkpoint'], checkpoint = _brain_generation(source)
            seed_source = {'kind': 'explicit_existing_brain_checkpoint', 'current_live_brain_copied': False, **checkpoint}
        active_scenario = lab.get('scenario')
        active_name = active_scenario.get('name') if isinstance(active_scenario, dict) else active_scenario
        same_body_scenario = active_name == scenario.get('name')
        motor = (payload.get('motor_checkpoint') if 'motor_checkpoint' in payload else
                 kwargs.get('motor_readout_checkpoint') or
                 (((info.get('motor_readout') or {}).get('checkpoint') or
                   (shared.get('motor_readout') or {}).get('checkpoint')) if same_body_scenario else None))
        if motor and spec.get('body_action_shape') == [9]:
            raise ValueError('统一腿翼身体不能套用旧左右两维运动存档；请使用独立身体策略或当前明确的工程脑读出')
        if motor:
            motor_path = _local_directory(motor)
            if not (motor_path / 'metadata.json').is_file() or not (motor_path / 'weights.npz').is_file():
                raise ValueError('请选择已训练的运动读出存档')
            kwargs['motor_readout_checkpoint'] = str(motor_path)
        elif 'motor_checkpoint' in payload:
            kwargs.pop('motor_readout_checkpoint', None)
        configs = [{'env_id': f'fly{i}', 'seed': seed + i, 'scenario': deepcopy(scenario)} for i in range(count)]
        return {'configs': configs, 'text': text.strip(), 'speak': payload.get('speak', False),
                'source_lab_session_id': lab.get('session_id'), 'brain_seed_source': seed_source,
                'motor_checkpoint': kwargs.get('motor_readout_checkpoint')}

    def start(self, payload=None):
        readiness = acceptance_status(self.acceptance_path)
        if not readiness['ready']:
            raise ValueError(readiness['readiness_error'])
        config = self._configuration(payload or {})
        with self.lock:
            if self._closed or self.shutdown_requested.is_set() or self.cached['enabled']:
                raise ValueError('环境池已经启动或正在停止')
            self.stop_requested.clear()
            self.stopped.clear()
            pool_id = uuid.uuid4().hex[:12]
            self.cached.update(enabled=True, state='initializing', pool_id=pool_id, error=None,
                               loop_paused=True, stop_requested=False, selected_id=None, environments=[],
                               brain_seed_source=config['brain_seed_source'], motor_checkpoint=config['motor_checkpoint'],
                               instruction=config['text'], speak=config['speak'], pending_selected_id=None)
        self.commands.put(('start', {**config, 'pool_id': pool_id, 'acceptance': readiness}))
        return {'accepted': True, **self.status()}

    def select(self, env_id):
        with self.lock:
            if not self.cached['enabled'] or env_id not in {item['id'] for item in self.cached['environments']}:
                raise ValueError('请选择已经启动的果蝇')
            self.cached['pending_selected_id'] = env_id
        self.commands.put(('select', env_id))
        return {'accepted': True, **self.status()}

    def stop(self):
        self.stop_requested.set()
        with self.lock:
            if self.cached['enabled']:
                self.cached.update(state='stopping', loop_paused=True, stop_requested=True)
        self.commands.put(('stop', None))
        return {'accepted': True, **self.status()}

    def online_start(self):
        from model_training.policy import require_llm_training_allowed
        require_llm_training_allowed()
        if self.training_jobs is None:
            raise ValueError('当前服务没有配置显卡1训练调度')
        with self.lock:
            if self.cached['state'] != 'running':
                raise ValueError('先启动并运行真实环境池，再开启此独立训练数据源')
            if self.cached['online_requested'] or self.cached['online_learning'].get('active'):
                return {'accepted': True, 'online_learning': deepcopy(self.cached['online_learning'])}
            self.cached.update(online_requested=True, online_error=None)
        self.commands.put(('online_start', None))
        return {'accepted': True, **self.status()}

    def online_stop(self):
        self.commands.put(('online_stop', None))
        return {'accepted': True, **self.status()}

    def _online_stop_owned(self):
        if self.flow:
            self.flow.transition_observer = None
        if self.online_observer:
            self.online_observer.close()
        if self.online_loop:
            self.online_loop.stop(wait_timeout=0)
        with self.lock:
            self.cached['online_requested'] = False

    def _online_start_owned(self):
        if not self.flow or not self.pool or self.stop_requested.is_set():
            raise ValueError('环境池不在运行，未开启在线采集')
        if self.online_loop and self.online_loop.status().get('active'):
            return
        from .fleet_online import create_fleet_online
        self.online_loop, self.online_observer, self.online_dispatcher = create_fleet_online(
            self.pool, self.training_jobs, pool_id=self.cached['pool_id'], flow=self.flow)
        self.flow.transition_observer = self.online_observer
        self._publish(online_requested=False, online_error=None)

    def _publish(self, **updates):
        pool_state = self.pool.status() if self.pool else {'workers': {}, 'selected_env_id': None}
        flow_state = self.flow.status() if self.flow else {}
        pending = set(flow_state.get('pending_model_environments', []))
        errors = flow_state.get('errors', {})
        responses = flow_state.get('responses', {})
        online = ({**self.online_loop.status(), 'enabled': True, 'data_source': 'independent_fleet_actual_transitions',
                   'capture': self.online_observer.status() if self.online_observer else None}
                  if self.online_loop else {'enabled': False, 'data_source': 'independent_fleet_actual_transitions'})
        with self.lock:
            paused = updates.get('loop_paused', self.cached['loop_paused'])
            environments = []
            self.audio_descriptors = {}
            for name, worker in pool_state['workers'].items():
                response = responses.get(name) or {}
                say = response.get('text', '')
                audio = response.get('audio') if isinstance(say, str) and say.strip() and response.get('requested_audio') is True else None
                if isinstance(audio, dict) and audio.get('path') and audio.get('sha256'):
                    self.audio_descriptors[name] = deepcopy(audio)
                else:
                    audio = None
                environments.append({'id': name, 'instance_id': worker.get('pid'), 'status': 'paused' if paused else worker.get('phase', 'running'),
                    'playing': not paused, 'step': worker.get('step'), 'episode': worker.get('episode'),
                    'frame_seq': worker.get('frame_seq'), 'brain_simulation_ms': worker.get('simulation_ms'),
                    'brain_neurons': worker.get('brain_neurons'), 'brain_edges': worker.get('brain_edges'),
                    'brain_learning': worker.get('brain_learning'), 'seed': worker.get('seed'),
                    'model_pending': name in pending, 'error': errors.get(name),
                    'generation_phase': 'generating' if name in pending else 'response_ready' if say else 'waiting',
                    'last_say': say if isinstance(say, str) else '',
                    'audio_sha256': audio.get('sha256') if audio else None,
                    'native_audio_available': audio is not None,
                    'actual_model_applications': flow_state.get('actual_model_applications', {}).get(name, 0),
                    'body_steps_while_model_pending': flow_state.get('body_steps_while_model_pending', {}).get(name, 0)})
            self.cached.update(environments=environments, selected_id=pool_state['selected_env_id'],
                               updated_at=time.time(), last_checkpoints=deepcopy(self.saved), online_learning=online, **updates)
            self.cached['flow'] = {key: deepcopy(flow_state.get(key)) for key in
                ('model_backend', 'flow_closed', 'pending_model_environments', 'actual_model_applications',
                 'body_steps_while_model_pending', 'errors', 'feedback_timing')}

    def _render(self, force=False):
        snapshot = self.pool.status()
        selected = snapshot['selected_env_id']
        current = snapshot['workers'][selected]
        identity = (selected, current.get('episode'), current.get('frame_seq'))
        if not force and identity == self.jpeg_identity:
            return
        frame = self.pool.render_selected()
        pixels = base64.b64decode(frame['jpeg_base64'], validate=True)
        if hashlib.sha256(pixels).hexdigest() != frame['sha256']:
            raise RuntimeError('Selected camera JPEG identity mismatch')
        with self.lock:
            self.jpeg, self.jpeg_identity = pixels, identity

    def _start_owned(self, config):
        self.directory = self.output_root / config['pool_id']
        self.directory.mkdir(parents=True, exist_ok=False)
        _write_json(self.directory / 'request.json', config)
        self.saved = {}
        self.pool = self.pool_factory(config['configs'], directory=self.directory / 'environments')
        self.pool.start()
        self._render()
        self._publish(state='initializing', loop_paused=True)
        if self.stop_requested.is_set():
            return
        if self.flow_factory is None:
            from .native_flow import NativeOmniFlow
            factory = NativeOmniFlow
        else:
            factory = self.flow_factory
        self.flow = factory(self.pool, text=config['text'], speak=config['speak'])
        self.audio_root = Path(self.flow.directory).resolve() if hasattr(self.flow, 'directory') else None
        self._publish(state='running', loop_paused=False)

    def _stop_owned(self):
        self._online_stop_owned()
        if self.flow:
            pending = bool(self.flow.status().get('pending_model_environments'))
            self.flow.close()
            self.flow = None
            with self.lock:
                self.cached['inflight_model_requests_may_finish_without_applying'] = pending
        if self.pool:
            self._publish(state='stopping', loop_paused=True)
            failures = {}
            for env_id, worker in self.pool.status()['workers'].items():
                if env_id in self.saved or not worker.get('alive', True):
                    continue
                try:
                    self.saved[env_id] = self.pool.save(env_id)
                    _write_json(self.directory / 'brain-checkpoints.json', self.saved)
                except Exception as exc:
                    failures[env_id] = str(exc)
            if failures:
                self._publish(state='save_failed', loop_paused=True, error=f'保存未完成，原环境仍保留，可重试停止：{failures}')
                self.stop_requested.clear()
                return
            self.pool.close()
            self.pool = None
        with self.lock:
            self.jpeg = None
            self.jpeg_identity = None
            self.audio_descriptors = {}
            self.audio_root = None
        self._publish(enabled=False, state='stopped', loop_paused=True, stop_requested=False,
                      pending_selected_id=None)
        if self.directory:
            _write_json(self.directory / 'stopped.json', self.status())
        self.stop_requested.clear()
        self.stopped.set()

    def _owner(self):
        while not self.shutdown_requested.is_set():
            try:
                operation, payload = self.commands.get(timeout=.025)
            except queue.Empty:
                operation = None
            try:
                if operation == 'start':
                    self._start_owned(payload)
                elif operation == 'select' and self.pool and not self.stop_requested.is_set():
                    self.pool.select(payload)
                    self._render()
                    self._publish(pending_selected_id=None)
                elif operation in {'online_start', 'online_stop'}:
                    try:
                        self._online_start_owned() if operation == 'online_start' else self._online_stop_owned()
                        self._publish(online_requested=False)
                    except Exception as exc:
                        self._publish(online_requested=False, online_error=str(exc))
                if self.stop_requested.is_set() or operation == 'stop':
                    self._stop_owned()
                    continue
                if not self.flow or not self.pool:
                    if self.online_loop:
                        self._publish()
                    continue
                free = shutil.disk_usage(self.directory).free
                if free < self.min_free_bytes:
                    self._publish(state='paused_low_disk', loop_paused=True, disk_free_bytes=free,
                                  disk_minimum_free_bytes=self.min_free_bytes,
                                  error='环境池已暂停写入：可用磁盘空间不足；模型服务保持运行。')
                    self.stop_requested.wait(.5)
                    continue
                self.flow.tick()
                self._render()
                if self.stop_requested.is_set():
                    self._stop_owned()
                    continue
                self._publish(state='running', loop_paused=False, disk_free_bytes=free,
                              disk_minimum_free_bytes=self.min_free_bytes, error=None)
                if self.directory:
                    _write_json(self.directory / 'status.json', self.status())
            except Exception as exc:
                self._publish(state='failed', loop_paused=True, error=f'{type(exc).__name__}: {exc}')
                # Keep live state available for an explicit, checkpointed stop.
                if self.flow:
                    self.flow.close()
                    self.flow = None

    def close(self, timeout=300):
        if self._closed:
            return
        self.stop()
        if not self.stopped.wait(timeout):
            raise RuntimeError('环境池仍在完成当前步骤或保存；未强行丢弃大脑状态')
        self.shutdown_requested.set()
        self.thread.join(timeout=2)
        self._closed = True
