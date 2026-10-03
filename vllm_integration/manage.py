"""Prepare and supervise only the isolated CyberFly vLLM-Omni runtime."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'vendor/vllm-omni'
ENV = Path('/root/.cache/cyberfly/vllm-omni-env')
RUNTIME = ROOT / 'artifacts/vllm_omni_migration/runtime'
PORT = 18649
GPU = 'GPU-7c819eea-a864-1b0a-90b8-64854f446945'
SPEECH_GPU = 'GPU-ece2affe-2484-b8cb-1c7d-e3c6ceb9ec9c'
VOICE_REFERENCE = Path('/root/.cache/cyberfly/minicpm-runtime/llama.cpp-omni/tools/omni/assets/default_ref_audio/default_ref_audio.wav')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def source_identity():
    commit = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    manifest = ROOT / 'model_training/weights-manifest.json'
    return {'vllm_omni_commit': commit, 'model_revision': json.loads(manifest.read_text())['revision'],
        'weights_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _speech_sources():
    names = ('speech_cache.py', 'code2wav_plugin.py', 'worker_plugin.py', 'resource_lease.py')
    return {name: _sha(Path(__file__).with_name(name)) for name in names}


def gpu_snapshot(gpu_uuid):
    fields = subprocess.check_output(['nvidia-smi', '-i', gpu_uuid,
        '--query-gpu=uuid,memory.total,memory.used,memory.free',
        '--format=csv,noheader,nounits'], text=True, timeout=10).strip().split(',')
    if len(fields) != 4 or fields[0].strip() != gpu_uuid:
        raise RuntimeError('GPU memory query did not return the explicitly requested UUID')
    total, used, free = map(lambda x: int(x.strip()), fields[1:])
    if min(total, used, free) < 0:
        raise RuntimeError('Invalid GPU memory accounting')
    return {'gpu_uuid': gpu_uuid, 'total_mib': total, 'used_mib': used, 'free_mib': free}


def _gpu1_guard():
    """A resident HF model may share the card only after its training route is guarded."""
    endpoint = 'http://127.0.0.1:18648/health'
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(endpoint, timeout=2) as response:
            health = json.load(response)
    except (OSError, ValueError) as exc:
        # An absent HF endpoint is fine only if its lifetime training lock is
        # actually free. We never steal or alter that lock.
        from model_training.cli import training_lock_path
        with training_lock_path(SPEECH_GPU).open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return {'ready': False, 'reason': 'GPU1 resident owner is not verifiably guarded',
                        'endpoint': endpoint, 'error': str(exc)}
            return {'ready': True, 'resident_owner': 'none', 'endpoint': endpoint}
    return {'ready': health.get('gpu_uuid') == SPEECH_GPU and health.get('gpu_capacity_guard_version', 0) >= 1,
            'endpoint': endpoint, 'resident_pid': health.get('pid'),
            'gpu_capacity_guard_version': health.get('gpu_capacity_guard_version'),
            'reason': 'Resident HF training must expose capacity guard version1 before co-location'}


def inspect_resources():
    from .resource_lease import inspect_gpu_capacity
    return {'schema': 1, 'gpus': {g: gpu_snapshot(g) for g in (GPU, SPEECH_GPU)},
            'gpu1_capacity': inspect_gpu_capacity(SPEECH_GPU),
            'gpu1_resident_training_guard': _gpu1_guard(),
            'production_mutated': False}


def prepare_speech():
    """Archive exact active bytes and write a candidate; never change the running config."""
    import yaml
    active_config = RUNTIME / 'config.json'
    current = json.loads(active_config.read_text())
    active_profile = Path(current['deploy_config'])
    folder = RUNTIME / 'candidates' / ('speech-' + str(time.time_ns()))
    folder.mkdir(parents=True)
    (folder / 'baseline-config.json').write_bytes(active_config.read_bytes())
    (folder / 'baseline-profile.yaml').write_bytes(active_profile.read_bytes())
    deploy = yaml.safe_load(active_profile.read_text())
    by_id = {stage['stage_id']: stage for stage in deploy['stages']}
    if set(by_id) != {0, 1, 2} or by_id[0]['devices'] != '0' or by_id[1]['devices'] != '0':
        raise ValueError('Speech candidate requires the existing verified three-stage GPU0 baseline')
    by_id[2]['devices'] = '1'
    by_id[2]['engine_args'] = {**by_id[2].get('engine_args', {}),
                              'model_arch': 'CyberFlyMiniCPMO45Code2Wav'}
    extra = deploy['connectors']['connector_of_shared_memory']['extra']
    extra.update(codec_chunk_frames=10, codec_left_context_frames=3,
        enable_hift_graph=False, enable_cfm_graph=True, cfm_max_graphs=32,
        token2wav_float16=False, token2wav_n_timesteps=10,
        reference_cache_max_entries=2, reference_cache_max_bytes=512 * 1024**2,
        reference_feature_cache_entries=4, code2wav_allocator_limit_mib=9216)
    profile = folder / 'gpu01-cfm10-reference-cache.yaml'
    profile.write_text(yaml.safe_dump(deploy, sort_keys=False, allow_unicode=True))
    config = {**current, 'status': 'speech_candidate_not_deployed',
        'deployment_layout': 'gpu01_cfm10_reference_cache', 'deploy_config': str(profile),
        'deploy_config_sha256': _sha(profile), 'gpu_uuids': [GPU, SPEECH_GPU],
        'stage_gpu_uuids': {'0': GPU, '1': GPU, '2': SPEECH_GPU},
        'speech_source_sha256': _speech_sources(),
        'speech_resource_budget': {'minimum_free_before_start_mib': 12288,
            'maximum_added_mib': 10240, 'minimum_remaining_mib': 2048,
            'worker_torch_allocator_limit_mib': 9216,
            'cpu_reference_cache_bytes': 512 * 1024**2},
        'original_speech_scratch_retained': True, 'original_s3_retained_on_gpu': True,
        'requires_gpu1_resident_training_guard_version': 1}
    write(folder / 'candidate-config.json', config)
    candidate = {'schema': 1, 'status': 'prepared_not_activated', 'prepared_at': time.time(),
        'folder': str(folder), 'candidate_config': str(folder / 'candidate-config.json'),
        'candidate_config_sha256': _sha(folder / 'candidate-config.json'),
        'baseline_config': str(folder / 'baseline-config.json'),
        'baseline_config_sha256': _sha(folder / 'baseline-config.json'),
        'baseline_profile': str(folder / 'baseline-profile.yaml'),
        'baseline_profile_sha256': _sha(folder / 'baseline-profile.yaml'),
        'original_profile_path': str(active_profile), 'resources_at_preparation': inspect_resources(),
        'running_config_changed': False, 'model_restart_performed': False}
    write(folder / 'candidate.json', candidate)
    write(RUNTIME / 'speech-candidate.json', candidate)
    return candidate


def _candidate():
    value = json.loads((RUNTIME / 'speech-candidate.json').read_text())
    for key in ('candidate_config', 'baseline_config', 'baseline_profile'):
        if _sha(value[key]) != value[key + '_sha256']:
            raise ValueError('Prepared candidate/rollback bytes changed: ' + key)
    return value


def activate_speech():
    if status()['supervisor_running']:
        raise RuntimeError('Stop only the owned18649 supervisor before activating its candidate')
    candidate = _candidate()
    config = json.loads(Path(candidate['candidate_config']).read_text())
    preflight(config)
    write(RUNTIME / 'config.json', config)
    write(RUNTIME / 'speech-activation.json', {'status': 'activated_not_started',
        'at': time.time(), 'candidate': candidate, 'config': config})
    return {'status': 'activated_not_started', 'config': config}


def rollback():
    if status()['supervisor_running']:
        raise RuntimeError('Stop only the owned18649 supervisor before restoring its baseline')
    candidate = _candidate()
    Path(candidate['original_profile_path']).write_bytes(Path(candidate['baseline_profile']).read_bytes())
    (RUNTIME / 'config.json').write_bytes(Path(candidate['baseline_config']).read_bytes())
    return {'status': 'baseline_restored_not_started', 'config_sha256': _sha(RUNTIME / 'config.json'),
            'profile_sha256': _sha(candidate['original_profile_path']), 'other_services_touched': False}


def prepare():
    import yaml
    source = SOURCE / 'vllm_omni/deploy/minicpmo_4_5.yaml'
    deploy = yaml.safe_load(source.read_text())
    deploy.update(active_stream_window=1, enable_prefix_caching=False)
    deploy['duplex_session']['max_sessions'] = 1
    for stage in deploy['stages']:
        stage.update(devices='0', max_num_seqs=1, enforce_eager=True, enable_prefix_caching=False)
    thinker, talker, wav = deploy['stages']
    thinker.update(quantization='fp8_per_tensor', load_format='auto', dtype='bfloat16',
        gpu_memory_utilization=.60, max_model_len=2048, max_num_batched_tokens=2048,
        enable_prompt_embeds=True, enable_chunked_prefill=False)
    thinker['engine_args'] = {'model_arch': 'CyberFlyMiniCPMO45ForConditionalGeneration',
                             'safetensors_load_strategy': 'eager'}
    thinker['limit_mm_per_prompt'] = {'image': 2, 'audio': 2, 'video': 1}
    thinker['media_io_kwargs'] = {'video': {'fps': 1, 'num_frames': 8}}
    thinker['default_sampling_params']['max_tokens'] = 128
    talker.update(gpu_memory_utilization=.14, max_model_len=4096, max_num_batched_tokens=4096)
    wav.update(gpu_memory_utilization=.14)
    deploy['platforms'] = {'cuda': {'stages': [{'stage_id': 1, 'kv_cache_memory_bytes': 536870912}]}}
    connector = deploy['connectors']['connector_of_shared_memory']['extra']
    if not VOICE_REFERENCE.is_file():
        raise RuntimeError('The existing native voice reference is missing')
    connector.update(enable_hift_graph=False, enable_cfm_graph=False,
        prompt_wav=str(VOICE_REFERENCE), prompt_cache_id='cyberfly_existing_official_default')
    RUNTIME.mkdir(parents=True, exist_ok=True)
    profile = RUNTIME / 'gpu0-three-stage.yaml'
    profile.write_text(yaml.safe_dump(deploy, sort_keys=False, allow_unicode=True))
    model = ROOT / 'model_training/models'
    for item in json.loads((ROOT / 'model_training/weights-manifest.json').read_text())['files']:
        if item['path'].endswith('.safetensors') and (model / item['path']).stat().st_size != item['size']:
            raise RuntimeError('Existing model shard size mismatch')
    identity = source_identity()
    config = {**identity, 'schema': 1, 'status': 'prepared_not_verified', 'gpu_uuid': GPU,
        'port': PORT, 'model': str(model), 'deploy_config': str(profile),
        'python': str(ENV / 'bin/python'), 'vllm': str(ENV / 'bin/vllm'),
        'chat_template': str(SOURCE / 'vllm_omni/transformers_utils/chat_templates/minicpmo45_native.jinja'),
        'served_model_name': 'cyberfly-minicpm-o-4.5', 'original_weights_reused': True,
        'original_base_weights_frozen': True, 'max_concurrent_requests_first_validation': 1,
        'native_brain_plugin_verified': False, 'main_lab_switched': False,
        'thinker_quantization': 'fp8_per_tensor', 'thinker_load_format': 'auto',
        'thinker_safetensors_load_strategy': 'eager',
        'disabled_kernels': ['CutlassFP8ScaledMMLinearKernel'],
        'sampling_backend': 'vllm_native_topk_topp',
        'voice_reference': {'path': str(VOICE_REFERENCE),
                            'sha256': hashlib.sha256(VOICE_REFERENCE.read_bytes()).hexdigest()},
        'quantization_note': 'Official vLLM online FP8 weight quantization; differs from the existing HF NF4 runtime. GPU loading and numerical validation remain pending.'}
    write(RUNTIME / 'config.json', config)
    return config


def owned(pid):
    try:
        args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        return b'vllm_integration.manage' in args and b'run' in args and Path(f'/proc/{pid}/cwd').resolve() == ROOT
    except OSError:
        return False


def status():
    path = RUNTIME / 'process.json'
    process = json.loads(path.read_text()) if path.is_file() else {}
    result = {**process, 'supervisor_running': bool(process.get('pid') and owned(process['pid']))}
    exit_path = RUNTIME / 'exit.json'
    if not result['supervisor_running'] and exit_path.is_file():
        ended = json.loads(exit_path.read_text())
        if ended.get('finished_at', 0) >= process.get('started_at', 0):
            result.update(status='stopped' if ended.get('exit_code') == 0 else 'failed', **ended)
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/v1/models', timeout=1) as response:
            result['models'] = json.load(response)
        result['api_ready'] = True
        result['status'] = 'ready'
    except (OSError, ValueError):
        result['api_ready'] = False
    return result


def preflight(config, *, capacity_held=False):
    if config['gpu_uuid'] != GPU or config['port'] != PORT:
        raise ValueError('This isolated pilot owns only its declared GPU0 and port')
    if source_identity()['vllm_omni_commit'] != config['vllm_omni_commit']:
        raise ValueError('Official source changed; prepare a new reviewable profile')
    if not Path(config['vllm']).is_file():
        raise RuntimeError('The isolated vLLM-Omni environment is not ready')
    info = gpu_snapshot(GPU)
    if info['free_mib'] < 22000:
        raise RuntimeError('GPU0 needs at least 22000 MiB free for this first measured pilot')
    if config.get('deployment_layout') == 'gpu01_cfm10_reference_cache':
        from .resource_lease import require_gpu_training_available
        if config.get('gpu_uuids') != [GPU, SPEECH_GPU]:
            raise ValueError('Speech deployment must expose only the two declared UUIDs')
        if _sha(config['deploy_config']) != config['deploy_config_sha256']:
            raise ValueError('Prepared speech deployment profile changed')
        if config.get('speech_source_sha256') != _speech_sources():
            raise ValueError('Speech extension source changed; prepare a fresh candidate')
        budget = config['speech_resource_budget']
        if budget['maximum_added_mib'] != 10240 or budget['minimum_free_before_start_mib'] < 12288:
            raise ValueError('Unreviewed Code2Wav memory budget')
        speech = gpu_snapshot(SPEECH_GPU)
        if speech['free_mib'] < budget['minimum_free_before_start_mib']:
            raise RuntimeError('显卡1空余显存不足12GiB；原模型与训练任务均保持不动。')
        if not _gpu1_guard()['ready']:
            raise RuntimeError('显卡1原模型须先安全重载训练容量保护，再启动共享语音服务。')
        if not capacity_held:
            require_gpu_training_available(SPEECH_GPU)
    with socket.socket() as probe:
        try:
            probe.bind(('127.0.0.1', PORT))
        except OSError:
            raise RuntimeError('Pilot port is occupied; no existing service was touched')


def start():
    if 'microsoft' in Path('/proc/sys/kernel/osrelease').read_text().lower():
        from .network_route import ensure_local_route
        ensure_local_route()
    current = status()
    if current['supervisor_running']:
        return current
    config = json.loads((RUNTIME / 'config.json').read_text())
    preflight(config)
    with (RUNTIME / 'supervisor.log').open('ab') as log:
        process = subprocess.Popen([sys.executable, '-u', '-m', 'vllm_integration.manage', 'run'], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(RUNTIME / 'process.json', {'pid': process.pid, 'status': 'starting', 'gpu_uuid': GPU, 'port': PORT,
        'started_at': time.time(), 'project': str(ROOT)})
    return {'status': 'starting', 'pid': process.pid, 'port': PORT}


def run():
    config = json.loads((RUNTIME / 'config.json').read_text())
    from model_training.cli import training_lock_path
    from .resource_lease import vocoder_capacity
    speech = config.get('deployment_layout') == 'gpu01_cfm10_reference_cache'
    with ExitStack() as resources:
        lock = resources.enter_context(training_lock_path(GPU).open('a'))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lease = resources.enter_context(vocoder_capacity(SPEECH_GPU, 10240)) if speech else None
        preflight(config, capacity_held=speech)
        memory_before = gpu_snapshot(SPEECH_GPU) if speech else None
        visible = ','.join(config['gpu_uuids']) if speech else GPU
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': visible, 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
            'PATH': str(ENV / 'bin') + os.pathsep + os.environ.get('PATH', ''),
            'HF_HUB_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false',
            'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
            'PYTHONPATH': str(ROOT) + os.pathsep + str(SOURCE),
            'VLLM_WORKER_MULTIPROC_METHOD': 'spawn',
            # vLLM 0.29's Python CUTLASS guard admits SM86 even though the
            # FP8 kernel fails on this GPU. The official selector then uses
            # its compatible Marlin implementation without a vendor patch.
            'VLLM_DISABLED_KERNELS': 'CutlassFP8ScaledMMLinearKernel',
            'VLLM_USE_FLASHINFER_SAMPLER': '0'}
        command = [config['vllm'], 'serve', config['model'], '--omni', '--trust-remote-code',
            '--deploy-config', config['deploy_config'], '--chat-template', config['chat_template'],
            '--chat-template-content-format', 'openai', '--served-model-name', config['served_model_name'],
            '--host', '127.0.0.1', '--port', str(PORT)]
        write(RUNTIME / 'launch.json', {'command': command, 'config': config, 'gpu_uuid': GPU,
            'gpu_uuids': config.get('gpu_uuids', [GPU]), 'started_at': time.time(),
            'speech_capacity_lease': lease, 'speech_memory_before': memory_before})
        with (RUNTIME / 'server.log').open('ab') as log:
            child = subprocess.Popen(command, cwd=SOURCE, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            def signal_owned_children(sig):
                try:
                    os.killpg(child.pid, sig)
                except ProcessLookupError:
                    pass
            signal.signal(signal.SIGTERM, lambda sig, frame: signal_owned_children(sig))
            signal.signal(signal.SIGINT, lambda sig, frame: signal_owned_children(sig))
            try:
                peak_added = 0
                samples = 0
                failure = None
                stop_requested = None
                query_errors = 0
                while child.poll() is None:
                    try:
                        child.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        pass
                    if speech and child.poll() is None and failure is None:
                        try:
                            snapshot = gpu_snapshot(SPEECH_GPU)
                            query_errors = 0
                            samples += 1
                            added = max(0, snapshot['used_mib'] - memory_before['used_mib'])
                            peak_added = max(peak_added, added)
                            budget = config['speech_resource_budget']
                            if added > budget['maximum_added_mib'] or snapshot['free_mib'] < budget['minimum_remaining_mib']:
                                failure = {'reason': 'speech_gpu_capacity_limit', 'actual': snapshot,
                                    'added_mib': added, 'budget': budget, 'stopped_only_owned18649': True}
                        except (OSError, ValueError, subprocess.SubprocessError) as exc:
                            query_errors += 1
                            if query_errors >= 3:
                                failure = {'reason': 'cannot_verify_speech_memory', 'error': str(exc),
                                    'stopped_only_owned18649': True}
                        if failure:
                            write(RUNTIME / 'speech-memory-guard.json', {**failure, 'at': time.time()})
                            signal_owned_children(signal.SIGTERM)
                            stop_requested = time.monotonic()
                        if samples % 5 == 0 or failure:
                            write(RUNTIME / 'speech-resource-status.json', {'schema': 1, 'pid': os.getpid(),
                                'gpu_uuid': SPEECH_GPU, 'baseline': memory_before, 'peak_added_mib': peak_added,
                                'samples': samples, 'failure': failure, 'updated_at': time.time()})
                    if failure and stop_requested is not None and time.monotonic() - stop_requested > 30:
                        # The child owns its vLLM worker lifecycle. Never signal an
                        # unrelated GPU process or the resident HF service.
                        signal_owned_children(signal.SIGKILL)
                code = child.returncode
            finally:
                # Release the capacity lease only after this exclusively owned
                # child process group has stopped, including failed API parents.
                signal_owned_children(signal.SIGTERM)
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    try:
                        os.killpg(child.pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(.05)
                else:
                    signal_owned_children(signal.SIGKILL)
                if child.poll() is None:
                    child.wait(timeout=5)
        write(RUNTIME / 'exit.json', {'exit_code': code, 'finished_at': time.time(),
            'speech_capacity_failure': failure, 'speech_peak_added_mib': peak_added,
            'speech_capacity_lease_released_on_exit': bool(lease)})
        return {'exit_code': code}


def stop():
    current = status()
    if not current.get('supervisor_running'):
        return {'status': 'not_running'}
    pid = current['pid']
    if os.getpgid(pid) != pid:
        raise RuntimeError('Unexpected pilot process group; no signal sent')
    os.killpg(pid, signal.SIGTERM)
    return {'status': 'stop_requested', 'pid': pid, 'other_services_touched': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['prepare', 'prepare_speech', 'activate_speech',
        'rollback', 'inspect_resources', 'start', 'run', 'stop', 'status'])
    args = parser.parse_args()
    print(json.dumps(globals()[args.operation](), ensure_ascii=False, indent=2))
