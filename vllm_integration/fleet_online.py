"""Fleet-owned measured replay; no model-generated text is used as a label."""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
BUSY_BEFORE_DISPATCH = '隐空间语言适配器已经在训练'


class QueuedLanguageTraining:
    """Share the existing Jobs GPU1 gate without ambiguous automatic retries."""
    def __init__(self, jobs, *, scope='fleet'):
        if scope not in ('fleet', 'lab'):
            raise ValueError('Training queue requires an explicit Lab or Fleet owner')
        self.jobs = jobs
        self.scope = scope
        self.lock = threading.Lock()
        self.records = {}

    def submit(self, manifest, request):
        identity = self.scope + '-online-' + uuid.uuid4().hex[:16]
        with self.lock:
            self.records[identity] = {'state': 'queued', 'actual_job_id': None, 'cancel': threading.Event(),
                                      'created_at': time.time(), 'error': None}
            completed = [key for key, row in self.records.items() if row['state'] in {'completed', 'failed', 'cancelled', 'interrupted'}]
            for key in completed[:-64]:
                self.records.pop(key)
        def dispatch():
            from .resource_lease import GPUCapacityBusy
            row = self.records[identity]
            config = {key: value for key, value in request.items() if key != 'activate_after_training'}
            config['manifest_path'] = str(manifest)
            while not row['cancel'].is_set():
                if self.jobs.language_adapter_training_lock.locked():
                    row['cancel'].wait(.5)
                    continue
                try:
                    result = self.jobs.launch_language_adapter(config)
                except GPUCapacityBusy as exc:
                    # Typed local preflight rejection, before job creation.
                    with self.lock:
                        row['queue_reason'] = str(exc)
                    row['cancel'].wait(.5)
                    continue
                except ValueError as exc:
                    if str(exc) == BUSY_BEFORE_DISPATCH:
                        # This exact existing guard runs before job creation.
                        row['cancel'].wait(.5)
                        continue
                    with self.lock:
                        row.update(state='failed', error=str(exc))
                    return
                except Exception as exc:
                    # A launch may have reached the GPU: never retry it.
                    with self.lock:
                        row.update(state='failed', error='Dispatch outcome not safely retryable: ' + str(exc))
                    return
                with self.lock:
                    row.update(state='submitted', actual_job_id=result['job_id'])
                if row['cancel'].is_set():
                    self.jobs.cancel(result['job_id'])
                return
            with self.lock:
                row['state'] = 'cancelled'
        threading.Thread(target=dispatch, name=identity, daemon=True).start()
        return identity

    def poll(self, identity):
        with self.lock:
            row = dict(self.records[identity])
        if not row['actual_job_id']:
            return {'state': row['state'], 'error': row['error'], 'job_id': identity,
                    'queue_reason': row.get('queue_reason', '等待显卡 1 的训练容量'), 'gpu_index': 1}
        result = self.jobs.poll_language_adapter(row['actual_job_id'])
        with self.lock:
            self.records[identity]['state'] = result.get('state', result.get('status', 'unknown'))
        return {**result, 'fleet_queue_id': identity, 'actual_job_id': row['actual_job_id'], 'gpu_index': 1}

    def cancel(self, identity):
        with self.lock:
            row = self.records[identity]
            row['cancel'].set()
            actual = row['actual_job_id']
        return self.jobs.cancel(actual) if actual else {'cancel_requested': True, 'state': 'queued_cancellation'}


class FleetOnlineCapture:
    """Observe one actual pool transition every two seconds, round-robin.

    Collector code invokes no model. An external Omni controller can have
    produced the action or can be generating concurrently; both are explicit
    metadata. All targets are later computed by the unchanged grounding verifier
    from original physical packets and 2,129 annotated output spike counts.
    """
    def __init__(self, pool, loop, *, pool_id, flow=None, interval_s=2.0,
                 controller_kind='vllm_omni'):
        self.pool, self.loop, self.pool_id = pool, loop, pool_id
        self.flow = flow
        self.interval_s = float(interval_s)
        if self.interval_s < 2:
            raise ValueError('Fleet replay samples at most one transition every two seconds.')
        self.controller_kind = controller_kind
        self.adapters = {}
        self.last_capture = 0.
        self.next_index = 0
        self.error = None
        self.collected = 0
        self.pending_samples = {}

    def _retention(self):
        return getattr(self.flow, 'retention', None) or getattr(self.flow, 'raw_retention', None)

    def _discard(self, folder):
        try:
            self.loop.discard_sample_dir(folder)
        except Exception as exc:
            self.error = (self.error or '') + '; sample retained after cleanup refusal: ' + str(exc)

    def _copy_brain(self, descriptor, folder, name):
        from connectome_adapter.hot_state import read_observation_snapshot
        _, raw = read_observation_snapshot(descriptor)
        destination = folder / (name + '.npz')
        with destination.open('xb') as stream: stream.write(raw)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != descriptor['sha256']:
            raise ValueError('Persistent learner brain copy failed complete SHA verification')
        result = {key: deepcopy(value) for key, value in descriptor.items() if key != 'hot_storage'}
        result.update(path=str(destination), temporary=False)
        if descriptor.get('hot_storage'):
            result['source_hot_snapshot'] = {'path': descriptor['path'], 'sha256': descriptor['sha256'],
                'hot_storage': deepcopy(descriptor['hot_storage']), 'persisted_before_lease_release': True}
        return result

    def _body(self, env_id, descriptor, folder):
        from model_training.grounding_data import read_checked
        from shared_io.body_senses import BodySenseAdapter
        packet = json.loads(read_checked(descriptor['path'], descriptor['sha256'], limit=8*1024*1024))
        adapter = self.adapters.get(env_id)
        if adapter is None or adapter.codec.layout_sha != packet['model_layout_sha256']:
            adapter = self.adapters[env_id] = BodySenseAdapter(packet)
        return adapter.export(packet, folder)

    def before_steps(self, captures, commands):
        if not self.loop.can_accept() or time.monotonic() - self.last_capture < self.interval_s or not commands:
            return {}
        ids = list(commands)
        env_id = ids[self.next_index % len(ids)]
        self.next_index += 1
        self.last_capture = time.monotonic()
        folder = None
        protected = None
        retention = self._retention()
        try:
            folder = self.loop.allocate_sample_dir()
            owned_before = env_id not in captures and not commands[env_id].get('input_id')
            owned_feedback = not commands[env_id].get('capture_neural_state', False)
            # A pre-existing model input is returned with the same pinned ID.
            before = self.pool.capture_inputs([env_id], include_neural_state=True)[env_id]
            commands[env_id]['input_id'] = before['input_id']
            commands[env_id]['capture_neural_state'] = True
            if retention:
                if owned_before:
                    protected = retention.register_observation_copy(env_id, before,
                        consumer='learner-before-' + folder.name)
                else:
                    protected = 'learner-copy-before:' + folder.name
                    retention.protect(protected, [before])
            sample = self.copy_before(env_id, before, folder)
            sample['collector_source_ownership'] = {'before': owned_before, 'feedback': owned_feedback}
            sample['model_pending_at_capture'] = env_id in (self.flow.status().get('pending_model_environments', []) if self.flow else [])
            self.pending_samples[str(folder)] = sample
            return {env_id: sample}
        except Exception as exc:
            self.error = str(exc)
            if folder is not None:
                self._discard(folder)
            return {}
        finally:
            if protected:
                try: retention.release(protected)
                except Exception as exc: self.error = 'Retention lease release failed: ' + str(exc)

    def copy_before(self, env_id, before, folder):
        """Also usable to validate format against an immutable actual archive."""
        import numpy as np
        from PIL import Image
        from model_training.grounding_data import read_checked
        raw = read_checked(before['rgb_file']['path'], before['rgb_file']['sha256'], limit=12*1024*1024)
        rgb = np.load(io.BytesIO(raw), allow_pickle=False)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError('Collector requires original RGB uint8 pixels.')
        rgb_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
        if rgb_sha != before['rgb_file']['raw_rgb_sha256']:
            raise ValueError('Original camera pixels changed before collection.')
        image = folder / 'image.png'
        Image.fromarray(rgb).save(image)
        features = self._body(env_id, before['body_packet_file'], folder / 'body-before')
        brain = self._copy_brain(before['neural_state_file'], folder, 'brain-before')
        episode = f'fleet/{self.pool_id}/{env_id}/episode/{before["episode"]}'
        return {'folder': folder, 'env_id': env_id, 'input_id': before['input_id'],
                'episode_number': before['episode'], 'episode_id': episode,
                'image_path': str(image), 'image_rgb_sha256': rgb_sha,
                'body_features_file': features, 'brain_state_before': brain}

    def copy_after(self, sample, result):
        from model_training.collect_grounding import write
        folder, env_id = sample['folder'], sample['env_id']
        feedback = result.get('feedback')
        if not feedback or feedback.get('source_input_id') != sample['input_id'] or feedback['episode'] != sample['episode_number']:
            raise ValueError('Collected after-state is from another input or episode.')
        if result['info']['neural']['input_rgb_sha256'] != sample['image_rgb_sha256']:
            raise ValueError('The actual brain saw a different camera input.')
        body = self._body(env_id, feedback['body_packet_file'], folder / 'body-after')
        brain = self._copy_brain(feedback['neural_state_file'], folder, 'brain-after')
        info = result['info']
        row = {key: value for key, value in sample.items() if key not in {'folder', 'episode_number'}}
        row.update(event_id=folder.name, created_at=datetime.now(timezone.utc).isoformat(),
            step_index=int(info['step']), collection_source='brain_body_simulation_not_shared_model_event',
            brain_state_after=brain, body_feedback_file=body, actual_neural=info['neural']['neural'],
            brain_learning=info['neural']['learning'], original_visual_input_enabled=bool(info['visual_input_enabled']),
            original_neural_drive=info['applied_neural_drive'], general_stimulation=info['general_stimulation'],
            io_scope='sensory_motor', trajectory_config={'source': 'actual_independent_fleet', 'pool_id': self.pool_id,
                'env_id': env_id, 'episode_id': sample['episode_id'], 'scenario': info['scenario']},
            terminated=result['terminated'], truncated=result['truncated'], model_called=False,
            model_called_field_meaning='collector_did_not_invoke_model', external_model_may_have_driven_body=True,
            external_controller=self.controller_kind,
            actual_model_input_applied=bool(result.get('parallel_inputs', {}).get('actual_sum_verified')),
            capture_controller='fleet_transition_observer',
            collection_note='Only actual before/after measurements become labels. External model requests may run concurrently and model outputs may drive the body; they are never teaching targets.')
        source = folder / 'source.json'
        write(source, row)
        return {**row, 'source_path': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}

    def after_steps(self, samples, results):
        for env_id, sample in (samples or {}).items():
            protected = None
            retention = self._retention()
            try:
                result = results[env_id]
                if retention:
                    if sample.get('collector_source_ownership', {}).get('feedback'):
                        protected = retention.register_observation_copy(env_id, result['feedback'],
                            consumer='learner-after-' + sample['folder'].name)
                    else:
                        protected = 'learner-copy-after:' + sample['folder'].name
                        retention.protect(protected, [result.get('feedback')])
                row = self.copy_after(sample, result)
                outcome = self.loop.offer(row, episode_id=row['episode_id'], owned_source_dir=sample['folder'])
                self.collected += int(outcome['accepted'])
                self.error = None
            except Exception as exc:
                self.error = str(exc)
                self._discard(sample['folder'])
            finally:
                self.pending_samples.pop(str(sample['folder']), None)
                if protected:
                    try: retention.release(protected)
                    except Exception as exc: self.error = 'Retention lease release failed: ' + str(exc)

    def close(self):
        """Release only incomplete, unqueued samples after the owner stops steps."""
        for folder in list(self.pending_samples):
            self._discard(folder)
            self.pending_samples.pop(folder)

    def status(self):
        return {'data_source': 'independent_fleet_actual_transitions', 'pool_id': self.pool_id,
                'sampling_interval_s': self.interval_s, 'submitted_for_verification': self.collected,
                'error': self.error, 'pending_body_transitions': len(self.pending_samples),
                'target_source': 'unchanged measured-body and actual output-neuron verifier',
                'model_outputs_used_as_labels': False, 'retention_copies_owned_by_learner': True}


def create_fleet_online(pool, jobs, *, pool_id, flow):
    from model_training.policy import require_llm_training_allowed
    require_llm_training_allowed()
    from model_training.online_learning import OnlineLearningLoop
    dispatcher = QueuedLanguageTraining(jobs)
    loop = OnlineLearningLoop(ROOT / 'datasets/latent_grounding/fleet_online' / uuid.uuid4().hex,
        submit_training=dispatcher.submit, poll_training=dispatcher.poll, cancel_training=dispatcher.cancel,
        min_new_records=4, train_steps=8, replay_capacity=32, max_pending=2)
    observer = FleetOnlineCapture(pool, loop, pool_id=pool_id, flow=flow)
    loop.start()
    return loop, observer, dispatcher
