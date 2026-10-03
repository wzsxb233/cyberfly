"""Continuous CPU embodiment around actual streaming Omni hidden states.

The body never waits for a language reply. A generated hidden update affects
one future body step; that body's later measured state conditions the next
model turn. Current-turn prompt/KV history is never retroactively rewritten.
"""
from __future__ import annotations
from concurrent.futures import CancelledError, ThreadPoolExecutor
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import re
import time
import threading
import uuid

ROOT = Path(__file__).resolve().parents[1]


class RequestPreparation:
    """Single publication of request-private heads and verified input files."""
    def __init__(self):
        self._lock = threading.Lock()
        self._bundle = None

    def publish(self, bundle):
        saved = {key: value if key in {'coupler', 'body_head'} else deepcopy(value)
                 for key, value in bundle.items()}
        with self._lock:
            if self._bundle is not None:
                raise RuntimeError('Request preparation was already published')
            self._bundle = saved

    def snapshot(self):
        with self._lock:
            bundle = self._bundle
        return None if bundle is None else {
            key: value if key in {'coupler', 'body_head'} else deepcopy(value)
            for key, value in bundle.items()}


class NativeOmniFlow:
    def __init__(self, pool, *, text='观察眼前环境，自由探索，用一句简短中文描述变化。', speak=False,
                 max_tokens=32, model_interval_s=2, hidden_ttl_s=5, transition_observer=None,
                 io_scope='sensory_motor', audio_path=None, inference_cache=None):
        if not pool.started or pool.closed: raise ValueError('Start the owned CPU environment pool first')
        if not isinstance(text, str) or not 1 <= len(text) <= 1000: raise ValueError('Use a short actual instruction')
        self.pool, self.text, self.speak, self.max_tokens = pool, text, bool(speak), max_tokens
        if io_scope not in {'sensory_motor', 'all_neurons'}:
            raise ValueError('Choose a numerical neuron scope')
        self.io_scope = io_scope
        self.interval, self.ttl = float(model_interval_s), float(hidden_ttl_s)
        self.transition_observer = transition_observer
        self.executor = ThreadPoolExecutor(max_workers=len(pool.workers), thread_name_prefix='cyberfly-omni-request')
        self.turns, self.couplers, self.body_heads, self.previous_body = {}, {}, {}, {}
        self.inference_cache = inference_cache
        from shared_io.sensory_sources import SharedSensorySources
        self.sensory_sources = inference_cache.sensory_sources if inference_cache else SharedSensorySources()
        self.retired = {}
        self.next_turn = {key: 0. for key in pool.workers}
        self.responses, self.errors, self.applied, self.steps_during_model = {}, {}, {}, {}
        self.closed = False
        self.directory = ROOT / 'artifacts/vllm_omni_migration/flows' / uuid.uuid4().hex[:12]
        self.directory.mkdir(parents=True)
        self.audio_path = None
        if audio_path is not None:
            from .client import OmniClient
            raw, _ = OmniClient._media(audio_path)
            self.audio_path = self.directory / 'input-audio.wav'
            self.audio_path.write_bytes(raw)
        from .retention import NativeRawRetention
        self.retention = NativeRawRetention(self.directory, pool.directory)
        self.events = []

    def _prepare(self, env_id, observation):
        # The existing multi-process fleet keeps its original execution path.
        # Only the Lab has an owner cache that pins immutable request versions.
        if self.inference_cache is None:
            return self._prepare_synchronous(env_id, observation)
        from .client import OmniClient
        from .streaming import LatestProgress
        observation = deepcopy(observation)
        event_id = uuid.uuid4().hex
        folder = self.directory / env_id / event_id
        started = time.perf_counter()
        heads = self.inference_cache.request_heads(observation['neural_state_file'],
            observation['body_packet_file'], io_scope=self.io_scope)
        previous_body = deepcopy(self.previous_body.get(env_id))
        lease = 'preparation:' + env_id + ':' + event_id
        client = OmniClient(timeout=180)
        preparation, cancel = RequestPreparation(), threading.Event()
        progress = LatestProgress()
        turn = {'event_id': event_id, 'env_id': env_id, 'episode': observation['episode'],
            'client_request_id': uuid.uuid4().hex, 'expected_model': client.model,
            'observation': observation, 'started_at': time.time(),
            'last_applied_sequence': -1, 'verified_applications': [], 'directory': str(folder),
            'interval_semantics': 'Previous model observation to current observation; original senses continue between turns'}
        request = {'turn': deepcopy(turn), 'heads': heads, 'previous_body': previous_body,
            'text': self.text, 'audio_path': self.audio_path, 'speak': self.speak,
            'max_tokens': self.max_tokens, 'lease': lease,
            'owner_pin_seconds': time.perf_counter() - started,
            'queued_at': time.time()}
        self.retention.protect(lease, [observation, previous_body])
        try:
            request['owner_pin_seconds'] = time.perf_counter() - started
            request['queued_at'] = time.time()
            future = self.executor.submit(self._run_request, request, client, preparation, progress, cancel)
        except BaseException:
            self.retention.release(lease)
            raise
        self.turns[env_id] = {**turn, 'future': future, 'progress': progress,
                            'preparation': preparation, 'cancel_event': cancel}
        self.steps_during_model.setdefault(env_id, 0)
        self.applied.setdefault(env_id, 0)

    def _run_request(self, request, client, preparation, progress, cancel):
        """Background owns preparation; it never accesses a body or brain client."""
        import numpy as np
        from PIL import Image
        from .latent_contract import checked_file, write_json
        turn = request['turn']
        env_id, event_id = turn['env_id'], turn['event_id']
        observation = turn['observation']
        folder = Path(turn['directory'])
        registered = False
        created = []
        started_at = time.time()
        try:
            if cancel.is_set():
                raise CancelledError('Request cancelled before preparing its captured input')
            folder.mkdir(parents=True)
            # All bytes are still independently checked in this consumer.
            _, raw = checked_file(observation['body_packet_file'])
            packet = json.loads(raw)
            _, raw_rgb = checked_file(observation['rgb_file'])
            rgb = np.load(io.BytesIO(raw_rgb), allow_pickle=False)
            if hashlib.sha256(rgb.tobytes()).hexdigest() != observation['rgb_file']['raw_rgb_sha256']:
                raise ValueError('Observed actual pixels changed before model submission')
            image = folder / 'input.png'
            Image.fromarray(rgb).save(image)
            created.append(str(image))
            coupler, body_head = request['heads']
            current_body = body_head.export(packet, folder / 'body-observation')
            created.extend(current_body[key] for key in ('path', 'packet_path', 'layout_path'))
            condition = coupler.condition(event_id, observation['neural_state_file'],
                body_features_file=request['previous_body'] or current_body, body_feedback_file=current_body)
            created.append(condition['path'])
            timing = {'owner_pin_seconds': request['owner_pin_seconds'],
                'queued_at': request['queued_at'], 'worker_started_at': started_at,
                'worker_thread_id': threading.get_ident(), 'input_captured_before_background': True}
            prepared_turn = {**turn, 'condition': condition, 'body_observation': current_body,
                             'preparation_timing': timing}
            write_json(folder / 'request-source.json', {**prepared_turn, 'text': request['text'],
                                                       'model_backend': 'vllm_omni'})
            self.retention.register_turn(env_id, event_id, observation,
                created_descriptors=[current_body, condition],
                created_files=[image, current_body['packet_path'], current_body['layout_path']])
            registered = True
            # After this publication the worker never mutates these heads again.
            preparation.publish({'condition': condition, 'body_observation': current_body,
                'coupler': coupler, 'body_head': body_head,
                'prepared_at': time.time(), 'preparation_timing': timing})
            if cancel.is_set():
                raise CancelledError('Request cancelled after preparing its captured input')
            return client.generate(request['text'], image_path=image, audio_path=request['audio_path'],
                speak=request['speak'], max_tokens=request['max_tokens'], condition=condition,
                output=folder / 'response', stream=True, on_progress=progress.publish,
                request_id=turn['client_request_id'])
        except BaseException as exc:
            # Failed Lab turns stop the live flow. Keep their exact producer
            # returns for diagnosis; do not scan directories or claim a partial
            # producer write was successfully registered or deleted.
            try:
                write_json(folder / 'request-failed.json', {'event_id': event_id,
                    'error': f'{type(exc).__name__}: {exc}', 'registered': registered,
                    'known_created_paths': created, 'failed_products_retained_for_diagnosis': True})
            except OSError:
                pass  # A disk failure must not hide the original request error.
            raise
        finally:
            # Cooperative cancellation always lets this worker finish its own
            # leases. Queued futures are never cancelled out from under it.
            try:
                if not registered:
                    self.retention.discard_unsubmitted_observation(env_id, observation)
            finally:
                self.retention.release(request['lease'])

    def _adopt_preparation(self, env_id, turn):
        pending = turn.get('preparation')
        if pending is None or 'condition' in turn:
            return True
        bundle = pending.snapshot()
        if bundle is None:
            return False
        turn.update({key: bundle[key] for key in ('condition', 'body_observation',
                                                'prepared_at', 'preparation_timing')})
        self.couplers[env_id] = bundle['coupler']
        self.body_heads[env_id] = bundle['body_head']
        self.previous_body[env_id] = bundle['body_observation']
        self.retention.protect('previous_body:' + env_id, bundle['body_observation'])
        return True

    @staticmethod
    def _cancel_turn(turn):
        if 'cancel_event' in turn:
            turn['cancel_event'].set()
        else:
            turn['future'].cancel()

    def _complete_turn(self, env_id, turn):
        if not turn['future'].done():
            raise RuntimeError('Cannot complete a request while its actual worker is running')
        preparation = turn.get('preparation')
        if preparation is None or preparation.snapshot() is not None:
            self.retention.complete_turn(env_id, turn['event_id'], future_done=True)
        if turn.get('release_previous_body'):
            self.retention.release('previous_body:' + env_id)

    def _prepare_synchronous(self, env_id, observation):
        import numpy as np
        from PIL import Image
        from .latent_contract import LatentCoupler, checked_file, write_json
        from .client import OmniClient
        event_id = uuid.uuid4().hex
        folder = self.directory / env_id / event_id
        folder.mkdir(parents=True)
        _, raw = checked_file(observation['body_packet_file'])
        packet = json.loads(raw)
        _, raw_rgb = checked_file(observation['rgb_file'])
        rgb = np.load(io.BytesIO(raw_rgb), allow_pickle=False)
        if hashlib.sha256(rgb.tobytes()).hexdigest() != observation['rgb_file']['raw_rgb_sha256']:
            raise ValueError('Observed actual pixels changed before model submission')
        image = folder / 'input.png'
        Image.fromarray(rgb).save(image)
        brain = observation['neural_state_file']
        if self.inference_cache is not None:
            self.couplers[env_id] = self.inference_cache.coupler(brain, io_scope=self.io_scope)
        elif env_id not in self.couplers:
            self.couplers[env_id] = LatentCoupler(brain, io_scope=self.io_scope)
        self.body_heads[env_id] = self.sensory_sources.body_adapter(packet)
        current_body = self.body_heads[env_id].export(packet, folder / 'body-observation')
        condition = self.couplers[env_id].condition(event_id, brain,
            body_features_file=self.previous_body.get(env_id, current_body), body_feedback_file=current_body)
        client = OmniClient(timeout=180)
        client_request_id = uuid.uuid4().hex
        turn = {'event_id': event_id, 'env_id': env_id, 'episode': observation['episode'],
            'client_request_id': client_request_id, 'expected_model': client.model,
            'condition': condition, 'observation': observation, 'started_at': time.time(),
                    'last_applied_sequence': -1, 'verified_applications': [],
                    'directory': str(folder), 'body_observation': current_body,
            'interval_semantics': 'Previous model observation to current observation; original senses continue between turns'}
        write_json(folder / 'request-source.json', {**turn, 'text': self.text, 'model_backend': 'vllm_omni'})
        self.retention.register_turn(env_id, event_id, observation,
            created_descriptors=[current_body, condition],
            created_files=[image, current_body['packet_path'], current_body['layout_path']])
        from .streaming import LatestProgress
        progress = LatestProgress()
        future = self.executor.submit(client.generate, self.text, image_path=image,
            audio_path=self.audio_path, speak=self.speak, max_tokens=self.max_tokens,
            condition=condition, output=folder / 'response', stream=True, on_progress=progress.publish,
            request_id=client_request_id)
        self.turns[env_id] = {**turn, 'future': future, 'progress': progress}
        self.previous_body[env_id] = current_body
        self.retention.protect('previous_body:' + env_id, current_body)
        self.steps_during_model.setdefault(env_id, 0)
        self.applied.setdefault(env_id, 0)

    def _latest(self, env_id):
        import numpy as np
        from .latent_contract import read_latest_hidden, write_json, sha
        turn = self.turns.get(env_id)
        if not turn: return None
        if not self._adopt_preparation(env_id, turn): return None
        try:
            found = read_latest_hidden(turn['event_id'], request_id=turn.get('request_id'), max_age_s=self.ttl)
        except FileNotFoundError:
            return None  # The model has not published a first forward yet.
        except ValueError as exc:
            if str(exc) == 'Latest model hidden expired':
                return None  # Never replay an old current when generation stalls.
            raise
        if found is None: return None
        hidden, descriptor = found
        if descriptor['sequence'] <= turn['last_applied_sequence']: return None
        if descriptor['event_id'] != turn['event_id']: raise ValueError('Hidden state belongs to another environment turn')
        self._check_hidden_request(turn, descriptor.get('request_id'))
        condition = turn['condition']
        for key in ('model_revision', 'weights_manifest_sha256', 'vllm_omni_commit',
                    'io_scope', 'codec_sha256', 'coupler_sha256', 'coupler_generation',
                    'read_ids_sha256', 'write_ids_sha256', 'soft_tokens_sha256'):
            if descriptor.get(key) != condition.get(key):
                raise ValueError('Actual model hidden identity differs from its numerical input: ' + key)
        if descriptor.get('condition_file_sha256') != condition['sha256']:
            raise ValueError('Model used a different numerical input file')
        if descriptor.get('same_original_latent_returned_to_native_tts') is not True:
            raise ValueError('Original native TTS latent identity was not retained')
        if descriptor.get('original_model_trainable_parameters') != 0:
            raise ValueError('Original model weights are not frozen')
        if 'request_id' in turn and turn['request_id'] != descriptor['request_id']:
            raise ValueError('Model request changed within an environment turn')
        turn['request_id'] = descriptor['request_id']
        # The worker overwrites its latest export. Keep exactly the bytes we
        # actually selected, with an immutable receipt, before projecting them.
        archive = Path(turn['directory']) / f"used-hidden-{descriptor['sequence']:06d}.npy"
        if archive.exists():
            with archive.open('rb') as stream: saved = np.load(stream, allow_pickle=False)
            if not np.array_equal(saved, hidden): raise ValueError('Selected hidden sequence changed')
        else:
            with archive.open('xb') as stream: np.save(stream, hidden, allow_pickle=False)
        selected = {**descriptor, 'worker_latest_export': {'path': descriptor['path'], 'sha256': descriptor['sha256']},
            'path': str(archive), 'sha256': sha(archive.read_bytes()), 'hidden_shape': [4096],
            'retention': 'Immutable actual selected hidden; original worker latest export may change'}
        write_json(archive.with_suffix('.json'), selected)
        current = self.couplers[env_id].currents(hidden, event_id=turn['event_id'],
            source={'backend': 'vllm_omni', 'request_id': descriptor['request_id'],
                    'generation_sequence': descriptor['sequence'], 'selected_hidden': selected})
        return current, selected

    @staticmethod
    def _check_hidden_request(turn, actual):
        expected = turn.get('client_request_id')
        if (not isinstance(expected, str) or not re.fullmatch(r'[a-f0-9]{32}', expected)
                or not isinstance(actual, str)
                or not re.fullmatch(r'chatcmpl-' + expected + r'-[a-f0-9]{8}', actual)):
            raise ValueError('Actual hidden request does not belong to the submitted model request')

    @classmethod
    def _check_response_request(cls, turn, value):
        expected = turn.get('client_request_id')
        if (not isinstance(expected, str) or not re.fullmatch(r'[a-f0-9]{32}', expected)
                or value.get('request_id') != expected
                or value.get('response_id') != 'chatcmpl-' + expected
                or value.get('model') != turn.get('expected_model')):
            raise ValueError('Model response differs from the submitted request/model')
        if turn.get('request_id') is not None:
            cls._check_hidden_request(turn, turn['request_id'])

    def tick(self):
        """Advance each owned body once, with latest valid model input if ready."""
        if self.closed: raise ValueError('Flow has been closed')
        from .latent_contract import write_json
        now = time.monotonic()
        for key, old in list(self.retired.items()):
            if old['future'].done():
                self._complete_turn(key, old)
                self.retired.pop(key)
        need_observation = [key for key in self.pool.workers if key not in self.turns
            and key not in self.retired and now >= self.next_turn[key]]
        captures = self.pool.capture_inputs(need_observation, include_neural_state=True) if need_observation else {}
        for env_id, observation in captures.items():
            try: self._prepare(env_id, observation)
            except Exception as exc:
                self.errors[env_id] = str(exc)
                self.next_turn[env_id] = now + 5
                if env_id not in self.turns:
                    try: self.retention.discard_unsubmitted_observation(env_id, observation)
                    except Exception as cleanup_error:
                        self.errors[env_id] += '; unsubmitted input retained: ' + str(cleanup_error)
        commands, updates = {}, {}
        for env_id in self.pool.workers:
            command = {'input_id': captures[env_id]['input_id']} if env_id in captures else {}
            turn = self.turns.get(env_id)
            if turn and turn['episode'] != self.pool.status()['workers'][env_id]['episode']:
                turn['release_previous_body'] = True
                self._cancel_turn(turn)
                if not turn['future'].done(): self.retired[env_id] = turn
                else: self._complete_turn(env_id, turn)
                self.turns.pop(env_id)
                self.previous_body.pop(env_id, None)
                turn = None
            if turn:
                try:
                    latest = self._latest(env_id)
                    if latest:
                        current, descriptor = latest
                        command.update(model_currents_file=current['currents_file'], capture_neural_state=True)
                        updates[env_id] = (current, descriptor)
                except Exception as exc:
                    self.errors[env_id] = str(exc)
            commands[env_id] = command
        samples = self.transition_observer.before_steps(captures, commands) if self.transition_observer else None
        results = self.pool.step_many(commands)
        if self.transition_observer:
            self.transition_observer.after_steps(samples, results)
        for env_id, result in results.items():
            turn = self.turns.get(env_id)
            if turn and not turn['future'].done(): self.steps_during_model[env_id] += 1
            if env_id in updates:
                current, descriptor = updates[env_id]
                if not result.get('parallel_inputs', {}).get('actual_sum_verified'):
                    raise RuntimeError('Actual neural current application was not verified')
                turn['last_applied_sequence'] = descriptor['sequence']
                self.applied[env_id] += 1
                evidence = {'env_id': env_id, 'event_id': turn['event_id'], 'hidden': descriptor,
                    'applied_current': current, 'actual_transition': result,
                    'model_observation': turn['observation'], 'asynchronous': True,
                    'same_observation_and_application_frame_claimed': False}
                receipt = Path(turn['directory']) / f"application-{self.applied[env_id]:06d}.json"
                write_json(receipt, evidence)
                turn['verified_applications'].append(str(receipt))
                self.retention.complete_application(env_id, turn['event_id'], str(self.applied[env_id]),
                    current_descriptors=[current['currents_file'], result['parallel_inputs']['applied_currents_file']],
                    feedback=result['feedback'], actual_sum_verified=True)
                self.events.append({'env_id': env_id, 'event_id': turn['event_id'], 'sequence': descriptor['sequence'],
                    'phase': descriptor['phase'], 'receipt': str(receipt),
                    'brain_simulation_ms': result['status']['simulation_ms']})
                self.events = self.events[-128:]
            if turn and turn['future'].done():
                # The bounded exporter can finish its last atomic write just
                # after HTTP returns. Preserve the turn for one short grace.
                if 'completed_seen_at' not in turn:
                    turn['completed_seen_at'] = time.monotonic()
                if time.monotonic() - turn['completed_seen_at'] >= .5:
                    try:
                        response = turn['future'].result()
                        self._check_response_request(turn, response)
                        if not turn['verified_applications']:
                            raise RuntimeError('Model reply completed without any verified numerical feedback to this brain')
                        self.responses[env_id] = {**response, 'native_latent_verified': True,
                            'event_id': turn['event_id'], 'application_receipts': list(turn['verified_applications'])}
                        write_json(Path(turn['directory']) / 'coupled-response.json', self.responses[env_id])
                    except Exception as exc: self.errors[env_id] = str(exc)
                    self._complete_turn(env_id, turn)
                    self.turns.pop(env_id)
                    self.next_turn[env_id] = time.monotonic() + self.interval
            if result['terminated'] or result['truncated']:
                old = self.turns.pop(env_id, None)
                if old:
                    old['release_previous_body'] = True
                    self._cancel_turn(old)
                    if not old['future'].done(): self.retired[env_id] = old
                    else: self._complete_turn(env_id, old)
                self.pool.reset(env_id, seed=self.pool.status()['workers'][env_id]['seed'] + 1)
                self.previous_body.pop(env_id, None)
                if old is None:
                    self.retention.release('previous_body:' + env_id)
                self.next_turn[env_id] = time.monotonic()
        self.retention.cleanup(blocking=False)
        return results

    def stream_progress(self, env_id):
        turn = self.turns.get(env_id)
        if turn:
            value = turn['progress'].snapshot()
            if value is not None:
                error = self.errors.get(env_id)
                try:
                    self._check_response_request(turn, value)
                    future = turn['future']
                    if future.done():
                        if future.cancelled():
                            raise RuntimeError('Model stream was cancelled')
                        failure = future.exception()
                        if failure is not None:
                            raise RuntimeError(str(failure))
                except (ValueError, RuntimeError) as exc:
                    error = str(exc)
                    self.errors[env_id] = error
                return {**value, 'event_id': turn['event_id'], 'phase': 'streaming',
                        'numerical_feedback_verified': bool(turn['verified_applications']) and not error,
                        'error': error}
            return None
        response = self.responses.get(env_id)
        if response and response.get('streaming'):
            value = response['streaming']
            return {**value, 'sequence': value['events'], 'event_id': response['event_id'],
                    'request_id': response['request_id'], 'text': response['text'],
                    'phase': 'complete', 'complete': True,
                    'numerical_feedback_verified': response['native_latent_verified']}
        return None

    def status(self):
        return {'model_backend': 'vllm_omni', 'flow_closed': self.closed,
            'inference_cache': self.inference_cache.status() if self.inference_cache else None,
            'pending_model_environments': sorted(set(self.turns) | set(self.retired)), 'actual_model_applications': dict(self.applied),
            'body_steps_while_model_pending': dict(self.steps_during_model),
            'responses': deepcopy(self.responses), 'errors': dict(self.errors), 'recent_applications': list(self.events),
            'raw_retention': self.retention.status(),
            'brain_to_model_contract': 'Actual brain/body soft embeddings injected into attention',
            'model_to_brain_contract': 'Latest real generated hidden projected into one bounded additive neuron current',
            'feedback_timing': 'New neural/body observations condition subsequent turns, not already-computed KV history'}

    def close(self):
        self.closed = True
        turns = (*self.turns.values(), *self.retired.values())
        futures = [turn['future'] for turn in turns]
        for turn in turns: self._cancel_turn(turn)
        self.retention.close_after_futures(futures)
        self.turns.clear()
        self.executor.shutdown(wait=False, cancel_futures=False)
