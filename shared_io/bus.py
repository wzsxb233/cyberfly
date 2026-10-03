"""One shared sensory event, actual numerical model/brain coupling, one output.

The brain and model do not acquire independent screenshots or prompts. Numeric
currents and returned brain features are archived with the event that caused
them, so a text-only supervisory call cannot masquerade as the internal link.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import time
import uuid
import urllib.request
import urllib.error
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(np.asarray(value, dtype='<f4').tobytes()).hexdigest()


class SharedLinkClient:
    def __init__(self, endpoint='http://127.0.0.1:18647', timeout=300):
        self.endpoint, self.timeout = endpoint.rstrip('/'), timeout

    def call(self, operation, payload):
        request = urllib.request.Request(self.endpoint + '/' + operation,
            data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(),
            headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.load(response)
        except urllib.error.HTTPError as exc:
            body = exc.read(4000).decode('utf-8', errors='replace')
            if exc.code == 409 and operation == 'language_adapter/train':
                try:
                    detail = json.loads(body).get('detail', {})
                except (ValueError, AttributeError):
                    detail = {}
                if (isinstance(detail, dict) and detail.get('code') == 'gpu_capacity_busy'
                        and detail.get('request_consumed') is False and detail.get('retryable') is True):
                    from vllm_integration.resource_lease import GPUCapacityBusy
                    raise GPUCapacityBusy(detail) from None
            raise RuntimeError('数值连接服务返回错误：' + body) from None
        except (OSError, TimeoutError) as exc:
            raise RuntimeError('MiniCPM 内部数值连接服务尚未就绪：' + str(exc)) from None
        if data.get('error'):
            raise RuntimeError(data['error'])
        return data


class SharedIOBus:
    def __init__(self, brain, client=None, *, connection_mode='grouped', io_scope='sensory_motor', transport='simplex', fusion_mode='latent_only'):
        from .brain_codec import BrainFeatureCodec
        if connection_mode not in {'grouped', 'neuron_direct'} or io_scope not in {'sensory_motor', 'all_neurons'}:
            raise ValueError('Unknown brain connection mode or neural I/O scope')
        self.brain, self.codec = brain, BrainFeatureCodec(brain.groups())
        self.connection_mode, self.io_scope = connection_mode, io_scope
        if connection_mode == 'neuron_direct':
            from .neuron_link import NeuronStateCodec
            self.neuron_codec = NeuronStateCodec(io_scope)
        if transport not in {'simplex','duplex'} or (transport=='duplex' and connection_mode!='neuron_direct'):
            raise ValueError('Native streaming requires the per-neuron connection')
        self.transport=transport
        if fusion_mode not in {'latent_only','latent_with_measurements'}:raise ValueError('Unknown latent fusion mode')
        self.fusion_mode=fusion_mode
        if client is None and transport=='duplex':
            from .duplex_client import DuplexLinkClient
            client=DuplexLinkClient()
        self.client = client or SharedLinkClient()
        from .sensory_sources import SharedSensorySources
        self.sources = SharedSensorySources()
        self.sequence = 0
        self.last = None
        self.pending = None

    def prepare(self, text, rgb, *, audio_path=None, brain_result=None,
                legacy_general=None, legacy_neural_drive=None, visual_input_enabled=True, body_packet=None):
        if self.pending is not None:
            raise RuntimeError('The previous shared event has not finished')
        from .manage import start
        start(wait=True)
        self.sequence += 1
        event_id = uuid.uuid4().hex
        output = ROOT / 'artifacts/shared_io' / event_id
        output.mkdir(parents=True)
        pixels = np.asarray(rgb, dtype=np.uint8).copy()
        Image.fromarray(pixels).save(output / 'input.png')
        event = {'schema': 2, 'event_id': event_id, 'sequence': self.sequence,
                 'connection_mode': self.connection_mode, 'io_scope': self.io_scope, 'transport':self.transport, 'fusion_mode':self.fusion_mode,
                 'original_visual_input_enabled': bool(visual_input_enabled),
                 'original_general_stimulation': legacy_general or [],
                 'original_neural_drive': legacy_neural_drive or {},
                 'created_at': time.time(), 'text': text, 'image_path': str(output / 'input.png'),
                 'image_rgb_sha256': hashlib.sha256(pixels.tobytes()).hexdigest(),
                 'image_shape': list(pixels.shape), 'modalities': (['text'] if text else []) + ['image'],
                 'input_ownership': 'same event supplied to MiniCPM and the real connectome'}
        if audio_path:
            source = Path(audio_path)
            raw = source.read_bytes()
            (output / 'input.wav').write_bytes(raw)
            event.update(audio_path=str(output / 'input.wav'), audio_sha256=hashlib.sha256(raw).hexdigest())
            event['modalities'].append('audio')
        if body_packet is not None:
            if body_packet.get('vision', {}).get('rgb_sha256') != event['image_rgb_sha256']:
                raise RuntimeError('Body senses and shared image were not sampled together')
            body = self.sources.prepare_body(body_packet, output / 'body-before')
            event.update(body_sensory_input=body, body_features_file=body['body_features_file'],
                         body_before=self._body_summary(body_packet))
            event['modalities'].append('body_numeric')
        if self.connection_mode == 'neuron_direct':
            return self._prepare_neuron(event, output)
        if brain_result and (brain_result.get('neural') or {}).get('simulation_ms', 0) > 0:
            features = self.codec.encode_step(brain_result)
        else:
            features = self.codec.encode_group_state(self.brain.group_state(), duration_ms=0.0,
                learning=bool(self.brain.describe().get('learning', {}).get('enabled', False)))
            event['initial_state_note'] = 'Validated genuine zero-time, zero-spike initial state; no inferred interval.'
        features = np.asarray(features, dtype=np.float32)
        event['neural_time_before_ms'] = float(self.brain.group_state()['simulation_ms'])
        request = {key: event[key] for key in ('event_id', 'text', 'image_path', 'audio_path', 'body_features_file') if key in event}
        request['brain_features'] = features.tolist()
        self.pending = (event, output, request)
        self._save(output, event)
        encoded = self.client.call('encode', request)
        if encoded.get('event_id') != event_id or encoded.get('codec_sha256') != self.codec.codec_sha256:
            raise RuntimeError('Model response does not match the shared event and verified brain feature codec')
        hidden = np.asarray(encoded['hidden'], dtype=np.float32)
        currents = np.asarray(encoded['currents_mv'], dtype=np.float32)
        if hidden.ndim != 1 or hidden.size != 4096 or currents.shape != (47,) or not np.isfinite(hidden).all() or not np.isfinite(currents).all() or np.max(np.abs(currents)) > 30.0001:
            raise RuntimeError('Internal model output does not match the finite 4096→47 numeric connection')
        np.save(output / 'model_hidden.npy', hidden, allow_pickle=False)
        np.savez_compressed(output / 'model_to_brain.npz', hidden=hidden, currents_mv=currents, brain_features_before=features)
        event.update(phase='model_encoded', model_hidden_shape=list(hidden.shape),
                     model_hidden_sha256=digest(hidden), currents_mv=currents.tolist(),
                     current_vector_sha256=digest(currents), model_evidence={key:value for key,value in encoded.items() if key not in {'hidden','currents_mv'}})
        before = self._raw_state(output / 'before')
        event['brain_state_before'] = before
        with np.load(before['path'], allow_pickle=False) as arrays:
            expanded = currents[arrays['macro_group_index']]
        from connectome_adapter.neuron_currents import write_neuron_currents
        descriptor = write_neuron_currents(np.ascontiguousarray(expanded, dtype=np.float32), self._neuron_ids)
        self._parallel(event, before, descriptor, encoded)
        self.pending = (event, output, request)
        self.last = event.copy()
        self._save(output, event)
        return event['currents_file']

    def _parallel(self, event, before, descriptor, encoded):
        from .parallel_inputs import prepare_parallel_stimulation
        modal = self.sources.project_modalities(encoded, event['event_id'])
        candidates = []
        body = event.get('body_sensory_input')
        if body:
            candidates.append(('body', body['currents_file']))
        for name, result in modal.items():
            if result['active']:
                candidates.append((name, result['currents_file']))
        candidates.append(('language', descriptor))
        general = list(event['original_general_stimulation'])
        additions, sources = [], {}
        for name, currents in candidates:
            parallel = prepare_parallel_stimulation(self.brain, before, currents,
                general, event['original_neural_drive'])
            applied = parallel['applied_currents_file']
            item = {'neuron_currents_file': applied}
            additions.append(item)
            general.append(item)
            sources[name] = {'requested_currents_file': currents, 'applied_currents_file': applied,
                            'budget': parallel['requested_vs_applied']}
        event.update(model_requested_currents_file=descriptor, currents_file=parallel['applied_currents_file'],
                     current_vector_sha256=parallel['applied_currents_file']['currents_sha256'],
                     parallel_inputs=parallel, sensory_sources=sources, modality_inputs=modal,
                     additional_stimulation=additions,
                     expected_general_currents_sha256=parallel['expected_general_currents_sha256'])

    def stimulation(self):
        if self.pending is None:
            raise RuntimeError('No prepared shared event')
        return list(self.pending[0]['additional_stimulation'])

    @staticmethod
    def _body_summary(packet):
        return {'simulation_time_s': packet['simulation_time_s'], 'decision_step': packet['decision_step'],
                'position_mm': packet['thorax']['position_world_mm'],
                'velocity_world_rot_rad_s_then_lin_mm_s': packet['thorax']['velocity_world_rot_rad_s_then_lin_mm_s'],
                'contacts': packet['contact_count'], 'packet_sha256': packet['packet_sha256']}

    def _raw_state(self, output):
        descriptor = self.brain.read_state(output)
        raw = Path(descriptor['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != descriptor['sha256']:
            raise RuntimeError('Raw brain snapshot changed before it reached the model')
        with np.load(io.BytesIO(raw), allow_pickle=False) as data:
            ids, volts, counts = data['ids'], data['v'], data['counts']
            if any(a.shape != (166700,) for a in (ids, volts, counts)) or not np.isfinite(volts).all() or np.any(counts < 0):
                raise RuntimeError('Expected the real voltage and spike count of every retained neuron')
            ids_sha = hashlib.sha256(ids.tobytes()).hexdigest()
            if ids_sha != self.brain.groups()['ids_sha256']:
                raise RuntimeError('Raw brain neuron order differs from the actual connectome')
            self._neuron_ids = ids.copy()
            descriptor['ids_sha256'] = ids_sha
            descriptor['counts_sha256'] = hashlib.sha256(counts.tobytes()).hexdigest()
            descriptor['spikes_this_step'] = int(counts.sum())
        return descriptor

    def _prepare_neuron(self, event, output):
        before = self._raw_state(output / 'before')
        self.neuron_codec.read(before)
        event.update(brain_state_before=before, neural_time_before_ms=before['simulation_ms'])
        request = {key: event[key] for key in ('event_id', 'text', 'image_path', 'audio_path', 'body_features_file') if key in event}
        request.update(connection_mode='neuron_direct', io_scope=self.io_scope, brain_state=before)
        self.pending = (event, output, request)
        self._save(output, event)
        encoded = self.client.call('encode', request)
        if (encoded.get('event_id') != event['event_id'] or encoded.get('connection_mode') != 'neuron_direct' or
                encoded.get('io_scope') != self.io_scope or encoded.get('codec_sha256') != self.neuron_codec.codec_sha):
            raise RuntimeError('The model did not perform this event’s requested per-neuron connection')
        hidden = np.asarray(encoded['hidden'], dtype=np.float32)
        descriptor = encoded['currents_file']
        if hidden.shape != (4096,) or not np.isfinite(hidden).all():
            raise RuntimeError('Expected the actual finite 4096-dimensional model hidden state')
        if descriptor.get('neurons') != 166700 or descriptor.get('ids_sha256') != before['ids_sha256']:
            raise RuntimeError('Model currents do not address this full brain’s neuron IDs')
        from connectome_adapter.neuron_currents import load_neuron_currents
        currents = load_neuron_currents(descriptor, self._neuron_ids, before['ids_sha256'])
        read_mask, write_mask = self.neuron_codec.masks()
        if (np.any(currents[~write_mask] != 0) or
                encoded.get('write_ids_sha256') != hashlib.sha256(self._neuron_ids[write_mask].tobytes()).hexdigest() or
                encoded.get('read_ids_sha256') != hashlib.sha256(self._neuron_ids[read_mask].tobytes()).hexdigest()):
            raise RuntimeError('Model I/O does not match the selected real sensory and motor neuron IDs')
        np.save(output / 'model_hidden.npy', hidden, allow_pickle=False)
        event.update(phase='model_encoded', model_hidden_shape=list(hidden.shape), model_hidden_sha256=digest(hidden),
                     currents_file=descriptor, current_vector_sha256=descriptor['currents_sha256'],
                     model_evidence={key:value for key,value in encoded.items() if key not in {'hidden','currents_mv'}})
        self._parallel(event, before, descriptor, encoded)
        self.last = event.copy()
        self._save(output, event)
        return event['currents_file']

    def finish(self, actual_brain_result, *, body_state, speak=False, body_packet=None):
        if self.pending is None:
            raise RuntimeError('No shared sensory event is pending')
        event, output, request = self.pending
        if actual_brain_result.get('input_rgb_sha256') != event['image_rgb_sha256']:
            raise RuntimeError('The model and brain did not receive the same image pixels')
        neural_time = float(actual_brain_result.get('neural', {}).get('simulation_ms', -1))
        if neural_time <= event['neural_time_before_ms']:
            raise RuntimeError('The brain did not advance after this shared sensory event')
        if actual_brain_result.get('visual_input_enabled') is not event['original_visual_input_enabled']:
            raise RuntimeError('The shared connection changed the original visual input setting')
        actual_ports = actual_brain_result.get('neural_drive', {}).get('normalized', {})
        expected_ports = event['parallel_inputs']['preserved_neural_drive']
        if any(actual_ports.get(key) != value for key, value in expected_ports.items()):
            raise RuntimeError('The shared connection changed an existing sensory input')
        if actual_brain_result.get('general_stimulation', {}).get('current_vector_sha256') != event['expected_general_currents_sha256']:
            raise RuntimeError('Actual brain input does not equal original stimulation plus the new model input')
        event.update(original_inputs_preserved=True, visual_input_enabled=actual_brain_result['visual_input_enabled'])
        if self.connection_mode == 'neuron_direct':
            after = self._raw_state(output / 'after')
            if (after['simulation_ms'] != neural_time or
                    after['counts_sha256'] != actual_brain_result['neural']['counts_sha256'] or
                    after['spikes_this_step'] != actual_brain_result['neural']['spikes_this_step']):
                raise RuntimeError('Raw neural output does not belong to the brain interval that just ran')
            event.update(brain_state_after=after, brain_features_shape=[166700, 2],
                         brain_features_sha256=after['sha256'],
                         brain_features_meaning='Per-neuron actual membrane voltage and interval spike counts, fixed original ID order')
            request['brain_state'] = after
        else:
            features = np.asarray(self.codec.encode_step(actual_brain_result), dtype=np.float32)
            np.save(output / 'brain_to_model.npy', features, allow_pickle=False)
            event.update(brain_features_shape=list(features.shape), brain_features_sha256=digest(features))
            request['brain_features'] = features.tolist()
        event.update(phase='brain_integrated',
                     actual_neural=actual_brain_result.get('neural'),
                     actual_stimulation=actual_brain_result.get('general_stimulation'),
                     native_motor_output=actual_brain_result.get('native_action'),
                     motor_readout=actual_brain_result.get('motor_readout'),
                     shared_input_verified=True, applied_currents_verified=True,
                     body_state={key:body_state[key] for key in ('step','simulation_time_s','position_mm','last_reward','is_success','feedback_visual','motor_readout','body_action') if key in body_state})
        if event.get('body_features_file') is not None:
            if body_packet is None:
                raise RuntimeError('The physical body did not return its actual sensory state')
            feedback = self.sources.feedback_body(body_packet, output / 'body-after')
            event['body_feedback_file'] = feedback
            request['body_feedback_file'] = feedback
            measured = self._body_summary(body_packet)
            before_body = event['body_before']
            elapsed = measured['simulation_time_s'] - before_body['simulation_time_s']
            if elapsed <= 0:
                raise RuntimeError('The physical body did not advance during the shared event')
            displacement = np.asarray(measured['position_mm']) - np.asarray(before_body['position_mm'])
            measured.update(interval_s=elapsed, position_before_mm=before_body['position_mm'],
                            contacts_before=before_body['contacts'], contact_count_change=measured['contacts']-before_body['contacts'],
                            displacement_mm=displacement.tolist(),
                            horizontal_displacement_mm=float(np.linalg.norm(displacement[:2])))
            event['body_feedback'] = measured
        self._save(output, event)
        request['generate_audio'] = bool(speak)
        from .measurements import compact_measurements
        measured=compact_measurements(actual_brain_result,event.get('body_feedback'),event['body_state'])
        event['measured_summary']=measured
        if self.fusion_mode=='latent_with_measurements':request['facts']=measured
        event['model_facts']=request.get('facts')
        event['brain_state_text_input_enabled']=self.fusion_mode=='latent_with_measurements'
        self._save(output,event)
        decoded = self.client.call('decode', request)
        if self.connection_mode == 'neuron_direct' and (decoded.get('connection_mode') != self.connection_mode or decoded.get('io_scope') != self.io_scope):
            raise RuntimeError('Brain decoding used a different neural I/O scope')
        if (decoded.get('event_id') != event['event_id'] or decoded.get('codec_sha256') != event['model_evidence'].get('codec_sha256')
                or decoded.get('coupler_sha256') != event['model_evidence'].get('coupler_sha256')):
            raise RuntimeError('Brain-conditioned decoding changed the event, codec or coupling weights')
        say = decoded.get('say')
        listening=self.transport=='duplex' and decoded.get('is_listen') is True
        if not isinstance(say,str) or (not say.strip() and not listening and self.transport!='duplex'):
            raise RuntimeError('MiniCPM did not produce a response from the shared numeric state')
        event['is_listen']=listening
        event.update(phase='complete', say=say, model_decode_evidence={key:value for key,value in decoded.items() if key not in {'say','hidden','logits'}},
                     output_event_id=event['event_id'], elapsed_s=time.time()-event['created_at'])
        if (speak and self.transport=='simplex') or decoded.get('audio_file') is not None:
            import wave as wave_reader
            audio = decoded.get('audio_file') or {}
            path = Path(audio.get('path', '')).resolve()
            if (audio.get('event_id') != event['event_id'] or audio.get('same_generation_hidden') is not True
                    or not any(path.is_relative_to(ROOT / folder) for folder in ('artifacts/native_speech','artifacts/native_duplex')) or not path.is_file()
                    or path.stat().st_size > 32_000_000):
                raise RuntimeError('Missing verified speech from the same brain-conditioned generation')
            wave = path.read_bytes()
            if hashlib.sha256(wave).hexdigest() != audio.get('sha256'):
                raise RuntimeError('Native speech bytes changed before playback')
            with wave_reader.open(io.BytesIO(wave)) as wav:
                if (wav.getnchannels() != 1 or wav.getsampwidth()!=2 or wav.getframerate() != 24000
                        or wav.getnframes() < (1 if self.transport=='duplex' else 2400)
                        or wav.getnframes()!=audio.get('samples')):
                    raise RuntimeError('Invalid native speech waveform')
            (output / 'output.wav').write_bytes(wave)
            event.update(voice_source=audio['source'], native_speech=audio,playback_requested=bool(speak))
            if speak:
                event.update(audio_output_path=str(output / 'output.wav'), audio_output_sha256=audio['sha256'])
        self.pending = None
        self.last = event
        self._save(output, event)
        return event

    def close(self):
        if callable(getattr(self.client,'close',None)):
            self.client.close()

    def abort(self, error):
        if self.pending:
            event, output, _ = self.pending
            event.update(phase='failed', error=str(error))
            self.last = event
            self._save(output, event)
            self.pending = None
        if self.transport=='duplex':
            try:self.close()
            except Exception as exc:
                if self.last:
                    self.last['stream_close_error']=str(exc)

    @staticmethod
    def _save(output, event):
        path = output / 'event.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(event, indent=2, ensure_ascii=False, allow_nan=False))
        temporary.replace(path)
