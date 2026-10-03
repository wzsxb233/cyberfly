"""Delayed native numerical control without putting HTTP on the body actor.

This is deliberately a separate protocol from SharedIOBus: the model observes
an earlier image, while the real brain keeps receiving the current image.
Only the actor captures or advances the live brain. Worker threads see files.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import io
from pathlib import Path
import threading
import time
import uuid
import wave

import numpy as np
from PIL import Image

from .bus import SharedIOBus,SharedLinkClient,ROOT


def rgb_copy(value):
    pixels=np.asarray(value)
    if pixels.dtype!=np.uint8 or pixels.ndim!=3 or pixels.shape[2]!=3:
        raise ValueError('An actual uint8 RGB image is required')
    return np.ascontiguousarray(pixels).copy()


class SnapshotBrain:
    """No live backend reference: even accidental worker reads are immutable."""
    def __init__(self,catalog,description,snapshot):
        self.catalog,self.description,self.snapshot=catalog,description,copy.deepcopy(snapshot)
    def groups(self):return self.catalog
    def describe(self):return self.description
    def read_state(self,*args,**kwargs):return copy.deepcopy(self.snapshot)


class AsyncSharedLoop:
    def __init__(self,brain,client=None,*,io_scope='sensory_motor',max_delay_s=30,
                 max_new_tokens=16,speak=False,measure_brain_ablation=False):
        if not 0<float(max_delay_s)<=300:raise ValueError('max_delay_s must be 0..300')
        if type(max_new_tokens) is not int or not 0<=max_new_tokens<=256:raise ValueError('Invalid generation budget')
        if speak and max_new_tokens==0:raise ValueError('A zero-token event cannot request speech')
        self.brain=brain;self.client=client or SharedLinkClient();self.io_scope=io_scope
        self.max_delay_s=float(max_delay_s);self.max_new_tokens=max_new_tokens
        self.speak=bool(speak);self.measure_brain_ablation=bool(measure_brain_ablation)
        self.prototype=SharedIOBus(brain,self.client,connection_mode='neuron_direct',io_scope=io_scope,
            transport='simplex',fusion_mode='latent_only')
        self.catalog=brain.groups();self.description=brain.describe()
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='cyberfly-delayed-model')
        self.owner=threading.get_ident();self.future=None;self.bus=None;self.phase='idle'
        self.last=None;self.observation=None;self.closed=False;self.sequence=0
        self.cancelled=threading.Event()

    def _actor(self):
        if threading.get_ident()!=self.owner:raise RuntimeError('Only the body actor may use the asynchronous loop')
        if self.closed:raise RuntimeError('Asynchronous loop is closed')

    def status(self):
        return {'phase':self.phase,'busy':self.phase!='idle','closed':self.closed,
            'observation_id':(self.observation or {}).get('observation_id'),
            'event_id':(self.bus.last or {}).get('event_id') if self.bus else None,
            'elapsed_s':time.monotonic()-self.observation['monotonic'] if self.observation else 0.}

    def observe(self,text,rgb,*,body_packet,audio_path=None,legacy_general=None,
                legacy_neural_drive=None,visual_input_enabled=True):
        """Actor: snapshot once and enqueue; returns without calling the model."""
        self._actor();self.poll()
        if self.phase!='idle':return {'accepted':False,**self.status()}
        pixels=rgb_copy(rgb)
        if body_packet is None or body_packet['vision']['rgb_sha256']!=hashlib.sha256(pixels.tobytes()).hexdigest():
            raise ValueError('Model observation image/body must be sampled together')
        self.sequence+=1
        observation_id=uuid.uuid4().hex
        directory=ROOT/'artifacts/shared_async_observations'/observation_id
        snapshot=self.prototype._raw_state(directory/'brain')
        observation={'observation_id':observation_id,'monotonic':time.monotonic(),'wall_time':time.time(),
            'brain_state':snapshot,'body_simulation_time_s':body_packet['simulation_time_s'],
            'image_rgb_sha256':hashlib.sha256(pixels.tobytes()).hexdigest()}
        frozen=SnapshotBrain(self.catalog,self.description,snapshot)
        bus=copy.copy(self.prototype);bus.brain=frozen;bus.pending=None;bus.last=None;bus.sequence=self.sequence-1
        # A single in-flight event owns these CPU source adapters. Actor-side
        # re-budgeting happens only after the encode worker has finished.
        arguments={'audio_path':str(audio_path) if audio_path else None,'body_packet':copy.deepcopy(body_packet),
            'legacy_general':copy.deepcopy(legacy_general or []),'legacy_neural_drive':copy.deepcopy(legacy_neural_drive or {}),
            'visual_input_enabled':bool(visual_input_enabled)}
        self.observation=observation;self.bus=bus;self.phase='encoding'
        self.future=self.executor.submit(self._encode,bus,text,pixels,arguments,observation,self.cancelled)
        return {'accepted':True,**self.status()}

    @staticmethod
    def _encode(bus,text,pixels,arguments,observation,cancelled):
        bus.prepare(text,pixels,**arguments)
        event,output,_=bus.pending
        event.update(schema=3,transport='async_simplex',phase='awaiting_delayed_application',
            asynchronous=True,shared_input_verified=False,
            input_ownership='Model observation is earlier; original brain senses remain current at application',
            model_observation={k:v for k,v in observation.items() if k!='monotonic'})
        if cancelled.is_set():
            bus.abort('Asynchronous actor closed before model current application')
            return bus
        bus._save(output,event);bus.last=event.copy()
        return bus

    def poll(self):
        """Actor: consume completed futures only; never wait on HTTP."""
        self._actor()
        if self.future is None or not self.future.done():return self.status()
        future,self.future=self.future,None
        try:
            result=future.result()
            if self.phase=='encoding':
                self.bus=result
                if time.monotonic()-self.observation['monotonic']>self.max_delay_s:
                    self.bus.abort('Model current expired before application; nothing was applied')
                    self.last=self.bus.last;self.phase='idle'
                else:self.phase='ready'
            elif self.phase=='decoding':self.last=result;self.phase='idle'
            else:raise RuntimeError('Unexpected asynchronous completion state')
        except Exception as exc:
            self.bus.abort(exc)
            self.last=self.bus.last or {'phase':'failed','error':str(exc),'asynchronous':True}
            self.phase='idle'
        return self.status()

    def take_stimulation(self,rgb,*,body_packet,legacy_general=None,
                         legacy_neural_drive=None,visual_input_enabled=True):
        """Actor: re-budget against current inputs; consume the result once."""
        self._actor();self.poll()
        if self.phase!='ready':return None
        if time.monotonic()-self.observation['monotonic']>self.max_delay_s:
            self.bus.abort('Model current expired before application; nothing was applied')
            self.last=self.bus.last;self.phase='idle';return None
        bus=self.bus;event,output,_=bus.pending
        pixels=rgb_copy(rgb);actual_hash=hashlib.sha256(pixels.tobytes()).hexdigest()
        if body_packet['vision']['rgb_sha256']!=actual_hash:raise ValueError('Actual application image/body mismatch')
        bus.brain=self.brain  # This method is on the actor, after worker completion.
        before=bus._raw_state(output/'application-before')
        if before['simulation_ms']<event['neural_time_before_ms']:raise RuntimeError('Brain reset while model result was in flight')
        if body_packet['simulation_time_s']<event['body_before']['simulation_time_s']:raise RuntimeError('Body reset while model result was in flight')
        Image.fromarray(pixels).save(output/'application.png')
        event['model_observation_original_inputs']={key:event[key] for key in
            ('original_general_stimulation','original_neural_drive','original_visual_input_enabled')}
        event.update(original_general_stimulation=copy.deepcopy(legacy_general or []),
            original_neural_drive=copy.deepcopy(legacy_neural_drive or {}),original_visual_input_enabled=bool(visual_input_enabled))
        fresh_body=bus.sources.prepare_body(body_packet,output/'application-body')
        original_body=event.get('body_sensory_input')
        event['body_sensory_input']=fresh_body
        try:bus._parallel(event,before,event['model_requested_currents_file'],event['model_evidence'])
        finally:event['body_sensory_input']=original_body
        event['actual_body_sensory_input']=fresh_body
        event['sensory_sources']['body']['sample_stage']='actual_application'
        event.update(phase='delayed_current_consumed',brain_state_at_application=before,
            actual_application={'wall_time':time.time(),'model_delay_s':time.monotonic()-self.observation['monotonic'],
                'neural_time_before_ms':before['simulation_ms'],'body_before':bus._body_summary(body_packet),
                'image_path':str(output/'application.png'),'image_rgb_sha256':actual_hash,
                'image_matches_model_observation':actual_hash==event['image_rgb_sha256']},
            current_consumption_count=1)
        self.phase='applied';bus._save(output,event);bus.last=event.copy()
        return {'event_id':event['event_id'],'additional_stimulation':bus.stimulation(),
            'model_observation':event['model_observation'],'actual_application':event['actual_application']}

    def feedback(self,actual_brain_result,*,body_state,body_packet):
        """Actor: verify this one real applied step, snapshot, enqueue decode."""
        self._actor()
        if self.phase!='applied':raise RuntimeError('Feedback requires exactly one consumed model input')
        bus=self.bus;event,output,request=bus.pending
        applied=event['actual_application'];neural=actual_brain_result.get('neural',{})
        if actual_brain_result.get('input_rgb_sha256')!=applied['image_rgb_sha256']:
            raise RuntimeError('The real brain did not receive the freshly declared application pixels')
        if actual_brain_result.get('visual_input_enabled') is not event['original_visual_input_enabled']:
            raise RuntimeError('Asynchronous control altered original vision')
        ports=actual_brain_result.get('neural_drive',{}).get('normalized',{})
        if any(ports.get(k)!=v for k,v in event['parallel_inputs']['preserved_neural_drive'].items()):
            raise RuntimeError('Asynchronous control altered an original sensory port')
        if actual_brain_result.get('general_stimulation',{}).get('current_vector_sha256')!=event['expected_general_currents_sha256']:
            raise RuntimeError('The actual combined neuron current differs from the one-time application')
        after=bus._raw_state(output/'after')
        if (after['simulation_ms']<=applied['neural_time_before_ms'] or after['simulation_ms']!=neural.get('simulation_ms') or
                after['counts_sha256']!=neural.get('counts_sha256') or after['spikes_this_step']!=neural.get('spikes_this_step')):
            raise RuntimeError('Actual neural feedback belongs to a different interval')
        feedback=bus.sources.feedback_body(body_packet,output/'body-after')
        measured=bus._body_summary(body_packet);before=event['body_before']
        if measured['simulation_time_s']<=applied['body_before']['simulation_time_s']:
            raise RuntimeError('The body did not advance during the actual current application')
        displacement=np.asarray(measured['position_mm'])-np.asarray(before['position_mm'])
        measured.update(interval_s=measured['simulation_time_s']-before['simulation_time_s'],
            position_before_mm=before['position_mm'],contacts_before=before['contacts'],
            contact_count_change=measured['contacts']-before['contacts'],displacement_mm=displacement.tolist(),
            horizontal_displacement_mm=float(np.linalg.norm(displacement[:2])),
            interval_meaning='From the model observation to the final feedback; includes intervening body steps')
        event.update(phase='decoding_delayed_feedback',brain_state_after=after,body_feedback_file=feedback,
            body_feedback=measured,body_state={k:copy.deepcopy(body_state[k]) for k in
                ('step','simulation_time_s','position_mm','last_reward','is_success','feedback_visual','motor_readout','body_action') if k in body_state},actual_neural=neural,
            actual_stimulation=actual_brain_result.get('general_stimulation'),
            native_motor_output=actual_brain_result.get('native_action'),motor_readout=actual_brain_result.get('motor_readout'),
            original_inputs_preserved=True,applied_currents_verified=True,actual_input_verified=True,
            shared_input_verified=False,model_facts=None,brain_state_text_input_enabled=False,
            brain_features_shape=[166700,2],brain_features_sha256=after['sha256'])
        request.update(brain_state=after,body_feedback_file=feedback,generate_audio=self.speak,
            max_new_tokens=self.max_new_tokens,measure_brain_ablation=self.measure_brain_ablation)
        request.pop('facts',None)
        bus._save(output,event);bus.last=event.copy();self.phase='decoding'
        self.future=self.executor.submit(self._decode,bus,copy.deepcopy(request),self.speak)
        return {'accepted':True,**self.status()}

    @staticmethod
    def _decode(bus,request,speak):
        event,output,_=bus.pending;decoded=bus.client.call('decode',request)
        expected=event['model_evidence']
        for key in ('codec_sha256','coupler_sha256','coupler_generation','read_ids_sha256','write_ids_sha256','connection_mode','io_scope','model_revision','weights_manifest_sha256',
                    'language_adapter_sha256','language_adapter_generation','sensory_sha256'):
            if decoded.get(key)!=expected.get(key):raise RuntimeError('Model identity changed during delayed event: '+key)
        if decoded.get('event_id')!=event['event_id']:raise RuntimeError('Delayed decode event mismatch')
        say=decoded.get('say');skipped=decoded.get('generation_skipped') is True
        if not isinstance(say,str) or not say.strip() and not (request['max_new_tokens']==0 and skipped):
            raise RuntimeError('Delayed model response was empty without an explicit zero-token request')
        audio=decoded.get('audio_file')
        if speak or audio is not None:
            if not isinstance(audio,dict):raise RuntimeError('Requested original speech was not returned')
            path=Path(audio.get('path','')).resolve()
            if not path.is_relative_to(ROOT/'artifacts/native_speech') or not path.is_file() or path.stat().st_size>32_000_000:
                raise RuntimeError('Invalid native speech artifact')
            raw=path.read_bytes()
            if audio.get('event_id')!=event['event_id'] or audio.get('same_generation_hidden') is not True or hashlib.sha256(raw).hexdigest()!=audio.get('sha256'):
                raise RuntimeError('Original delayed speech provenance mismatch')
            with wave.open(io.BytesIO(raw)) as wav:
                if (wav.getnchannels(),wav.getsampwidth(),wav.getframerate())!=(1,2,24000) or wav.getnframes()<2400 or wav.getnframes()!=audio.get('samples'):
                    raise RuntimeError('Invalid actual native waveform')
            (output/'output.wav').write_bytes(raw)
            event.update(native_speech=audio,voice_source=audio['source'],playback_requested=speak)
            if speak:event.update(audio_output_path=str(output/'output.wav'),audio_output_sha256=audio['sha256'])
        event.update(phase='complete',say=say,generation_skipped=skipped,
            model_decode_evidence={k:v for k,v in decoded.items() if k not in {'say','hidden','logits'}},
            output_event_id=event['event_id'],elapsed_s=time.time()-event['created_at'])
        bus.pending=None;bus.last=event;bus._save(output,event);return event

    def abort(self,error):
        self._actor()
        if self.future is not None and not self.future.done():
            # A pending HTTP operation can finish, but its event will never be
            # applied after a mode/episode change. Closing does not wait for it.
            self.close();return
        if self.bus:self.bus.abort(error);self.last=self.bus.last
        self.future=None;self.phase='idle'

    def close(self):
        if threading.get_ident()!=self.owner:raise RuntimeError('Only the body actor may close the asynchronous loop')
        if self.closed:return
        self.closed=True;self.phase='closed'
        self.cancelled.set()
        if self.future:self.future.cancel()
        self.executor.shutdown(wait=False,cancel_futures=True)
