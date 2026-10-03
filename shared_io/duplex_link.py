"""Exclusive original MiniCPM duplex session with real numeric brain/body IO.

The network can receive/queue audio while chunks play. GPU prefill/generation is
serial. This wraps the original persistent APM/LLM/TTS/flow caches; it does not
loop over the turn-based snapshot API. Live validation is recorded separately.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time
import uuid
import numpy as np
from .native_speech import NativeSpeech,write_tensors

ROOT=Path(__file__).resolve().parents[1]
def sha(raw):return hashlib.sha256(raw).hexdigest()


class DuplexProtocol:
    """Small ownership state machine, independently testable without a model."""
    def __init__(self,session_id):
        if not isinstance(session_id,str) or not 1<=len(session_id)<=128:raise ValueError('session_id must contain 1..128 characters')
        self.session_id=session_id;self.epoch=uuid.uuid4().hex;self.sequence=0;self.pending=None;self.closed=False

    def owner(self,req):
        if self.closed or req.get('session_id')!=self.session_id or req.get('epoch')!=self.epoch:
            raise ValueError('Duplex session/epoch is closed or does not match its owner')

    def before_prefill(self,req):
        self.owner(req)
        if self.pending is not None:raise ValueError('Finish or cancel the pending duplex unit before another prefill')
        if type(req.get('sequence')) is not int or req['sequence']!=self.sequence:raise ValueError('Duplex sequence must increase from zero without gaps')
        if not isinstance(req.get('event_id'),str) or not 1<=len(req['event_id'])<=128:raise ValueError('event_id is required')

    def before_generate(self,req):
        self.owner(req)
        if self.pending is None or type(req.get('sequence')) is not int or req.get('sequence')!=self.sequence or req.get('event_id')!=self.pending['event_id']:
            raise ValueError('Generate requires the exact pending duplex sequence and event')

    def completed(self):
        self.pending=None;self.sequence+=1

    def identity(self):return {'session_id':self.session_id,'epoch':self.epoch,'sequence':self.sequence,'streaming':True}


class DuplexLink:
    def __init__(self,host,req):
        self.host=host;self.protocol=DuplexProtocol(req.get('session_id'));self.closed=False;self.stream=None
        if req.get('connection_mode','neuron_direct')!='neuron_direct':raise ValueError('Initial native duplex supports neuron_direct only')
        self.native=host._native({**req,'connection_mode':'neuron_direct'})
        self.pinned=self.native.identity();self.history=[];self.turn_sources=[];self.body_identities={};self.started=time.time();self.last_brain_time=None
        self.baseline=req.get('diagnostic_baseline',False)
        if type(self.baseline) is not bool:raise ValueError('diagnostic_baseline must be boolean')
        self.anchor_mode=req.get('decision_anchor','native_boundary')
        if self.anchor_mode not in ('none','last_sensory','native_boundary'):raise ValueError('Invalid decision_anchor')
        self.decode_mode=req.get('decode_mode','greedy')
        if self.decode_mode not in ('greedy','sampling'):raise ValueError('Invalid decode_mode')
        self.numeric_placement=req.get('numeric_placement','after_sensory')
        if self.numeric_placement not in ('before_sensory','after_sensory'):raise ValueError('Invalid numeric_placement')
        self.old_cache=host.llm.config.use_cache
        self.old_checkpointing=bool(getattr(host.llm,'is_gradient_checkpointing',False))
        self.old_processor=getattr(host.model,'processor',None)
        self.directory=ROOT/'artifacts/native_duplex'/self.protocol.epoch;self.directory.mkdir(parents=True,exist_ok=True)
        try:
            if host.native_speech is None:host.native_speech=NativeSpeech(host)
            host.model.processor=host.processor
            host.llm.config.use_cache=True
            host.llm.gradient_checkpointing_disable();host.llm.eval()
            host.model.reset_session(reset_token2wav_cache=False)
            # Original as_duplex unconditionally calls init_tts. Reuse the exact
            # already-verified native codec instead of loading it a second time.
            original_init=host.model.init_tts
            host.model.init_tts=lambda *a,**k:host.native_speech.token2wav
            try:
                self.stream=host.model.as_duplex(device=str(next(host.llm.parameters()).device),
                    generate_audio=True,enable_float16=False,sliding_window_mode='basic',
                    basic_window_high_tokens=4096,basic_window_low_tokens=3072)
            finally:host.model.init_tts=original_init
            # Upstream StreamDecoder defaults listen_id to tokenizer EOS; bind
            # the explicitly named native listen token for probability controls.
            self.stream.decoder.listen_id=self.stream.listen_token_id
            prompt=req.get('system_prompt','Streaming Omni Conversation.' if self.baseline else 'Streaming Omni Conversation. 你与真实果蝇脑和身体数值相连，用简短自然中文互动。只把实际测量当事实，不声称读懂主观想法。')
            if not isinstance(prompt,str) or not 1<=len(prompt)<=2000:raise ValueError('system_prompt must contain 1..2000 characters')
            with host.torch.no_grad():self.stream.prepare(prefix_system_prompt=prompt,prompt_wav_path=str(host.native_speech.reference))
            self._write('open.json',self.status())
        except Exception:
            self._release();raise

    def _write(self,name,value):
        path=self.directory/name
        path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))

    def _check(self,req):
        self.protocol.owner(req)
        current=self.native.identity()
        keys=('coupler_sha256','coupler_generation','model_revision','weights_manifest_sha256','codec_sha256','read_ids_sha256','write_ids_sha256',
            'language_adapter_sha256','language_adapter_generation')
        if any(current.get(k)!=self.pinned.get(k) for k in keys):raise RuntimeError('Duplex model/codec/coupler identity changed')
        if req.get('connection_mode','neuron_direct')!='neuron_direct' or req.get('io_scope',self.pinned['io_scope'])!=self.pinned['io_scope']:
            raise ValueError('Do not change connection mode or IO scope within duplex session')

    def status(self):
        return {**self.pinned,**self.protocol.identity(),'status':'closed' if self.closed else 'native_duplex_ready',
            'pending_event_id':self.protocol.pending['event_id'] if self.protocol.pending else None,
            'llm_cache_tokens':int(self.stream.decoder.get_cache_length()) if self.stream else 0,
            'audio_chunk_samples':16000,'audio_sample_rate':16000,'first_chunk_padding':'official native first-window alignment is recorded per unit',
            'cancel_granularity':'GPU unit boundary; client playback can stop immediately; closed epoch outputs must be discarded',
            'model_reloaded':False,'native_codec_reused':True,'directory':str(self.directory),
            'diagnostic_baseline':self.baseline,'numeric_feedback_injected':not self.baseline,'brain_involved':not self.baseline,
            'llm_trainable_parameters':sum(p.numel() for p in self.host.llm.parameters() if p.requires_grad),
            'original_model_trainable_parameters':sum(p.numel() for p in self.host.model.parameters() if p.requires_grad),
            'decision_anchor':self.anchor_mode,'decode_mode':self.decode_mode,'numeric_placement':self.numeric_placement}

    def _top(self,logits):
        if logits is None:return []
        values,ids=logits[0].float().topk(10)
        return [{'id':int(i),'token':self.stream.tokenizer.decode([int(i)]),'logit':float(v)} for i,v in zip(ids.tolist(),values.tolist())]

    @contextmanager
    def _capture_perception(self):
        model=self.host.model;arrays={'vision':[],'audio':[]}
        original_feed=self.stream.decoder.feed
        def capture_feed(embeds,return_logits=False):
            # Observe the exact native feed; asking for its real final hidden
            # does not add tokens or replace the official pending decision.
            result=original_feed(embeds,return_logits=True)
            arrays['last_input']=embeds[-1:].detach().clone()
            arrays['last_hidden']=result[1][:,-1:].detach().clone()
            arrays['last_logits']=result[0].detach()
            return result if return_logits else None
        self.stream.decoder.feed=capture_feed
        originals={name:getattr(model,name) for name in ('get_vision_embedding','get_audio_embedding_streaming')}
        def collect(value,destination):
            if self.host.torch.is_tensor(value):
                if value.numel():destination.append(value.detach().float().cpu().numpy().reshape(-1,4096))
            elif isinstance(value,(list,tuple)):
                for item in value:collect(item,destination)
        def wrap(name,key):
            def invoke(*args,**kwargs):
                result=originals[name](*args,**kwargs);collect(result,arrays[key]);return result
            return invoke
        model.get_vision_embedding=wrap('get_vision_embedding','vision')
        model.get_audio_embedding_streaming=wrap('get_audio_embedding_streaming','audio')
        try:yield arrays
        finally:
            for name,value in originals.items():setattr(model,name,value)
            self.stream.decoder.feed=original_feed

    def _soft(self,brain_descriptor,body_descriptor=None):
        torch=self.host.torch
        features,state=self.native.codec.read(brain_descriptor)
        device=next(self.host.llm.parameters()).device
        with torch.no_grad():
            x=torch.as_tensor(features,device=device,dtype=torch.float32).unsqueeze(0)
            soft=self.native.to_model(x)
            info={**state,'brain_soft_shape':list(soft.shape),'brain_soft_sha256':sha(soft.float().cpu().numpy().tobytes())}
            if body_descriptor is not None:
                from .embodied_embedding import inject_body_embedding
                soft,body=inject_body_embedding(self.host,soft,body_descriptor)
                layout=body['layout_sha256'];old=self.body_identities.get(layout)
                if old is not None and old!=body['body_embedding_sha256']:raise RuntimeError('Body head changed during duplex session')
                self.body_identities[layout]=body['body_embedding_sha256'];info['body']=body
        return soft.squeeze(0).to(self.host.llm.get_input_embeddings().weight.dtype),info

    def prefill(self,req):
        self._check(req);self.protocol.before_prefill(req)
        user_text=req.get('text','')
        if not isinstance(user_text,str) or len(user_text)>4000:raise ValueError('Duplex text must contain at most 4000 characters')
        if not user_text and not req.get('image_path') and not req.get('audio_path'):raise ValueError('A duplex unit requires actual text, image or audio input')
        # Reuse bounded media loading only; the validation placeholder never
        # enters the native model. Actual text_list below is empty if absent.
        _,_,_,images,audios,modal,fingerprint=self.host._request({**req,'text':user_text or '\u2060'},with_features=False)
        if audios and len(audios[0])!=16000:raise ValueError('Native duplex audio_path must be exactly one second (16000 mono samples); split longer input')
        soft,numeric=self._soft(req.get('brain_state'),req.get('body_features_file'))
        if self.last_brain_time is not None and numeric['simulation_ms']<self.last_brain_time:
            raise ValueError('Brain simulation time moved backwards; a reset requires a new duplex epoch')
        pre_cache=int(self.stream.decoder.get_cache_length());audio=audios[0] if audios else None;padding=0
        # Native first audio after video-only units still needs its first mel
        # window. Pad that missing context explicitly, never invent an audio file.
        if audio is not None:
            target=self.stream.processor.get_streaming_chunk_size()
            available=len(self.stream.audio_buffer)+len(audio)
            if self.stream.audio_chunk_idx>0 and available<target:
                padding=target-available;audio=np.pad(audio,(padding,0))
            elif self.stream.audio_chunk_idx==0:
                padding=max(0,int(self.stream.FIRST_CHUNK_MS*16)-available)
        try:
            with self.host.torch.no_grad(),self._capture_perception() as captured:
                if not self.baseline and self.numeric_placement=='before_sensory':
                    self.stream.decoder.feed(soft)
                detail=self.stream.streaming_prefill(audio_waveform=audio,frame_list=images or None,
                    text_list=[user_text] if user_text else None,max_slice_nums=1,batch_vision_feed=True)
                if not detail.get('success'):raise RuntimeError('Original native streaming prefill did not complete: '+str(detail))
                if not self.baseline and self.numeric_placement=='before_sensory':
                    # The original unit-start hook runs after our prefix. Count
                    # those real tokens in THIS unit so native sliding-window
                    # eviction never leaves untracked numeric KV behind.
                    self.stream.decoder._pending_unit_start_cache_len=pre_cache
                    self.stream.prefill_schema_tokens[-1].insert(0,('brain_body_prefix',int(soft.shape[0])))
                anchor=captured['last_input'];hidden=captured['last_hidden'];native_pending=self.stream.pending_logits
                trace={'original_pending':self._top(self.stream.pending_logits),'last_native_feed':self._top(captured['last_logits'])}
                if not self.baseline and self.numeric_placement=='after_sensory':
                    logits,hidden=self.stream.decoder.feed(soft,return_logits=True)
                    self.stream.pending_logits=logits
                    self.stream.prefill_schema_tokens[-1].append(('brain_body',int(soft.shape[0])))
                    trace['after_brain_body']=self._top(logits)
                    if self.anchor_mode=='last_sensory':
                        logits,hidden=self.stream.decoder.feed(anchor,return_logits=True)
                        self.stream.pending_logits=logits
                        self.stream.prefill_schema_tokens[-1].append(('repeated_native_sensory_anchor',1))
                        trace['after_anchor']=self._top(logits)
                    elif self.anchor_mode=='native_boundary':
                        # Preserve the original sensory-boundary distribution
                        # for the first listen/speak control token only. Numeric
                        # KV is already real; all later generated tokens use it.
                        self.stream.pending_logits=native_pending
                        trace['preserved_native_boundary']=self._top(native_pending)
                current=self.native.to_brain(hidden[0,-1].float()).cpu().numpy().astype(np.float32)
            from connectome_adapter.neuron_currents import write_neuron_currents
            currents=write_neuron_currents(current,self.native.codec.ids)
            arrays={k:np.concatenate(captured[k],axis=0).astype(np.float32) if captured[k] else np.zeros((0,4096),dtype=np.float32) for k in ('vision','audio')}
            arrays['language']=hidden[0,-1:].detach().float().cpu().numpy()
            descriptor=write_tensors(arrays,req['event_id'],directory=ROOT/'artifacts/shared_modal_embeddings',
                identity={k:self.pinned[k] for k in ('model_revision','weights_manifest_sha256','language_adapter_sha256','language_adapter_generation','language_adapter_enabled')})
            result={**self.pinned,**self.protocol.identity(),**{k:v for k,v in numeric.items() if k!='body'},
                'event_id':req['event_id'],'status':'actual_native_duplex_prefill','hidden':arrays['language'][0].tolist(),
                'currents_file':currents,'modal_embeddings_file':descriptor,'body_input':numeric.get('body'),
                'modalities_encoded':[x for x in modal if x!='text' or user_text],'sensory_sha256':fingerprint,
                'image_token_count':len(arrays['vision']),'audio_token_count':len(arrays['audio']),
                'llm_cache_tokens_before':pre_cache,'llm_cache_tokens_after':int(self.stream.decoder.get_cache_length()),
                'audio_padding_samples':padding,'apm_audio_chunk_index':self.stream.audio_chunk_idx,'native_prefill':detail,
                'diagnostic_baseline':self.baseline,'numeric_feedback_injected':not self.baseline,'brain_involved':not self.baseline,
                'decision_anchor':self.anchor_mode,'decode_mode':self.decode_mode,'numeric_placement':self.numeric_placement,'decision_trace':trace}
            if result['llm_cache_tokens_after']<=pre_cache:raise RuntimeError('Original LLM KV did not advance')
            self.protocol.pending={'event_id':req['event_id'],'prefill':result,'anchor':anchor,'hidden':hidden,'native_pending':native_pending};self._write(f'{self.protocol.sequence:04d}-prefill.json',result)
            return result
        except Exception:
            self._release();raise

    def generate(self,req):
        self._check(req);self.protocol.before_generate(req)
        pending=self.protocol.pending;soft,numeric=self._soft(req.get('brain_state'),req.get('body_feedback_file'))
        elapsed=numeric['simulation_ms']-pending['prefill']['simulation_ms']
        if elapsed<=0 or numeric['duration_ms']<=0 or numeric['duration_ms']>elapsed+1e-6:
            raise ValueError('Duplex generate requires a real advanced brain snapshot after prefill')
        facts=req.get('facts');facts_sha=None;torch=self.host.torch
        if facts is not None:
            if not isinstance(facts,dict):raise ValueError('facts must contain measured evidence as an object')
            facts_text=json.dumps(facts,ensure_ascii=False,allow_nan=False)
            if len(facts_text)>10000:raise ValueError('facts too large')
            facts_sha=sha(facts_text.encode())
        try:
            with torch.no_grad():
                trace={'before_feedback':self._top(self.stream.pending_logits)}
                hidden=pending['hidden']
                if facts is not None and not self.baseline:
                    ids=self.stream.tokenizer.encode('\n本单元的实际测量证据（不代表主观思想）：'+facts_text+'\n',add_special_tokens=False)
                    facts_logits,_=self.stream.decoder.feed(self.stream.decoder.embed_tokens(ids),return_logits=True)
                    self.stream.prefill_schema_tokens[-1].extend(ids)
                    trace['after_facts']=self._top(facts_logits);trace['facts_tokens']=len(ids)
                ablation=None
                if req.get('diagnostic_brain_ablation',False) and not self.baseline:
                    import copy
                    control=int(pending['native_pending'][0].argmax().item())
                    # Counterfactual language logits after the SAME native
                    # control token and SAME body/context. Only the four real
                    # post-step brain soft tokens are zeroed in the second arm.
                    zero=soft.clone();zero[-4:]=0
                    arm_logits=[];arm_hidden=[]
                    for arm in (soft,zero):
                        combined=torch.cat((arm,self.stream.decoder.embed_token(control)),dim=0)
                        start=self.stream.decoder.get_cache_length()
                        evaluated=self.host.llm(inputs_embeds=combined.unsqueeze(0),
                            past_key_values=copy.deepcopy(self.stream.decoder.cache),use_cache=True,
                            position_ids=torch.arange(start,start+len(combined),device=combined.device).unsqueeze(0),
                            return_dict=True,output_hidden_states=True)
                        arm_logits.append(evaluated.logits[0,-1].detach().float().cpu().numpy())
                        arm_hidden.append(evaluated.hidden_states[-1][0,-1].detach().float().cpu().numpy())
                        del evaluated
                    delta=arm_logits[0]-arm_logits[1]
                    evidence=write_tensors({'logits_brain':arm_logits[0],'logits_zero_brain':arm_logits[1],
                        'hidden_brain':arm_hidden[0],'hidden_zero_brain':arm_hidden[1]},req['event_id'],directory=self.directory,
                        identity={k:self.pinned[k] for k in ('model_revision','weights_manifest_sha256','language_adapter_sha256','language_adapter_generation','language_adapter_enabled')})
                    ablation={'source':'Actual frozen LLM counterfactual forward; identical copied KV, body soft token and native control token; only four post-step brain soft tokens zeroed',
                        'control_token_id':control,'control_token':self.stream.tokenizer.decode([control]),
                        'logit_l2':float(np.linalg.norm(delta)),'logit_max_abs':float(np.abs(delta).max()),
                        'hidden_l2':float(np.linalg.norm(arm_hidden[0]-arm_hidden[1])),
                        'cache_mutated':False,'file':evidence}
                if not self.baseline:
                    logits,hidden=self.stream.decoder.feed(soft,return_logits=True)
                    self.stream.pending_logits=logits
                    self.stream.prefill_schema_tokens[-1].append(('brain_body_feedback',int(soft.shape[0])))
                    trace['after_brain_body_feedback']=self._top(logits)
                    if self.anchor_mode=='last_sensory':
                        logits,hidden=self.stream.decoder.feed(pending['anchor'],return_logits=True)
                        self.stream.pending_logits=logits
                        self.stream.prefill_schema_tokens[-1].append(('repeated_native_sensory_anchor',1))
                        trace['after_anchor']=self._top(logits)
                    elif self.anchor_mode=='native_boundary':
                        self.stream.pending_logits=pending['native_pending']
                        trace['preserved_native_boundary']=self._top(self.stream.pending_logits)
                history_length=len(self.stream.total_hidden)
                ids_length=len(self.stream.total_ids)
                result=self.stream.streaming_generate(prompt_wav_path=str(self.host.native_speech.reference),
                    max_new_speak_tokens_per_chunk=20,decode_mode=self.decode_mode)
                trace['actual_native_token_ids']=self.stream.total_ids[ids_length:]
                trace['actual_native_tokens']=[self.stream.tokenizer.decode([i]) for i in trace['actual_native_token_ids']]
            serial={k:v for k,v in result.items() if k!='audio_waveform'}
            records=self.stream.total_hidden[-1] if len(self.stream.total_hidden)==history_length+1 else []
            arrays={'text_token_ids':np.array([item[0] for item in records],dtype=np.int64),
                'generation_hidden':torch.cat([item[1].squeeze(0) for item in records],dim=0).float().cpu().numpy() if records else np.zeros((0,4096),dtype=np.float32),
                'post_feedback_hidden':hidden[0,-1:].detach().float().cpu().numpy() if not self.baseline else np.zeros((0,4096),dtype=np.float32)}
            native_tensors=write_tensors(arrays,req['event_id'],directory=self.directory,
                identity={k:self.pinned[k] for k in ('model_revision','weights_manifest_sha256','language_adapter_sha256','language_adapter_generation','language_adapter_enabled')})
            source={'sequence':self.protocol.sequence,'event_id':req['event_id'],'native_tensors_file':native_tensors,'semantic_tokens':len(records)}
            self.history.append(source)
            if records and result.get('text','').strip():self.turn_sources.append(source)
            audio_file=None;wave=result.get('audio_waveform')
            # Upstream emits a zero listen waveform. It is a timing signal,
            # not generated speech, and is never mislabeled as native TTS audio.
            has_wave=not result['is_listen'] and wave is not None and np.asarray(wave).size>0
            if has_wave and self.turn_sources:
                import soundfile as sf
                wave=np.asarray(wave,dtype=np.float32).reshape(-1)
                if not np.isfinite(wave).all():raise RuntimeError('Nonfinite native duplex waveform')
                path=self.directory/f'{self.protocol.sequence:04d}-output.wav';sf.write(path,wave,24000,subtype='PCM_16')
                audio_file={'schema':1,'path':str(path),'sha256':sha(path.read_bytes()),'event_id':req['event_id'],
                    **self.protocol.identity(),
                    'sample_rate':24000,'samples':len(wave),'duration_seconds':len(wave)/24000,'same_generation_hidden':True,'streaming':True,
                    'source':'original persistent MiniCPM duplex hidden -> original TTS.generate_chunk -> native Token2wav.stream',
                    'native_tensors_file':native_tensors,'source_units':self.turn_sources[-24:],
                    'source_note':'Native lookahead/vocoder cache can include preceding units of this same stream; not an independent re-prompted text session',
                    **{k:self.pinned[k] for k in ('model_revision','weights_manifest_sha256','language_adapter_sha256','language_adapter_generation','language_adapter_enabled')}}
            answer={**self.pinned,**self.protocol.identity(),**{k:v for k,v in numeric.items() if k!='body'},**serial,
                'event_id':req['event_id'],'say':result.get('text',''),'status':'actual_native_duplex_chunk',
                'body_input':pending['prefill'].get('body_input'),'body_feedback':numeric.get('body'),
                'modal_embeddings_file':pending['prefill']['modal_embeddings_file'],'facts_sha256':facts_sha,
                'audio_file':audio_file if req.get('generate_audio',True) else None,'audio_muted':not req.get('generate_audio',True),
                'same_generation_hidden':bool(self.turn_sources),'native_tensors_file':native_tensors,
                'nonsemantic_audio_suppressed':bool(has_wave and not self.turn_sources),
                'diagnostic_baseline':self.baseline,'numeric_feedback_injected':not self.baseline,'brain_involved':not self.baseline,
                'decision_anchor':self.anchor_mode,'decode_mode':self.decode_mode,'numeric_placement':self.numeric_placement,'decision_trace':trace,
                'brain_feedback_ablation':ablation,
                'native_window_stats':{
                    'cache_tokens':int(self.stream.decoder.get_cache_length()),
                    'protected_system_tokens':int(self.stream.decoder._system_preserve_length),
                    'retained_unit_lengths':[int(u['length']) for u in self.stream.decoder._unit_history]},
                'native_window_accounting_verified':self.stream.decoder._verify_consistency(),
                'gpu_peak_allocated_bytes':int(torch.cuda.max_memory_allocated()),'gpu_peak_reserved_bytes':int(torch.cuda.max_memory_reserved()),
                'llm_cache_tokens':int(self.stream.decoder.get_cache_length()),'apm_audio_chunk_index':self.stream.audio_chunk_idx}
            if result.get('end_of_turn'):self.turn_sources=[]
            self._write(f'{self.protocol.sequence:04d}-generate.json',answer);self.last_brain_time=numeric['simulation_ms'];self.protocol.completed()
            return answer
        except Exception:
            self._release();raise

    def _release(self):
        if self.closed:return
        self.closed=True;self.protocol.closed=True
        try:
            if self.stream is not None:
                self.stream.set_session_stop();self.stream.decoder.reset()
            self.host.model.reset_session(reset_token2wav_cache=False)
            if hasattr(self.host.processor,'reset_streaming'):self.host.processor.reset_streaming()
            if self.host.native_speech is not None:
                self.host.native_speech.token2wav.stream_cache=None
                self.host.native_speech.token2wav.hift_cache_dict={}
        finally:
            self.host.llm.config.use_cache=self.old_cache
            if self.old_checkpointing:self.host.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            else:self.host.llm.gradient_checkpointing_disable()
            self.host.model.processor=self.old_processor or self.host.processor
            self.protocol.pending=None;self.stream=None

    def close(self,req,cancel=False):
        if req.get('session_id')!=self.protocol.session_id or req.get('epoch')!=self.protocol.epoch:
            raise ValueError('Close belongs to another duplex session/epoch')
        already_closed=self.closed;self._release()
        result={**self.pinned,**self.protocol.identity(),'status':'cancelled' if cancel else 'closed',
            'config_restored':self.host.llm.config.use_cache==self.old_cache,'external_brain_reset':False,
            'native_codec_reused':True,'closed_epoch_outputs_must_be_discarded':True}
        result['already_closed']=already_closed
        self._write('close.json',result);return result
