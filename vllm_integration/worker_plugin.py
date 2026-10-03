"""Official Omni plugin: numerical slots in, actual per-step hidden states out.

The original returned OmniOutput and its latent tensor are never replaced.
Consequently the native Talker/Code2Wav stage consumes the same generation.
"""
from __future__ import annotations
from collections import OrderedDict
import json
from pathlib import Path
import queue
import threading
import time
import uuid

from .latent_contract import ARCHITECTURE,OMNI_COMMIT,ROOT,LatentCoupler,load_condition,sha,write_json

_registered=False

def register():
    global _registered
    if _registered:return
    from vllm.model_executor.models import ModelRegistry
    from vllm_omni.model_executor.models.registry import OmniModelRegistry
    path='vllm_integration.worker_plugin:'+ARCHITECTURE
    OmniModelRegistry.register_model(ARCHITECTURE,path)
    ModelRegistry.register_model(ARCHITECTURE,path)
    from .code2wav_plugin import ARCHITECTURE as speech_arch
    speech_path = 'vllm_integration.code2wav_plugin:' + speech_arch
    OmniModelRegistry.register_model(speech_arch, speech_path)
    ModelRegistry.register_model(speech_arch, speech_path)
    _registered=True


def slot_overlap(start,count,offset,span):
    if any(type(x)is not int for x in (start,count,offset,span)) or min(start,offset,span)<0 or not 4<=count<=6:
        raise ValueError('Invalid numerical/scheduled token span')
    low=max(start,offset);high=min(start+count,offset+span)
    return None if high<=low else (slice(low-offset,high-offset),slice(low-start,high-start))


class LatestHiddenExporter:
    """One latest queue and two atomic files per request; no per-token currents."""
    def __init__(self,trace_limit=0):
        if type(trace_limit)is not int or not 0<=trace_limit<=32:raise ValueError('trace_limit must be 0..32')
        self.queue=queue.Queue(maxsize=1);self.trace_limit=trace_limit
        self.lock=threading.Lock();self.dropped=0;self.records={};self.closed=False
        self.thread=threading.Thread(target=self._run,name='cyberfly-vllm-hidden',daemon=True);self.thread.start()

    def submit(self,hidden,condition,step,coupler):
        # Copy only 4096 floats off the GPU. No full-neuron projection or disk
        # write executes on the forward thread or on the physical-body actor.
        item=(hidden.detach().float().cpu().clone(),condition,dict(step))
        with self.lock:
            if self.closed:return
            try:self.queue.put_nowait(item)
            except queue.Full:
                try:self.queue.get_nowait();self.dropped+=1
                except queue.Empty:pass
                self.queue.put_nowait(item)

    def _run(self):
        while True:
            item=self.queue.get()
            if item is None:return
            hidden,condition,step=item
            event_id=condition['event_id'];request_id=step['request_id']
            folder=ROOT/'artifacts/vllm_latent'/sha(event_id.encode())[:24]/sha(request_id.encode())[:24]
            try:
                import numpy as np
                folder.mkdir(parents=True,exist_ok=True)
                values=hidden.numpy().reshape(1,4096)
                if not np.isfinite(values).all():raise RuntimeError('Original thinker hidden is nonfinite')
                path=folder/'latest-hidden.npz';temp=folder/(uuid.uuid4().hex+'.tmp')
                with temp.open('xb') as stream:
                    np.savez(stream,hidden=values,input_token_ids=np.asarray(step['input_token_ids'],dtype=np.int64))
                raw=temp.read_bytes();temp.replace(path)
                identity={k:condition[k] for k in ('model_revision','weights_manifest_sha256','vllm_omni_commit',
                    'io_scope','codec_sha256','coupler_sha256','coupler_generation','read_ids_sha256','write_ids_sha256')}
                descriptor={'schema':1,**identity,**step,'path':str(path),'sha256':sha(raw),
                    'hidden_shape':[1,4096],'hidden_sha256':sha(values.tobytes()),
                    'condition_file_sha256':condition['sha256'],'soft_tokens_sha256':condition['soft_tokens_sha256'],
                    'dropped_superseded_captures':self.dropped,
                    'same_original_latent_returned_to_native_tts':True,'published_at':time.time(),
                    'original_model_weights_frozen':True,'facts_in_model_input':False,
                    'source':'actual vLLM-Omni thinker forward output','status':'actual_latest_hidden',
                    'currents_projected':False,'retention':'Atomic latest snapshot; consumer archives before real application'}
                write_json(folder/'latest.json',descriptor);write_json(folder.parent/'latest.json',descriptor)
                key=(event_id,request_id);count=self.records.get(key,0)
                if count<self.trace_limit:
                    trace=folder/f"trace-{step['sequence']:06d}.npz";trace.write_bytes(raw)
                    write_json(trace.with_suffix('.json'),{**descriptor,'path':str(trace)})
                    self.records[key]=count+1
            except Exception as exc:
                write_json(folder/'error.json',{'event_id':event_id,'request_id':request_id,'sequence':step['sequence'],
                    'status':'hidden_export_failed','error':str(exc)})


def _model_class():
    import torch
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import MiniCPMO45OmniForConditionalGeneration
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
        MiniCPMO45OmniLLMMultiModalProcessor,MiniCPMO45OmniLLMProcessingInfo,MiniCPMO45OmniLLMDummyInputsBuilder)

    @MULTIMODAL_REGISTRY.register_processor(MiniCPMO45OmniLLMMultiModalProcessor,
        info=MiniCPMO45OmniLLMProcessingInfo,dummy_inputs=MiniCPMO45OmniLLMDummyInputsBuilder)
    class CyberFlyMiniCPMO45ForConditionalGeneration(MiniCPMO45OmniForConditionalGeneration):
        def __init__(self,*,vllm_config,prefix=''):
            if not vllm_config.model_config.enforce_eager:raise ValueError('First verified latent plugin requires enforce_eager')
            if vllm_config.cache_config.enable_prefix_caching:raise ValueError('Latent soft tokens require prefix caching disabled in this version')
            if vllm_config.scheduler_config.max_num_seqs!=1:raise ValueError('Current latent plugin proof is bounded to max_num_seqs=1')
            base=Path(vllm_config.model_config.model).resolve();original=ROOT/'model_training/models'
            index=json.loads((original/'model.safetensors.index.json').read_text())
            if any((base/name).resolve()!=(original/name).resolve() for name in set(index['weight_map'].values())):
                raise ValueError('This migration requires the same verified cached original weight shards')
            super().__init__(vllm_config=vllm_config,prefix=prefix)
            self.requires_grad_(False)
            self._cyberfly_runtime={'quantization':vllm_config.model_config.quantization,
                'compute_dtype':str(vllm_config.model_config.dtype),'enforce_eager':True,
                'prefix_caching':False,'max_num_seqs':1}
            self._cyberfly_pending=[];self._cyberfly_sequences={};self._cyberfly_conditions=OrderedDict()
            self._cyberfly_couplers=OrderedDict();self._cyberfly_exporter=LatestHiddenExporter()

        def preprocess(self,input_ids,input_embeds=None,**kwargs):
            ids,embeds,updates=super().preprocess(input_ids,input_embeds,**kwargs)
            descriptor=kwargs.get('cyberfly_latent')
            if descriptor is None:return ids,embeds,updates
            if self.model_stage!='llm':raise ValueError('Brain numerical inputs belong to the original thinker stage only')
            if kwargs.get('duplex') is not None:raise ValueError('Latent full-duplex unit injection awaits separate native-stream validation')
            if not isinstance(descriptor,dict):raise ValueError('cyberfly_latent must be a strict artifact descriptor')
            request_id=kwargs.get('request_id');offset=kwargs.get('_omni_num_computed_tokens')
            if not isinstance(request_id,str) or type(offset)is not int:raise ValueError('Original runner request identity/position metadata missing')
            key=(request_id,descriptor.get('event_id'),descriptor.get('sha256'))
            if key not in self._cyberfly_conditions:
                self._cyberfly_conditions[key]=(json.loads(json.dumps(descriptor)),load_condition(descriptor))
                if len(self._cyberfly_conditions)>8:self._cyberfly_conditions.popitem(last=False)
            recorded,values=self._cyberfly_conditions[key]
            if descriptor!=recorded:raise ValueError('Latent descriptor metadata changed without its file identity')
            coupler_key=descriptor['coupler_checkpoint']['sha256']
            if coupler_key not in self._cyberfly_couplers:
                coupler=LatentCoupler(descriptor['brain_state'],io_scope=descriptor['io_scope'],checkpoint=descriptor['coupler_checkpoint'])
                if coupler.identity['coupler_sha256']!=descriptor['coupler_sha256']:raise ValueError('Latent converter identity differs from the condition')
                self._cyberfly_couplers[coupler_key]=coupler
                if len(self._cyberfly_couplers)>2:self._cyberfly_couplers.popitem(last=False)
            overlap=slot_overlap(descriptor['slot_start'],len(values),offset,int(ids.numel()))
            injected=0
            if overlap is not None:
                target,source=overlap
                if descriptor.get('slot_token_id')!=151662 or ids[target].detach().cpu().reshape(-1).tolist()!=[151662]*(target.stop-target.start):
                    raise ValueError('Scheduled original tokens do not match the reserved numerical placeholders')
                embeds=embeds.clone();embeds[target]=torch.from_numpy(values[source]).to(device=embeds.device,dtype=embeds.dtype)
                injected=target.stop-target.start
            sequence=self._cyberfly_sequences.get(request_id,0);self._cyberfly_sequences[request_id]=sequence+1
            step={'event_id':descriptor['event_id'],'request_id':request_id,'sequence':sequence,
                'runtime':dict(self._cyberfly_runtime),
                'original_model_trainable_parameters':sum(p.numel() for p in self.parameters() if p.requires_grad),
                'phase':'prefill' if kwargs.get('_omni_is_prefill') else 'generation',
                'token_offset':offset,'tokens_processed_this_step':int(ids.numel()),
                'prompt_tokens':kwargs.get('_omni_prompt_len'),
                'prefill_complete':offset+int(ids.numel())>=int(kwargs.get('_omni_prompt_len',0)),
                'input_token_ids':ids.detach().cpu().reshape(-1).tolist(),'soft_rows_replaced':injected,
                'interpretation':'Actual processed-token hidden; autoregressive sampling of the next token follows this forward'}
            self._cyberfly_pending.append((descriptor,step,self._cyberfly_couplers[coupler_key]))
            return ids,embeds,updates

        def forward(self,*args,**kwargs):
            pending,self._cyberfly_pending=self._cyberfly_pending,[]
            output=super().forward(*args,**kwargs)
            if pending:
                latent=output.text_hidden_states
                if output.multimodal_outputs.get('latent') is not latent:raise RuntimeError('Original native TTS latent alias unexpectedly changed')
                flattened=latent.reshape(-1,latent.shape[-1]);start=0
                for descriptor,step,coupler in pending:
                    count=step['tokens_processed_this_step']
                    if count<1 or start+count>len(flattened):raise RuntimeError('Actual thinker output does not match scheduled token spans')
                    self._cyberfly_exporter.submit(flattened[start+count-1],descriptor,step,coupler);start+=count
            return output  # Identical original object and native TTS latent tensor.

    CyberFlyMiniCPMO45ForConditionalGeneration.__qualname__=ARCHITECTURE
    return CyberFlyMiniCPMO45ForConditionalGeneration


def __getattr__(name):
    if name==ARCHITECTURE:
        cls=_model_class();globals()[name]=cls;return cls
    raise AttributeError(name)
