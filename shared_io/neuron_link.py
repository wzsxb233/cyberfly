"""Per-ID low-rank connection to MiniCPM: no macro-group copies or summaries.

The converter is differentiable. The external spike simulator and its synaptic
updates are not part of this autograd graph; their measured arrays are inputs.
"""
from __future__ import annotations
from collections import OrderedDict
import hashlib
import io
import json
import math
from pathlib import Path
import time
import uuid
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
N = 166700
IDS_SHA = '6b6b40c3bddf84b1281ef0b18db06927c2bf61c5f4cb32a4219ec9b7839dc2e5'
MACRO_SHA = '185289a5e45fa8df33b088828cc92543c16bfdf76d6c728ca596ccf8b22f026c'
SENSORY_CLASSES = {'cb_sensory', 'ol_sensory', 'vnc_sensory'}
READOUT_CLASSES = {'cb_motor', 'vnc_motor', 'descending_neuron'}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(array):
    return sha(np.ascontiguousarray(array).tobytes())


def scalar(value, name, *, low=0, high=None):
    if type(value) not in (int, float) or not math.isfinite(value) or value < low or (high is not None and value > high):
        raise ValueError(name + ' is not a permitted finite number')
    return float(value)


class NeuronStateCodec:
    """Validate complete snapshots; choose explicit ID masks, never group means."""
    def __init__(self, scope='sensory_motor'):
        if scope not in ('sensory_motor', 'all_neurons'):
            raise ValueError('io_scope must be sensory_motor or all_neurons')
        self.scope = scope
        self.catalog = json.loads((ROOT/'artifacts/full_brain/catalog.json').read_text())
        if self.catalog.get('ids_sha256') != IDS_SHA or self.catalog.get('macro_partition_sha256') != MACRO_SHA or self.catalog.get('coverage') != N:
            raise ValueError('Per-neuron codec requires the verified fixed-order 166700-cell release')
        contract = {'schema':1,'connection_mode':'neuron_direct','io_scope':scope,'neurons':N,
            'ids_sha256':IDS_SHA,'macro_partition_sha256':MACRO_SHA,
            'input_arrays':{'v':'float32[N], real membrane mV','counts':'int32[N], spikes in exact last interval'},
            'feature_order':'concatenate (v+65)/30 for every ID, then log1p(spikes/interval_seconds)/log1p(200) for every ID',
            'scope_read_classes':sorted(READOUT_CLASSES) if scope=='sensory_motor' else 'every ID',
            'scope_write_classes':sorted(SENSORY_CLASSES) if scope=='sensory_motor' else 'every ID',
            'tbc_policy':'excluded from sensory_motor ports',
            'currents':'30*tanh(per-ID independent low-rank decoder row), masked exactly outside chosen input IDs'}
        self.codec_sha = sha(json.dumps(contract,sort_keys=True,separators=(',',':')).encode())
        self.contract = {**contract,'codec_sha256':self.codec_sha}
        self.read_mask = self.write_mask = self.ids = None

    def read(self, descriptor):
        if not isinstance(descriptor, dict) or descriptor.get('schema') != 1 or descriptor.get('neurons') != N or descriptor.get('ids_sha256') != IDS_SHA:
            raise ValueError('brain_state descriptor must identify every fixed-order neuron')
        from connectome_adapter.hot_state import read_observation_snapshot
        path,raw=read_observation_snapshot(descriptor)
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            if sum(info.file_size for info in z.infolist())>64*1024*1024:
                raise ValueError('brain_state decompressed arrays exceed 64MiB')
        with np.load(io.BytesIO(raw),allow_pickle=False) as archive:
            required={'ids','v','counts','metadata','macro_group_index'}
            if not required.issubset(archive.files):
                raise ValueError('brain_state lacks real per-ID arrays and identity metadata')
            ids,voltage,counts,groups = [archive[key].copy() for key in ('ids','v','counts','macro_group_index')]
            metadata=json.loads(str(archive['metadata'].item()))
        for value,dtype,name in ((ids,np.dtype('<i8'),'ids'),(voltage,np.dtype('<f4'),'v'),(counts,np.dtype('<i4'),'counts'),(groups,np.dtype('<i4'),'macro_group_index')):
            if value.shape!=(N,) or value.dtype!=dtype:
                raise ValueError(f'{name} must have exact native dtype {dtype} and all {N} entries')
        if digest(ids)!=IDS_SHA or digest(groups)!=MACRO_SHA:
            raise ValueError('Neuron ID order or annotation membership differs from verified release')
        if not np.isfinite(voltage).all() or np.max(np.abs(voltage))>10000 or np.any(counts<0):
            raise ValueError('Raw neuron voltage/count data are invalid')
        if metadata.get('neurons')!=N or metadata.get('ids_sha256')!=IDS_SHA:
            raise ValueError('Snapshot metadata identity differs from descriptor')
        duration=scalar(metadata.get('duration_ms'),'duration_ms',high=200)
        simulation=scalar(metadata.get('simulation_ms'),'simulation_ms')
        if descriptor.get('duration_ms') is not None and descriptor['duration_ms']!=duration:
            raise ValueError('Snapshot interval does not match descriptor')
        if descriptor.get('simulation_ms')!=simulation:
            raise ValueError('Snapshot time does not match descriptor')
        if duration==0 and (simulation!=0 or np.any(counts)):
            raise ValueError('Zero interval is only valid for actual zero-spike reset state')
        if self.ids is None:
            self.ids=ids
            if self.scope=='all_neurons':
                self.read_mask=np.ones(N,dtype=bool);self.write_mask=np.ones(N,dtype=bool)
            else:
                macro=self.catalog['macro_groups']
                self.read_mask=np.isin(groups,[g['index'] for g in macro if g['superclass'] in READOUT_CLASSES])
                self.write_mask=np.isin(groups,[g['index'] for g in macro if g['superclass'] in SENSORY_CLASSES])
                if self.read_mask.sum()!=2129 or self.write_mask.sum()!=17336:
                    raise ValueError('Confirmed sensory/motor annotation counts changed')
        rates=counts.astype(np.float32)/(duration/1000) if duration else np.zeros(N,dtype=np.float32)
        features=np.concatenate([(voltage+65)/30,np.log1p(rates)/np.log1p(200)]).astype(np.float32)
        if not np.isfinite(features).all():
            raise ValueError('Nonfinite raw per-ID features')
        return features,{'brain_state_sha256':descriptor['sha256'],'brain_state_path':str(path),
            'voltage_sha256':digest(voltage),'spikes_sha256':digest(counts),'duration_ms':duration,'simulation_ms':simulation,
            'raw_neurons_read':N,'projected_read_neurons':int(self.read_mask.sum()),
            'addressable_input_neurons':int(self.write_mask.sum()),'features_sha256':digest(features)}

    def masks(self):
        if self.ids is None:
            raise ValueError('Initialize masks from a verified real neuron snapshot first')
        return self.read_mask,self.write_mask


def build_modules(torch, scope, read_mask, write_mask, rank=16, device='cuda'):
    from torch import nn
    if type(rank) is not int or rank not in (8,16):
        raise ValueError('Native low rank must be 8 or 16')
    class ToBrain(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm=nn.LayerNorm(4096)
            self.latent=nn.Linear(4096,rank,bias=False)
            self.per_id=nn.Linear(rank,N,bias=True)
            nn.init.normal_(self.per_id.weight,std=.05)
            nn.init.constant_(self.per_id.bias,.25 if scope=='sensory_motor' else 0)
            self.register_buffer('write_mask',torch.as_tensor(write_mask,dtype=torch.float32),persistent=False)
        def forward(self,hidden):
            output=30*torch.tanh(self.per_id(self.latent(self.norm(hidden))))
            return torch.where(self.write_mask.bool(),output,torch.zeros_like(output))
    class ToModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.per_id=nn.Linear(N*2,rank,bias=False)
            self.soft=nn.Linear(rank,4*4096,bias=False)
            nn.init.normal_(self.per_id.weight,std=1/math.sqrt(2*int(read_mask.sum())))
            self.register_buffer('read_mask',torch.as_tensor(np.tile(read_mask,2),dtype=torch.float32),persistent=False)
        def forward(self,features):
            latent=torch.tanh(self.per_id(features*self.read_mask))
            return (.15*torch.tanh(self.soft(latent))).reshape(-1,4,4096)
    return ToBrain().to(device,dtype=torch.float32),ToModel().to(device,dtype=torch.float32)


class NeuronLink:
    def __init__(self,host,scope,initial_state,rank=16):
        self.host,self.torch,self.scope,self.rank=host,host.torch,scope,rank
        self.codec=NeuronStateCodec(scope)
        self.codec.read(initial_state)
        with self.torch.random.fork_rng(devices=[0]):
            self.torch.manual_seed(1701 if scope=='sensory_motor' else 1702)
            self.to_brain,self.to_model=build_modules(self.torch,scope,*self.codec.masks(),rank)
        self.training_steps,self.generation=0,0
        self.cache=OrderedDict()
        self.checkpoint=ROOT/'shared_io/checkpoints/neuron_direct'/scope/'current.safetensors'
        self.update_sha()
        if self.checkpoint.exists():self.load(self.checkpoint)

    def state(self):
        return {prefix+name:value.detach().contiguous().cpu() for prefix,module in [('to_brain.',self.to_brain),('to_model.',self.to_model)] for name,value in module.state_dict().items()}

    @staticmethod
    def state_sha(state):
        return sha(b''.join(name.encode()+value.float().numpy().tobytes() for name,value in sorted(state.items())))

    def update_sha(self):
        self.coupler_sha=self.state_sha(self.state())

    def identity(self):
        return {'model':'MiniCPM-o-4.5','model_revision':self.host.identity()['model_revision'],
            'weights_manifest_sha256':self.host.weights_manifest_sha,'connection_mode':'neuron_direct','io_scope':self.scope,
            'codec_sha256':self.codec.codec_sha,'ids_sha256':IDS_SHA,'neuron_coverage':N,
            'coupler_sha256':self.coupler_sha,'coupler_generation':self.generation,'coupler_training_steps':self.training_steps,
            'coupler_rank':self.rank,'projected_read_neurons':int(self.codec.read_mask.sum()),
            'addressable_input_neurons':int(self.codec.write_mask.sum()),
            'read_ids_sha256':digest(self.codec.ids[self.codec.read_mask]),
            'write_ids_sha256':digest(self.codec.ids[self.codec.write_mask]),'brain_involved':True,
            'parameter_sharing':'Independent checkpoint per io_scope; each ID has independent low-rank decoder row and encoder columns, no macro-group value copying',
            'differentiation_boundary':'Frozen original LLM + trainable converters; external brain simulator/synaptic plasticity are not end-to-end differentiated',
            'coupler_interpretation':'Engineering learned numerical interface, not proof of natural neural semantics or subjective thought decoding',
            **self.host.language_identity()}

    def save(self,path=None):
        from safetensors.torch import save_file,load
        from .model_link import write_json
        path=Path(path or self.checkpoint);path.parent.mkdir(parents=True,exist_ok=True)
        state=self.state();temp=path.with_suffix('.tmp.safetensors')
        save_file(state,str(temp),metadata={'identity':json.dumps(self.identity(),sort_keys=True)})
        got=load(temp.read_bytes())
        if set(state)!=set(got) or any(not self.torch.equal(v,got[k]) for k,v in state.items()):
            raise RuntimeError('Native checkpoint did not reproduce actual trained parameters')
        temp.replace(path)
        result={**self.identity(),'path':str(path),'file_sha256':sha(path.read_bytes()),'reload_verified':True}
        write_json(path.with_suffix('.json'),result)
        return result

    def load(self,path):
        from safetensors import safe_open
        from safetensors.torch import load
        path=Path(path)
        with safe_open(str(path),framework='pt',device='cpu') as f:metadata=json.loads((f.metadata() or {}).get('identity','{}'))
        identity=self.identity()
        for key in ('model_revision','weights_manifest_sha256','connection_mode','io_scope','codec_sha256','ids_sha256','coupler_rank'):
            if metadata.get(key)!=identity[key]:raise ValueError('Native checkpoint identity mismatch: '+key)
        data=load(path.read_bytes());expected=self.state()
        if set(data)!=set(expected) or any(v.shape!=expected[k].shape or v.dtype!=self.torch.float32 or not self.torch.isfinite(v).all() for k,v in data.items()):
            raise ValueError('Native checkpoint tensors missing, invalid, or nonfinite')
        if self.state_sha(data)!=metadata.get('coupler_sha256'):raise ValueError('Native checkpoint content hash mismatch')
        for prefix,module in [('to_brain.',self.to_brain),('to_model.',self.to_model)]:
            module.load_state_dict({k[len(prefix):]:v for k,v in data.items() if k.startswith(prefix)},strict=True)
        self.training_steps=metadata['coupler_training_steps'];self.generation=metadata['coupler_generation'];self.update_sha()

    def parse(self,req):
        event,text,_,images,audios,modal,fingerprint=self.host._request(req,with_features=False)
        features,state=self.codec.read(req.get('brain_state'))
        return event,text,features,images,audios,modal,fingerprint,state

    def fused(self,emb,features):
        x=self.torch.as_tensor(features,device='cuda',dtype=self.torch.float32).unsqueeze(0)
        soft=self.to_model(x)
        return self.torch.cat([soft.to(emb.dtype),emb],dim=1),soft

    def encode(self,req):
        from connectome_adapter.neuron_currents import write_neuron_currents
        event,text,features,images,audios,modal,fingerprint,state=self.parse(req)
        emb,info=self.host._embedding(text,images,audios,req)
        with self.torch.no_grad():
            fused,soft=self.fused(emb,features)
            hidden=self.host.llm.model(inputs_embeds=fused,use_cache=False,return_dict=True).last_hidden_state[0,-1].float()
            currents=self.to_brain(hidden).cpu().numpy().astype(np.float32)
        self.host._export_hidden(info,hidden)
        descriptor=write_neuron_currents(currents,self.codec.ids)
        self.cache[event]=(emb,fingerprint,self.coupler_sha,info,modal)
        self.cache.move_to_end(event)
        while len(self.cache)>4:self.cache.popitem(last=False)
        active=currents[self.codec.write_mask]
        return {**self.identity(),**state,**info,'event_id':event,'status':'actual_neuron_hidden_state',
            'modalities_encoded':modal,'sensory_sha256':fingerprint,'hidden':hidden.cpu().tolist(),'currents_file':descriptor,
            'brain_soft_shape':list(soft.shape),'brain_soft_sha256':sha(soft.float().cpu().numpy().tobytes()),
            'current_statistics':{'shape':[N],'min_mv':float(currents.min()),'max_mv':float(currents.max()),
                'active_unique_values':int(np.unique(active).size),'active_std_mv':float(active.std()),
                'outside_scope_nonzero':int(np.count_nonzero(currents[~self.codec.write_mask]))},
            'hidden_layer':'original llm.model final RMSNorm, last input token, before lm_head'}

    def decode(self,req):
        event,text,features,images,audios,modal,fingerprint,state=self.parse(req)
        cached=self.cache.get(event)
        if cached is None:raise ValueError('Native decode needs an unexpired encode in the same scope')
        emb,original,coupler,info,modal=cached
        if original!=fingerprint or coupler!=self.coupler_sha:raise ValueError('Native event input or scoped coupler changed')
        self.host._check_body_identity(req,info)
        emb,facts_sha=self.host._append_facts(emb,req.get('facts'))
        torch=self.torch
        measure_ablation=req.get('measure_brain_ablation',True)
        if type(measure_ablation) is not bool:raise ValueError('measure_brain_ablation must be boolean')
        with torch.no_grad():
            fused,soft=self.fused(emb,features)
            fused,body_info=self.host._body_feedback(fused,req)
            hidden=self.host.llm.model(inputs_embeds=fused,use_cache=False).last_hidden_state[:,-1]
            ablation={'performed':False,'reason':'Explicitly skipped for low-latency operation'}
            if measure_ablation:
                logits=self.host.llm.lm_head(hidden).float()
                ablated,_=self.fused(emb,np.zeros(N*2,dtype=np.float32))
                ablated,_=self.host._body_feedback(ablated,req)
                hidden0=self.host.llm.model(inputs_embeds=ablated,use_cache=False).last_hidden_state[:,-1]
                delta=logits-self.host.llm.lm_head(hidden0).float()
                ablation={'performed':True,'comparison':'same raw sensory input and factual text; normalized per-ID v/spike feedback zeroed before learned encoder',
                    'logit_delta_l2':float(delta.norm()),'logit_delta_max_abs':float(delta.abs().max()),
                    'interpretation':'Numeric influence, not proof of accurate brain interpretation'}
            say,speech_info=self.host._generate_reply(fused,req)
        if not say and not speech_info.get('generation_skipped'):raise RuntimeError('Native brain-conditioned model generated empty text')
        return {**self.identity(),**state,**info,'event_id':event,'status':'actual_neuron_conditioned_hidden' if speech_info.get('generation_skipped') else 'actual_neuron_conditioned_generation','say':say,**speech_info,**body_info,
            'modalities_encoded':modal,'sensory_sha256':fingerprint,'facts_sha256':facts_sha,'facts_same_in_ablation':True,
            'brain_soft_shape':list(soft.shape),'brain_soft_sha256':sha(soft.float().cpu().numpy().tobytes()),
            'brain_soft_l2':float(soft.float().norm()),'hidden_sha256':sha(hidden.float().cpu().numpy().tobytes()),
            'brain_ablation':ablation}

    def probe(self,req):
        """Counterfactual numerical sensitivity; never rewrite the real snapshot."""
        event,text,features,images,audios,modal,fingerprint,state=self.parse(req)
        neuron_id=req.get('neuron_id')
        if type(neuron_id) is not int:
            raise ValueError('probe neuron_id must be an exact integer')
        matches=np.flatnonzero(self.codec.ids==neuron_id)
        if len(matches)!=1:raise ValueError('probe neuron_id is not in the actual graph')
        index=int(matches[0]);delta_mv=scalar(req.get('delta_mv',30.0),'delta_mv',low=-30,high=30)
        changed=features.copy();changed[index]+=delta_mv/30
        emb,info=self.host._embedding(text,images,audios,req)
        emb,facts_sha=self.host._append_facts(emb,req.get('facts'))
        torch=self.torch
        with torch.no_grad():
            fused0,soft0=self.fused(emb,features);fused1,soft1=self.fused(emb,changed)
            hidden0=self.host.llm.model(inputs_embeds=fused0,use_cache=False).last_hidden_state[0,-1].float()
            hidden1=self.host.llm.model(inputs_embeds=fused1,use_cache=False).last_hidden_state[0,-1].float()
            logits0=self.host.llm.lm_head(hidden0.to(self.host.llm.lm_head.weight.dtype)).float()
            logits1=self.host.llm.lm_head(hidden1.to(self.host.llm.lm_head.weight.dtype)).float()
            currents0=self.to_brain(hidden0);currents1=self.to_brain(hidden1)
        return {**self.identity(),**state,**info,'event_id':event,'status':'actual_single_neuron_counterfactual',
            'modalities_encoded':modal,'sensory_sha256':fingerprint,'facts_sha256':facts_sha,
            'perturbation':{'neuron_id':neuron_id,'graph_index':index,'delta_mv':delta_mv,
                'within_read_scope':bool(self.codec.read_mask[index]),'changed_features':1,
                'is_recorded_brain_state':False,'description':'Controlled numerical perturbation of one measured voltage; original brain snapshot and real simulator were not changed'},
            'soft_delta_l2':float((soft1-soft0).norm()),'hidden_delta_l2':float((hidden1-hidden0).norm()),
            'logit_delta_l2':float((logits1-logits0).norm()),'logit_delta_max_abs':float((logits1-logits0).abs().max()),
            'currents_delta_l2':float((currents1-currents0).norm()),
            'original_soft_sha256':sha(soft0.float().cpu().numpy().tobytes()),'perturbed_soft_sha256':sha(soft1.float().cpu().numpy().tobytes())}

    def train(self,req):
        from connectome_adapter.neuron_currents import load_neuron_currents
        from .model_link import write_json
        torch=self.torch;records=req.get('records');steps=req.get('steps',2)
        if not isinstance(records,list) or not 1<=len(records)<=128 or type(steps) is not int or not 1<=steps<=10000:
            raise ValueError('Native training requires 1..128 records and 1..10000 steps')
        output=Path(req.get('output','')).resolve()
        if not output.is_relative_to(ROOT/'artifacts') or (output/'progress.json').exists():
            raise ValueError('Native training output must be a new workspace artifact directory')
        output.mkdir(parents=True,exist_ok=True);write_json(output/'input-records.json',records)
        parameters=list(self.to_model.parameters())+list(self.to_brain.parameters())
        before=[p.detach().clone() for p in parameters]
        optimizer=torch.optim.AdamW([{'params':self.to_model.parameters(),'lr':.0002},
            {'params':self.to_brain.parameters(),'lr':.00001}],weight_decay=0)
        progress={**self.identity(),'status':'preparing','component':'MiniCPM per-ID low-rank converters',
            'step':0,'total_steps':steps,'base_model_frozen':True,'brain_synapses_in_autograd':False,
            'initial_coupler_sha256':self.coupler_sha,'trainable_parameters':sum(p.numel() for p in parameters)}
        def emit(**values):
            progress.update(values);progress['updated_at']=time.time()
            write_json(output/'progress.json',progress)
            with (output/'events.jsonl').open('a') as f:f.write(json.dumps(progress,ensure_ascii=False,allow_nan=False)+'\n')
        emit();prepared=[];interrupted=False
        try:
            for record in records:
                if record.get('connection_mode','neuron_direct')!='neuron_direct' or record.get('io_scope',self.scope)!=self.scope:
                    raise ValueError('Do not mix native scopes or modes in one training run')
                _,text,features,images,audios,modal,fingerprint,state=self.parse(record)
                target=record.get('target_text')
                if not isinstance(target,str) or not 1<=len(target)<=1000:raise ValueError('Each native record needs target_text <=1000 chars')
                currents=load_neuron_currents(record.get('target_currents_file'),self.codec.ids,IDS_SHA)
                if np.any(currents[~self.codec.write_mask]!=0):raise ValueError('Training target writes outside selected native sensory scope')
                emb,_=self.host._embedding(text,images,audios,record)
                ids=self.host.processor.tokenizer(target+'<|im_end|>',return_tensors='pt',add_special_tokens=False,
                    return_token_type_ids=False).input_ids.to('cuda')
                if emb.shape[1]+ids.shape[1]+4+int(record.get('body_feedback_file') is not None)>2048:raise ValueError('Native training example exceeds 2048 tokens')
                prepared.append((emb,features,ids,currents,record))
            self.host.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});self.host.llm.train()
            for step in range(steps):
                if any(p.exists() for p in (output/'STOP',output/'stop.request',output.parent/'STOP')):
                    interrupted=True;break
                emb,features,ids,target_currents,record=prepared[step%len(prepared)]
                fused,soft=self.fused(emb,features)
                fused,training_body_info=self.host._body_feedback(fused,record)
                with torch.no_grad():
                    response_embeddings=self.host.llm.get_input_embeddings()(ids)
                    hidden=self.host.llm.model(inputs_embeds=fused.detach(),use_cache=False).last_hidden_state[0,-1].float()
                labels=torch.cat([torch.full(fused.shape[:2],-100,device='cuda',dtype=torch.long),ids],dim=1)
                optimizer.zero_grad(set_to_none=True)
                answer=self.host.llm(inputs_embeds=torch.cat([fused,response_embeddings],dim=1),labels=labels,use_cache=False)
                prediction=self.to_brain(hidden)
                target_tensor=torch.as_tensor(target_currents,dtype=torch.float32,device='cuda')
                mask=self.to_brain.write_mask.bool()
                current_loss=torch.nn.functional.mse_loss(prediction[mask]/30,target_tensor[mask]/30)
                loss=answer.loss+current_loss
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite native converter loss')
                loss.backward()
                gradients={name:sum(float(p.grad.float().square().sum()) for p in module.parameters() if p.grad is not None)**.5
                    for name,module in [('brain_to_model_grad_norm',self.to_model),('model_to_brain_grad_norm',self.to_brain)]}
                norm=torch.nn.utils.clip_grad_norm_(parameters,1.0);optimizer.step();self.training_steps+=1
                delta=sum(float((p.detach()-old).square().sum()) for p,old in zip(parameters,before))**.5
                emit(status='training',step=step+1,loss=float(loss),language_loss=float(answer.loss),current_imitation_loss=float(current_loss),
                    grad_norm=float(norm),**gradients,adapter_delta_l2=delta,learning_rate=.0002,current_learning_rate=.00001,
                    gpu_memory_mb=round(torch.cuda.max_memory_allocated()/1024**2,1))
                del answer,loss,soft,fused,hidden
            self.update_sha();self.generation+=1
            checkpoint=self.save(output/'neuron-coupler.safetensors');self.save()
            expected=self.coupler_sha;self.load(output/'neuron-coupler.safetensors')
            if expected!=self.coupler_sha:raise RuntimeError('Native trained checkpoint reload differs')
            emit(status='interrupted' if interrupted else 'completed',stage='saved_and_reloaded',**self.identity(),
                checkpoint=checkpoint,reload_verified=True,
                limitation='Real converter optimization only; no differentiation through the external connectome or proof of competent/biologically faithful control')
            write_json(output/'training-evidence.json',progress)
            return progress
        except Exception as exc:
            self.update_sha();self.generation+=1;emit(status='failed',error=str(exc),**self.identity());raise
        finally:
            self.host.llm.eval();self.host.llm.gradient_checkpointing_disable();torch.cuda.empty_cache()
