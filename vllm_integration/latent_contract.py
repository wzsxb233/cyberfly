"""Original numerical couplers, independent of either HF or vLLM execution.

No original model is loaded here. These CPU matrices are the already-trained
per-ID brain and body converters; all actual model hidden states are supplied
by the model worker and their source remains explicit in each descriptor.
"""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import uuid
import zipfile
import time
import math
import threading
from copy import copy, deepcopy
from collections import OrderedDict
from functools import lru_cache

ROOT=Path(__file__).resolve().parents[1]
REVISION='503e754207c94da6bb26850b4469f367c9ea3582'
OMNI_COMMIT='e284d907b5554038bef6ab861bdb78b0b99dd5ea'
ARCHITECTURE='CyberFlyMiniCPMO45ForConditionalGeneration'

def sha(raw):return hashlib.sha256(raw).hexdigest()

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False));temp.replace(path)

def checked_file(descriptor,*,root=ROOT/'artifacts',limit=64*1024*1024):
    from vllm_integration.checked_artifacts import read_checked_artifact
    return read_checked_artifact(descriptor,root=root,limit=limit)


class LatentCoupler:
    def __init__(self,brain_state,*,io_scope='sensory_motor',checkpoint=None):
        import torch
        from safetensors.torch import load
        from shared_io.neuron_link import NeuronStateCodec,build_modules,NeuronLink,digest
        self.torch=torch;self.codec=NeuronStateCodec(io_scope);self.codec.read(brain_state)
        self.manifest_sha=sha((ROOT/'model_training/weights-manifest.json').read_bytes())
        expected_path=ROOT/'shared_io/checkpoints/neuron_direct'/io_scope/'current.safetensors'
        if checkpoint is None:checkpoint={'path':str(expected_path),'sha256':sha(expected_path.read_bytes())}
        _,raw=checked_file(checkpoint,root=ROOT/'shared_io/checkpoints/neuron_direct',limit=64*1024*1024)
        header_len=int.from_bytes(raw[:8],'little')
        metadata=json.loads(json.loads(raw[8:8+header_len])['__metadata__']['identity'])
        for key,value in {'model_revision':REVISION,'weights_manifest_sha256':self.manifest_sha,
                'io_scope':io_scope,'codec_sha256':self.codec.codec_sha,'connection_mode':'neuron_direct',
                'read_ids_sha256':digest(self.codec.ids[self.codec.read_mask]),
                'write_ids_sha256':digest(self.codec.ids[self.codec.write_mask])}.items():
            if metadata.get(key)!=value:raise ValueError('Original numeric checkpoint identity mismatch: '+key)
        state=load(raw)
        self.to_brain,self.to_model=build_modules(torch,io_scope,*self.codec.masks(),metadata['coupler_rank'],device='cpu')
        expected={prefix+key:value for prefix,module in [('to_brain.',self.to_brain),('to_model.',self.to_model)] for key,value in module.state_dict().items()}
        if set(state)!=set(expected) or any(value.shape!=expected[key].shape or value.dtype!=torch.float32 or not torch.isfinite(value).all() for key,value in state.items()):
            raise ValueError('Invalid original per-ID converter tensors')
        if NeuronLink.state_sha(state)!=metadata.get('coupler_sha256'):raise ValueError('Original converter content SHA mismatch')
        for prefix,module in [('to_brain.',self.to_brain),('to_model.',self.to_model)]:
            module.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)})
            module.requires_grad_(False).eval()
        self.identity={key:metadata[key] for key in ('model_revision','weights_manifest_sha256','io_scope','codec_sha256','ids_sha256',
            'coupler_sha256','coupler_generation','coupler_training_steps','read_ids_sha256','write_ids_sha256')}
        self.identity.update(connection_mode='neuron_direct',coupler_checkpoint=dict(checkpoint),
            vllm_omni_commit=OMNI_COMMIT,model='MiniCPM-o-4.5',original_model_weights_frozen=True)
        self.body_heads={}
        self.modal_heads={}
        self.host=SimpleNamespace(torch=torch,weights_manifest_sha=self.manifest_sha,identity=lambda:{'model_revision':REVISION})

    def pin_body_embedding(self, layout, dimension):
        """Load an existing version without the training helper's create fallback.

        The owner calls this only after its checkpoint revision changed. Each
        request subsequently owns a wrapper around these frozen parameters.
        """
        from shared_io.embodied_embedding import BodyEmbedding
        path=ROOT/'shared_io/checkpoints/body_embedding'/layout/'current.safetensors'
        raw=path.read_bytes();file_sha=sha(raw)
        # Reproduce only the original architecture; load() retains all original
        # tensor, metadata and content-SHA checks and never saves a missing head.
        head=BodyEmbedding.__new__(BodyEmbedding)
        head.host,head.torch,head.layout_sha,head.dimension=self.host,self.torch,layout,dimension
        head.checkpoint=path;head.steps=0
        nn=self.torch.nn
        with self.torch.random.fork_rng(devices=[]):
            head.module=nn.Sequential(nn.LayerNorm(dimension),nn.Linear(dimension,128),
                nn.SiLU(),nn.Linear(128,4096)).to(device='cpu',dtype=self.torch.float32)
        head.load(path)
        if sha(path.read_bytes())!=file_sha:
            raise ValueError('Body embedding changed during full checkpoint verification')
        head.module.requires_grad_(False).eval()
        self.body_heads[layout]=(file_sha,head)
        return head

    def fork_request(self):
        """Separate mutable codecs/caches; share only frozen inference modules.

        No observation is decoded here. condition() validates this request's
        actual files in its worker before publication. currents() is stateless
        and uses its own already-verified ID vector, so the published request
        can subsequently be consumed by the body owner without codec access.
        """
        for module in (self.to_brain,self.to_model):
            if module.training or any(p.requires_grad for p in module.parameters()):
                raise ValueError('Request heads require frozen inference parameters')
        result=copy(self)
        result.codec=deepcopy(self.codec)
        for name in ('ids','read_mask','write_mask'):
            getattr(result.codec,name).setflags(write=False)
        result.identity=deepcopy(self.identity)
        result._current_ids=result.codec.ids.copy();result._current_ids.setflags(write=False)
        result.host=SimpleNamespace(torch=self.torch,weights_manifest_sha=self.manifest_sha,
                                  identity=lambda:{'model_revision':REVISION})
        result.body_heads={}
        for layout,(file_sha,head) in self.body_heads.items():
            if head.module.training or any(p.requires_grad for p in head.module.parameters()):
                raise ValueError('Request body embeddings must be frozen')
            private=copy(head);private.host=result.host
            result.body_heads[layout]=(file_sha,private)
        result.modal_heads={}
        result._request_thread=None;result._request_lock=threading.Lock()
        result._pinned_request=True
        return result

    def _request_owner(self):
        if not getattr(self,'_pinned_request',False):return
        current=threading.get_ident()
        with self._request_lock:
            if self._request_thread is None:self._request_thread=current
            elif self._request_thread!=current:
                raise RuntimeError('This request mutable codec belongs to its preparation worker')

    def body_soft(self,descriptor):
        self._request_owner()
        from shared_io.embodied_embedding import BodyEmbedding,read_body_features
        features,info=read_body_features(descriptor)
        layout=descriptor['layout_sha256'];path=ROOT/'shared_io/checkpoints/body_embedding'/layout/'current.safetensors'
        if getattr(self,'_pinned_request',False):
            if layout not in self.body_heads:
                raise ValueError('Body layout was not pinned for this request')
            file_sha,head=self.body_heads[layout]
            if head.dimension!=len(features):raise ValueError('Pinned body feature dimension changed')
        else:
            if not path.is_file():raise ValueError('Migration requires an existing verified body embedding; no random fallback')
            file_sha=sha(path.read_bytes())
            if layout not in self.body_heads or self.body_heads[layout][0]!=file_sha:
                self.body_heads[layout]=(file_sha,BodyEmbedding(self.host,layout,len(features),'cpu'))
        head=self.body_heads[layout][1]
        with self.torch.no_grad():soft=head.forward(features).reshape(1,4096).detach().float()
        return soft,{**info,**head.identity(),'checkpoint_file_sha256':file_sha}

    def condition(self,event_id,brain_state,*,body_features_file=None,body_feedback_file=None,slot_start=0):
        self._request_owner()
        import numpy as np
        if not isinstance(event_id,str) or not event_id or len(event_id)>128:raise ValueError('Invalid bounded event identity')
        if type(slot_start)is not int or slot_start<0 or slot_start>2048:raise ValueError('Invalid token slot offset')
        features,brain_info=self.codec.read(brain_state)
        with self.torch.no_grad():tokens=self.to_model(self.torch.from_numpy(features)).reshape(4,4096)
        body_info={}
        if body_features_file is not None:
            prior,body_info['body_input']=self.body_soft(body_features_file)
            tokens=self.torch.cat([tokens,prior],dim=0)
        if body_feedback_file is not None:
            if body_features_file is not None and body_feedback_file == body_features_file:
                # A first turn has the same actual body observation in both
                # slots. Reuse this call's verified tensor, retaining both slots.
                from copy import deepcopy
                feedback = prior
                body_info['body_feedback'] = deepcopy(body_info['body_input'])
            else:
                feedback,body_info['body_feedback']=self.body_soft(body_feedback_file)
            tokens=self.torch.cat([feedback,tokens],dim=0)
        values=np.ascontiguousarray(tokens.detach().float().numpy())
        folder=ROOT/'artifacts/vllm_latent'/sha(event_id.encode())[:24];folder.mkdir(parents=True,exist_ok=True)
        path=folder/('soft-'+uuid.uuid4().hex+'.npz')
        with path.open('xb') as stream:np.savez(stream,soft_tokens=values)
        descriptor={'schema':1,'event_id':event_id,'path':str(path),'sha256':sha(path.read_bytes()),
            'soft_tokens_sha256':sha(values.tobytes()),'shape':list(values.shape),'dtype':'float32',
            'slot_start':slot_start,'slot_count':len(values),'brain_state':brain_state,
            'body_features_file':body_features_file,'body_feedback_file':body_feedback_file,
            'brain_source':brain_info,**body_info,**self.identity,
            'facts_in_model_input':False,'placeholder_policy':'Reserve exactly these token positions; replace their embeddings without changing length or KV positions'}
        write_json(path.with_suffix('.json'),descriptor);return descriptor

    def currents(self,hidden,*,event_id,source):
        import numpy as np
        from connectome_adapter.neuron_currents import write_neuron_currents
        value=self.torch.as_tensor(hidden,dtype=self.torch.float32).reshape(-1)
        if value.shape!=(4096,) or not self.torch.isfinite(value).all():raise ValueError('Expected an actual finite 4096-dimensional model hidden state')
        with self.torch.no_grad():current=self.to_brain(value).cpu().numpy().astype(np.float32)
        ids=self._current_ids if getattr(self,'_pinned_request',False) else self.codec.ids
        result=write_neuron_currents(current,ids)
        return {'event_id':event_id,'currents_file':result,'hidden_sha256':sha(value.numpy().tobytes()),
            'hidden_source':source,**self.identity}

    def modal_currents(self,descriptor,*,event_id):
        """Reuse only the two existing, identity-checked original-token heads."""
        self._request_owner()
        from shared_io.modal_currents import ModalCurrentAdapter
        result={}
        for modality in ('vision','audio'):
            pointer=ROOT/'artifacts/sensory_checkpoints'/modality/'current.json'
            if not pointer.is_file():raise ValueError('No saved '+modality+' adapter; migration will not invent one')
            raw=pointer.read_bytes();pointer_sha=sha(raw);record=json.loads(raw)
            if modality not in self.modal_heads or self.modal_heads[modality][0]!=pointer_sha:
                adapter=ModalCurrentAdapter.load(record['path']);identity=adapter.describe()
                expected=record.get('identity',{})
                if adapter.modality!=modality or any(identity.get(k)!=expected.get(k) for k in
                        ('weights_sha256','configuration_sha256','optimizer_updates','model_revision','weights_manifest_sha256')):
                    raise ValueError('Existing modality head pointer identity mismatch')
                adapter.network.requires_grad_(False).eval()
                self.modal_heads[modality]=(pointer_sha,adapter)
            adapter=self.modal_heads[modality][1]
            result[modality]=adapter.project(descriptor,event_id=event_id,model_identity=self.identity)
            result[modality]['pointer_sha256']=pointer_sha
        return result


def load_condition(descriptor):
    import numpy as np
    _,raw=checked_file(descriptor,root=ROOT/'artifacts/vllm_latent',limit=256*1024)
    if descriptor.get('schema')!=1 or descriptor.get('vllm_omni_commit')!=OMNI_COMMIT or descriptor.get('model_revision')!=REVISION:
        raise ValueError('Unknown latent condition schema/model/runtime revision')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if archive.namelist()!=['soft_tokens.npy'] or sum(x.file_size for x in archive.infolist())>256*1024:raise ValueError('Invalid bounded numerical soft-token file')
    with np.load(io.BytesIO(raw),allow_pickle=False) as archive:array=archive['soft_tokens'].copy()
    if (array.dtype!=np.dtype('<f4') or array.ndim!=2 or not 4<=len(array)<=6 or array.shape[1]!=4096 or
            not np.isfinite(array).all() or abs(array).max()>.150001 or list(array.shape)!=descriptor.get('shape') or
            sha(array.tobytes())!=descriptor.get('soft_tokens_sha256') or len(array)!=descriptor.get('slot_count')):
        raise ValueError('Soft-token shape, finite values, bounds or content identity changed')
    return array


_tokenizer_lock=threading.Lock()
_tokenizer_launch_lock=threading.Lock()
_tokenizer_instance=None
_tokenizer_thread=None
_tokenizer_state={'ready':False,'state':'not_started','load_seconds':None,'initializations':0,
    'scope':'Original static local tokenizer only; no visual/brain input or model inference',
    'cost_accounting':'Preheating moves one-time startup work earlier; it does not eliminate computation'}


@lru_cache(maxsize=1)
def _original_template_tokenizer():
    global _tokenizer_instance,_tokenizer_state
    with _tokenizer_lock:
        if _tokenizer_instance is not None:
            return _tokenizer_instance
        started=time.perf_counter()
        _tokenizer_state={**_tokenizer_state,'state':'loading','started_at':time.time()}
        try:
            from transformers import PreTrainedTokenizerFast
            config_path=ROOT/'model_training/models/tokenizer_config.json'
            tokenizer_path=ROOT/'model_training/models/tokenizer.json'
            config_raw=config_path.read_bytes();config=json.loads(config_raw)
            tokenizer=PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path),
                chat_template=config['chat_template'],eos_token='<|im_end|>',unk_token='<unk>')
            config_sha=sha(config_raw);tokenizer_sha=sha(tokenizer_path.read_bytes())
            _tokenizer_state={**_tokenizer_state,'state':'ready','ready':True,
                'load_seconds':time.perf_counter()-started,'initializations':1,
                'config_sha256':config_sha,'tokenizer_sha256':tokenizer_sha}
            _tokenizer_instance=tokenizer
        except Exception as exc:
            _tokenizer_state={**_tokenizer_state,'state':'failed','ready':False,
                'load_seconds':time.perf_counter()-started,'error':str(exc)}
            raise
        return _tokenizer_instance


def tokenizer_preheat_status():
    return dict(_tokenizer_state)


def start_tokenizer_preheat():
    """Nonblocking static startup work. Request callers await the same instance."""
    global _tokenizer_thread
    def run():
        try:_original_template_tokenizer()
        except Exception:pass  # The observable state retains the exact error.
    with _tokenizer_launch_lock:
        if _tokenizer_thread is None:
            _tokenizer_thread=threading.Thread(target=run,name='cyberfly-static-tokenizer',daemon=True)
            _tokenizer_thread.start()
    return tokenizer_preheat_status()


def build_latent_content_prefix(condition,*,speak=False):
    """First content item of exactly one user message, before any raw media.

    The upstream processor expands later media placeholders. It cannot shift
    this preceding reserved span; the model worker verifies actual token IDs.
    """
    load_condition(condition)
    tokenizer=_original_template_tokenizer();token='<|fim_pad|>';token_id=151662
    if tokenizer.encode(token,add_special_tokens=False)!=[token_id]:raise RuntimeError('Original reserved token identity changed')
    marker=token*condition['slot_count']
    ids=tokenizer.apply_chat_template([{'role':'user','content':marker}],tokenize=True,
        add_generation_prompt=True,enable_thinking=False,use_tts_template=bool(speak))
    if hasattr(ids,'keys'):ids=ids['input_ids']
    hits=[i for i in range(len(ids)-condition['slot_count']+1) if ids[i:i+condition['slot_count']]==[token_id]*condition['slot_count']]
    if len(hits)!=1:raise RuntimeError('Original template does not yield one unambiguous numerical slot span')
    descriptor={**condition,'slot_start':hits[0],'slot_token_id':token_id,
        'template_contract':'one user message; this text item precedes every image/audio/text item',
        'template_sha256':sha(tokenizer.chat_template.encode())}
    return {'content_part':{'type':'text','text':marker},'additional_information':{'cyberfly_latent':descriptor}}


class LatestHiddenPending(FileNotFoundError):
    """No verified pair yet; the caller can keep stepping the physical body.

    A short first mismatch grace is inconclusive, not proof of writer progress.
    Stable corruption or endlessly inconsistent updates become ValueError.
    """
    def __init__(self, *, reason, elapsed_s, observed_changes):
        super().__init__('Latest hidden publication pending: '+reason)
        self.reason=reason;self.elapsed_s=elapsed_s;self.observed_changes=observed_changes


_hidden_publication_lock=threading.Lock()
_hidden_publications=OrderedDict()
_HIDDEN_STABLE_GRACE_S=.35
_HIDDEN_TOTAL_GRACE_S=2.0


def _hidden_identity(descriptor,event_id,request_id,max_age_s):
    if not isinstance(descriptor,dict) or descriptor.get('schema')!=1:
        raise ValueError('Invalid latest hidden descriptor')
    actual_request=descriptor.get('request_id')
    if descriptor.get('event_id')!=event_id or not isinstance(actual_request,str) or not 1<=len(actual_request)<=256 or request_id is not None and actual_request!=request_id:
        raise ValueError('Latest hidden event/request identity mismatch')
    if descriptor.get('model_revision')!=REVISION or descriptor.get('vllm_omni_commit')!=OMNI_COMMIT:
        raise ValueError('Unexpected latest hidden original model/runtime identity')
    if type(descriptor.get('sequence')) is not int or descriptor['sequence']<0:
        raise ValueError('Invalid latest hidden generation sequence')
    published=descriptor.get('published_at')
    if isinstance(published,bool) or not isinstance(published,(int,float)) or not math.isfinite(published):
        raise ValueError('Invalid latest hidden publication time')
    age=time.time()-published
    if age>max_age_s:raise ValueError('Latest model hidden expired')
    if age < -5:raise ValueError('Latest model hidden publication time is in the future')
    for name in ('sha256','hidden_sha256','condition_file_sha256','soft_tokens_sha256'):
        value=descriptor.get(name)
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Invalid latest hidden digest: '+name)
    return actual_request


def _hidden_pointer(path):
    from vllm_integration.checked_artifacts import read_bounded_artifact
    _,raw=read_bounded_artifact(path,root=ROOT/'artifacts/vllm_latent',limit=256*1024)
    return raw


def _hidden_uncommitted(key,fingerprint):
    now=time.monotonic()
    with _hidden_publication_lock:
        previous=_hidden_publications.get(key)
        if previous is None:
            previous={'first':now,'stable_since':now,'fingerprint':fingerprint,'changes':0}
        elif previous['fingerprint']!=fingerprint:
            previous={**previous,'stable_since':now,'fingerprint':fingerprint,'changes':previous['changes']+1}
        _hidden_publications[key]=previous;_hidden_publications.move_to_end(key)
        while len(_hidden_publications)>64:_hidden_publications.popitem(last=False)
        elapsed=now-previous['first'];stable=now-previous['stable_since']
        if stable>=_HIDDEN_STABLE_GRACE_S:
            raise ValueError('Stable latest hidden artifact SHA mismatch; publication did not commit')
        if elapsed>=_HIDDEN_TOTAL_GRACE_S:
            raise ValueError('Latest hidden remained inconsistent across changing publications')
        reason='observed_concurrent_replacement' if previous['changes'] else 'uncommitted_pair_grace_not_yet_classified'
        raise LatestHiddenPending(reason=reason,elapsed_s=elapsed,observed_changes=previous['changes'])


def read_latest_hidden(event_id,*,request_id=None,max_age_s=30):
    """Bounded latest-pair acquisition; no sleep on the physical actor.

    Parent JSON is only a request locator. The authoritative child pointer is
    read immediately, so a stale parent never lengthens the three-file window.
    At most two pointer/file reads happen per call. An uncommitted pair yields
    LatestHiddenPending; repeated stable mismatch fails within350ms of polling,
    and even changing invalid publications fail within2s. A verified old pair
    may be returned: its exact bytes are immutable in memory and SHA-checked.
    """
    import numpy as np
    from vllm_integration.checked_artifacts import read_bounded_artifact
    if not isinstance(event_id,str) or not 1<=len(event_id)<=128:
        raise ValueError('Invalid latest hidden event identity')
    if not isinstance(max_age_s,(int,float)) or isinstance(max_age_s,bool) or not math.isfinite(max_age_s) or not 0<max_age_s<=3600:
        raise ValueError('Invalid latest hidden TTL')
    event_folder=ROOT/'artifacts/vllm_latent'/sha(event_id.encode())[:24]
    if request_id is None:
        parent=json.loads(_hidden_pointer(event_folder/'latest.json'))
        request_id=_hidden_identity(parent,event_id,None,max_age_s)
    elif not isinstance(request_id,str) or not 1<=len(request_id)<=256:
        raise ValueError('Invalid latest hidden request identity')
    folder=event_folder/sha(request_id.encode())[:24]
    pointer=folder/'latest.json';expected_file=folder/'latest-hidden.npz'
    # Normal publishers provide this exact path. Its actual read is confined
    # by openat2; alternative spellings retain the original full resolve check.
    key=(str(folder),event_id,request_id)
    fingerprint=None
    for attempt in range(2):
        pointer_raw=_hidden_pointer(pointer)
        descriptor=json.loads(pointer_raw)
        _hidden_identity(descriptor,event_id,request_id,max_age_s)
        described_path=Path(descriptor.get('path',''))
        if described_path!=expected_file and described_path.resolve()!=expected_file.resolve():
            raise ValueError('Latest hidden path does not belong to its exact event/request')
        try:
            _,raw=read_bounded_artifact(expected_file,root=ROOT/'artifacts/vllm_latent',limit=128*1024)
        except FileNotFoundError:
            raw=None
        if raw is not None and len(raw)>128*1024:raise ValueError('Oversized latest hidden artifact')
        actual_sha=sha(raw) if raw is not None else 'missing'
        fingerprint=(sha(pointer_raw),actual_sha)
        if actual_sha!=descriptor['sha256']:
            continue  # One immediate reread can observe the completed child commit.
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                if set(archive.namelist())!={'hidden.npy','input_token_ids.npy'} or any(item.file_size>128*1024 for item in archive.infolist()):
                    raise ValueError('Unexpected/oversized latest hidden array archive')
            with np.load(io.BytesIO(raw),allow_pickle=False) as archive:
                if set(archive.files)!={'hidden','input_token_ids'}:raise ValueError('Unexpected latest hidden array keys')
                values=archive['hidden'].copy();token_ids=archive['input_token_ids'].copy()
        except (OSError,ValueError,KeyError,EOFError,zipfile.BadZipFile) as exc:
            raise ValueError('Committed latest hidden NPZ is invalid') from exc
        if values.dtype!=np.dtype('<f4') or values.shape!=(1,4096) or not np.isfinite(values).all() or sha(values.tobytes())!=descriptor['hidden_sha256']:
            raise ValueError('Actual latest hidden tensor identity mismatch')
        if token_ids.dtype!=np.dtype('<i8') or token_ids.ndim!=1 or not 1<=len(token_ids)<=4096 or token_ids.tolist()!=descriptor.get('input_token_ids') or len(token_ids)!=descriptor.get('tokens_processed_this_step'):
            raise ValueError('Actual latest hidden processed-token identity mismatch')
        with _hidden_publication_lock:_hidden_publications.pop(key,None)
        return values[0],descriptor
    _hidden_uncommitted(key,fingerprint)


def read_latest_modal_embeddings(event_id,*,request_id=None,max_age_s=30):
    """One immutable actual VPM/APM token file per request; None means pending."""
    from shared_io.modal_currents import read_modal_embeddings
    folder=ROOT/'artifacts/vllm_latent'/sha(event_id.encode())[:24]
    if request_id is not None:folder=folder/sha(request_id.encode())[:24]
    pointer=folder/'modal-latest.json'
    if not pointer.is_file():return None
    descriptor=json.loads(pointer.read_text())
    if descriptor.get('event_id')!=event_id or request_id is not None and descriptor.get('request_id')!=request_id:
        raise ValueError('Perception tokens belong to another event/request')
    if time.time()-descriptor['published_at']>max_age_s:raise ValueError('Perception token snapshot expired')
    if descriptor.get('vllm_omni_commit')!=OMNI_COMMIT:raise ValueError('Unexpected native perception runtime')
    read_modal_embeddings(descriptor,event_id=event_id,expected_model_identity={
        'model_revision':REVISION,'weights_manifest_sha256':sha((ROOT/'model_training/weights-manifest.json').read_bytes())})
    return descriptor
