"""Verified physical-body vectors as real trainable MiniCPM input embeddings.

This is an engineering numerical adapter. It does not turn simulator quantities
into natural fly sensations or change the original model weights.
"""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import time
import uuid
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def sha(raw):return hashlib.sha256(raw).hexdigest()

def _artifact(value, limit=8*1024*1024):
    p=Path(value).resolve()
    if not p.is_relative_to(ROOT/'artifacts') or not p.is_file() or p.stat().st_size>limit:
        raise ValueError('Physical-body inputs must be bounded local artifacts')
    return p,p.read_bytes()

def read_body_features(descriptor):
    from .body_senses import BodySenseCodec
    if not isinstance(descriptor,dict) or descriptor.get('schema')!=1:
        raise ValueError('Expected actual body features descriptor schema 1')
    path,raw=_artifact(descriptor.get('path',''))
    if sha(raw)!=descriptor.get('sha256'):raise ValueError('Body feature file SHA changed')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        if len(z.infolist())!=1 or z.namelist()!=['features.npy'] or sum(i.file_size for i in z.infolist())>131072:
            raise ValueError('Expected one bounded body features array')
    with np.load(io.BytesIO(raw),allow_pickle=False) as z:features=z['features'].copy()
    if features.dtype!=np.dtype('<f4') or features.ndim!=1 or not 1<=features.size<=8192 or not np.isfinite(features).all() or list(features.shape)!=descriptor.get('shape'):
        raise ValueError('Expected all finite float32 body feature entries')
    _,packet_raw=_artifact(descriptor.get('packet_path',''))
    if sha(packet_raw)!=descriptor.get('packet_file_sha256'):raise ValueError('Original body packet bytes changed')
    packet=json.loads(packet_raw)
    codec=BodySenseCodec(packet)
    actual=codec.encode(packet)
    if (codec.identity['feature_layout_sha256']!=descriptor.get('layout_sha256') or
            packet['packet_sha256']!=descriptor.get('packet_sha256') or
            packet['simulation_time_s']!=descriptor.get('simulation_time_s') or not np.array_equal(features,actual)):
        raise ValueError('Body vector differs from its measured packet, layout or physical time')
    return features,{'body_features_path':str(path),'body_features_sha256':descriptor['sha256'],
        'body_packet_sha256':packet['packet_sha256'],'body_layout_sha256':descriptor['layout_sha256'],
        'body_feature_dim':features.size,'body_simulation_time_s':packet['simulation_time_s']}

class BodyEmbedding:
    def __init__(self,host,layout_sha,dimension,device):
        self.host,self.torch,self.layout_sha,self.dimension=host,host.torch,layout_sha,dimension
        torch=host.torch
        devices=[torch.cuda.current_device()] if str(device).startswith('cuda') else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(layout_sha[:8],16))
            self.module=torch.nn.Sequential(torch.nn.LayerNorm(dimension),torch.nn.Linear(dimension,128),torch.nn.SiLU(),torch.nn.Linear(128,4096)).to(device=device,dtype=torch.float32)
            torch.nn.init.normal_(self.module[-1].weight,std=.01);torch.nn.init.zeros_(self.module[-1].bias)
        self.module.requires_grad_(False).eval()
        self.steps=0
        self.checkpoint=ROOT/'shared_io/checkpoints/body_embedding'/layout_sha/'current.safetensors'
        if self.checkpoint.exists():self.load(self.checkpoint)
        else:self.update_sha();self.save()
    def state(self):return {k:v.detach().contiguous().cpu() for k,v in self.module.state_dict().items()}
    def update_sha(self):
        self.weights_sha=sha(b''.join(k.encode()+v.numpy().tobytes() for k,v in sorted(self.state().items())))
    def identity(self):
        return {'component':'body_numeric_embedding','layout_sha256':self.layout_sha,'feature_dim':self.dimension,
            'model_revision':self.host.identity()['model_revision'],'weights_manifest_sha256':self.host.weights_manifest_sha,
            'body_embedding_sha256':self.weights_sha,'training_steps':self.steps,'soft_tokens':1,'soft_dimension':4096,
            'training_target':'Physical features to explicit paired model representations; original MiniCPM and brain weights are separate'}
    def forward(self,features):
        t=self.torch.as_tensor(features,dtype=self.torch.float32,device=next(self.module.parameters()).device)
        return .05*self.torch.tanh(self.module(t))
    def save(self,path=None):
        from safetensors.torch import save_file,load
        p=Path(path or self.checkpoint);p.parent.mkdir(parents=True,exist_ok=True)
        state=self.state();self.update_sha();temporary=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
        save_file(state,str(temporary),metadata={'cyberfly_body_embedding':json.dumps(self.identity(),sort_keys=True)})
        loaded=load(temporary.read_bytes())
        if set(loaded)!=set(state) or not all(self.torch.equal(state[k],loaded[k]) for k in state):raise RuntimeError('Body embedding checkpoint did not round-trip')
        temporary.replace(p)
        return {**self.identity(),'path':str(p),'file_sha256':sha(p.read_bytes()),'reload_verified':True}
    def load(self,path):
        from safetensors.torch import load
        raw=Path(path).read_bytes();length=int.from_bytes(raw[:8],'little')
        metadata=json.loads(json.loads(raw[8:8+length])['__metadata__']['cyberfly_body_embedding'])
        for key,expected in [('layout_sha256',self.layout_sha),('feature_dim',self.dimension),('model_revision',self.host.identity()['model_revision']),('weights_manifest_sha256',self.host.weights_manifest_sha)]:
            if metadata.get(key)!=expected:raise ValueError('Body embedding checkpoint identity changed: '+key)
        state=load(raw);expected=self.module.state_dict()
        if set(state)!=set(expected) or any(v.shape!=expected[k].shape or v.dtype!=self.torch.float32 or not self.torch.isfinite(v).all() for k,v in state.items()):raise ValueError('Body embedding has invalid tensors')
        actual=sha(b''.join(k.encode()+v.numpy().tobytes() for k,v in sorted(state.items())))
        if actual!=metadata['body_embedding_sha256']:raise ValueError('Body embedding parameter hash changed')
        self.module.load_state_dict(state);self.steps=metadata['training_steps'];self.update_sha()
        if type(self.steps)is not int or self.steps<0:raise ValueError('Invalid body embedding training count')

def _head(host,descriptor,device):
    features,info=read_body_features(descriptor)
    if not hasattr(host,'body_embeddings'):host.body_embeddings={}
    key=descriptor['layout_sha256']
    if key not in host.body_embeddings:host.body_embeddings[key]=BodyEmbedding(host,key,len(features),device)
    head=host.body_embeddings[key]
    if head.dimension!=len(features):raise ValueError('Body dimension changed for same layout')
    return head,features,info

def inject_body_embedding(host,emb,descriptor):
    if descriptor is None:return emb,{'body_numeric_input':False}
    head,features,info=_head(host,descriptor,emb.device)
    with host.torch.no_grad():soft=head.forward(features).reshape(1,1,4096)
    fused=host.torch.cat([soft.to(emb.dtype),emb],dim=1)
    return fused,{**info,**head.identity(),'body_numeric_input':True,'body_soft_shape':list(soft.shape),
        'body_soft_sha256':sha(soft.float().cpu().numpy().tobytes()),'body_soft_l2':float(soft.norm())}

def save_body_embeddings(host):
    return {key:head.save() for key,head in getattr(host,'body_embeddings',{}).items()}

def train_body_embedding(host,req):
    """Real supervised fit against explicitly archived paired LLM hidden states."""
    records=req.get('records');steps=req.get('steps',2)
    if not isinstance(records,list) or not 1<=len(records)<=128 or type(steps)is not int or not 1<=steps<=10000:raise ValueError('Use 1..128 real records and 1..10000 steps')
    output=Path(req.get('output','')).resolve()
    if not output.is_relative_to(ROOT/'artifacts') or (output/'progress.json').exists():raise ValueError('Use a new local training artifact directory')
    output.mkdir(parents=True,exist_ok=True)
    def write(name,data):(output/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    write('input-records.json',records)
    rows=[];targets=[];head=None
    for record in records:
        candidate,features,_=_head(host,record['body_features_file'],next(host.llm.parameters()).device)
        if head is not None and head is not candidate:raise ValueError('One body layout per training job')
        head=candidate
        _,raw=_artifact(record['target_hidden_path'],limit=1024*1024)
        if sha(raw)!=record.get('target_hidden_file_sha256'):raise ValueError('Recorded target hidden bytes changed')
        hidden=np.load(io.BytesIO(raw),allow_pickle=False)
        if hidden.shape!=(4096,) or hidden.dtype!=np.float32 or not np.isfinite(hidden).all():raise ValueError('Target must be an actual finite 4096 hidden vector')
        if not isinstance(record.get('label_source'),str) or not record['label_source'].strip():raise ValueError('Provide explicit teaching provenance')
        rows.append(features);targets.append(.05*np.tanh(hidden))
    torch=host.torch;before={k:v.clone() for k,v in head.state().items()};optimizer=torch.optim.Adam(head.module.parameters(),lr=.0002)
    x=torch.as_tensor(np.stack(rows),device=next(head.module.parameters()).device);y=torch.as_tensor(np.stack(targets),device=x.device)
    progress={'status':'training','component':'body_numeric_embedding','total_steps':steps,'step':0,'method':'paired_hidden_regression','target_interpretation':'.05*tanh(recorded actual LLM hidden); correspondence teaching, not natural receptor ground truth'}
    head.module.requires_grad_(True)
    try:
        for step in range(steps):
            if (output/'STOP').exists():progress['status']='interrupted';break
            optimizer.zero_grad(set_to_none=True);prediction=head.forward(x);loss=torch.nn.functional.mse_loss(prediction,y)
            loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(head.module.parameters(),1.0))
            if not np.isfinite(grad):raise RuntimeError('Body embedding gradient was nonfinite')
            optimizer.step();head.steps+=1
            progress.update(step=step+1,loss=float(loss.detach()),grad_norm=grad,updated_at=time.time())
            write('progress.json',progress)
            with (output/'events.jsonl').open('a') as f:f.write(json.dumps(progress)+'\n')
        head.update_sha();saved=head.save(output/'body-embedding.safetensors');head.save();expected=head.weights_sha;head.load(saved['path'])
        if head.weights_sha!=expected:raise RuntimeError('Body embedding reload changed weights')
        after=head.state();changed=sum(int(torch.count_nonzero(after[k]!=v)) for k,v in before.items());delta=sum(float((after[k]-v).square().sum()) for k,v in before.items())**.5
        progress.update(status='interrupted' if progress['status']=='interrupted' else 'completed',changed_parameters=changed,parameter_l2_change=delta,checkpoint=saved,reload_verified=True,**head.identity())
        write('progress.json',progress);write('evidence.json',progress)
        return progress
    except Exception as exc:
        head.update_sha();progress.update(status='failed',error=str(exc));write('progress.json',progress);raise
    finally:head.module.requires_grad_(False).eval()

def body_embedding_identity(host,descriptor):
    """Inspect the exact scoped head without inserting another input token."""
    if descriptor is None:return None
    head,_,_=_head(host,descriptor,next(host.llm.parameters()).device)
    return head.identity()
