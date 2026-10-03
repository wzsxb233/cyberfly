"""Strict offline export of existing grounded language LoRA, never deployment."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import uuid

from .latent_contract import ROOT,REVISION,OMNI_COMMIT,sha,write_json

SOURCE_NAME=re.compile(r'^model\.layers\.(\d+)\.self_attn\.(q_proj|v_proj)\.lora_([AB])\.weight$')
PREFIX='thinker.llm.'


def _file(descriptor,*,limit=32*1024*1024):
    if not isinstance(descriptor,dict):raise ValueError('An explicit path/SHA descriptor is required')
    path=Path(descriptor.get('path','')).resolve()
    roots=[ROOT/'artifacts',ROOT/'shared_io/checkpoints',Path('/root/.cache/cyberfly/candidate-language-gpu1')]
    if not any(path.is_relative_to(r.resolve()) for r in roots) or not path.is_file() or not 0<path.stat().st_size<=limit:
        raise ValueError('Source must be a bounded local training artifact')
    raw=path.read_bytes()
    if sha(raw)!=descriptor.get('sha256'):raise ValueError('Source file SHA mismatch')
    return path,raw


def _shapes():
    config=json.loads((ROOT/'model_training/models/config.json').read_text())
    shape={k:config[k] for k in ('num_hidden_layers','hidden_size','num_attention_heads','num_key_value_heads','head_dim')}
    if shape!={'num_hidden_layers':36,'hidden_size':4096,'num_attention_heads':32,'num_key_value_heads':8,'head_dim':128}:
        raise ValueError('This export is fixed to the verified original MiniCPM-o-4.5 language architecture')
    return shape


def _expected_names():
    return {f'model.layers.{layer}.self_attn.{projection}.lora_{side}.weight':
        ((4,4096) if side=='A' else (4096 if projection=='q_proj' else 1024,4))
        for layer in range(36) for projection in ('q_proj','v_proj') for side in ('A','B')}


def read_source(checkpoint,training_evidence):
    import torch
    from safetensors.torch import load
    from shared_io.language_adapter import state_sha
    path,raw=_file(checkpoint);header_len=int.from_bytes(raw[:8],'little')
    if not 0<header_len<1024*1024:raise ValueError('Invalid safetensors header bound')
    metadata=json.loads(json.loads(raw[8:8+header_len])['__metadata__']['identity'])
    manifest=json.loads((ROOT/'model_training/weights-manifest.json').read_text())
    identity={'schema':1,'model_revision':REVISION,'weights_manifest_sha256':sha((ROOT/'model_training/weights-manifest.json').read_bytes()),
        'rank':4,'target_modules':['q_proj','v_proj'],'adapter_name':'grounded'}
    if manifest['revision']!=REVISION or any(metadata.get(k)!=v for k,v in identity.items()):raise ValueError('Original model or bounded adapter identity mismatch')
    if type(metadata.get('training_steps'))is not int or metadata['training_steps']<1:raise ValueError('An actually trained source checkpoint is required')
    _shapes();state=load(raw);expected=_expected_names()
    if set(state)!=set(expected) or len(state)!=144:raise ValueError('Expected exactly 144 original-language q/v LoRA tensors')
    if any(tuple(value.shape)!=expected[name] or value.dtype!=torch.float32 or not torch.isfinite(value).all() for name,value in state.items()):
        raise ValueError('Unexpected language LoRA shape, dtype or nonfinite value')
    if state_sha(state)!=metadata.get('adapter_sha256'):raise ValueError('Source adapter content hash differs from its embedded identity')
    evidence_path,evidence_raw=_file(training_evidence);evidence=json.loads(evidence_raw)
    proof=evidence.get('checkpoint',{})
    if (evidence.get('status')!='completed' or evidence.get('component')!='grounded_language_qlora' or
            evidence.get('reload_verified') is not True or proof.get('sha256')!=checkpoint['sha256'] or
            Path(proof.get('path','')).resolve()!=path or evidence.get('original_trainable_parameters')!=0 or
            evidence.get('facts_in_model_input') is not False):
        raise ValueError('Actual completed latent-grounded training evidence does not match this checkpoint')
    return state,metadata,{'checkpoint':{'path':str(path),'sha256':sha(raw)},
        'training_evidence':{'path':str(evidence_path),'sha256':sha(evidence_raw)}}


def export_adapter(checkpoint,training_evidence,output,*,lora_alpha):
    """Alpha is explicit because the legacy checkpoint did not embed it."""
    import torch
    from safetensors.torch import save_file,load
    from shared_io.language_adapter import state_sha
    if type(lora_alpha)is not int or lora_alpha!=8:raise ValueError('Original grounded rank-4 training used explicit alpha=8')
    state,metadata,source=read_source(checkpoint,training_evidence)
    output=Path(output).resolve()
    if not output.is_relative_to((ROOT/'artifacts').resolve()) or output.exists():raise ValueError('Use a new artifact directory; never overwrite active adapters')
    output.parent.mkdir(parents=True,exist_ok=True);stage=output.with_name(output.name+'.'+uuid.uuid4().hex+'.tmp');stage.mkdir()
    mapped={PREFIX+name:value.clone() for name,value in state.items()}
    packing=[]
    for layer in range(36):
        base=f'thinker.llm.model.layers.{layer}.self_attn.'
        packing.append({'packed_module':base+'qkv_proj','submodules':[base+'q_proj',None,base+'v_proj'],
            'q_rows':[0,4096],'k_rows':[4096,5120],'v_rows':[5120,6144],
            'missing_k_means_no_adapter':True,'independent_q_and_v_rank':4})
    identity={'schema':1,'kind':'offline_original_language_lora_export','source':source,'source_adapter_identity':metadata,
        'model_revision':REVISION,'weights_manifest_sha256':metadata['weights_manifest_sha256'],'vllm_omni_commit':OMNI_COMMIT,
        'target_runtime_prefix':PREFIX,'source_adapter_sha256':metadata['adapter_sha256'],
        'exported_tensor_content_sha256':state_sha(mapped),'tensor_count':144,'rank':4,'lora_alpha':8,'lora_scaling':2,
        'scaling_provenance':{'path':str(ROOT/'shared_io/language_adapter.py'),'sha256':sha((ROOT/'shared_io/language_adapter.py').read_bytes()),
            'rule':'Original LanguageAdapter.ensure: lora_alpha=rank*2; explicit exporter argument required'},
        'original_weights_included':False,'deployed':False,'runtime_lora_verified':False,
        'quality_policy':'Candidate only; source held-out quality must be inspected, export is not approval of language grounding'}
    config={'base_model_name_or_path':str(ROOT/'model_training/models'),'peft_type':'LORA','task_type':'CAUSAL_LM',
        'r':4,'lora_alpha':8,'lora_dropout':0.0,'bias':'none','target_modules':['q_proj','v_proj'],
        'modules_to_save':None,'inference_mode':True,'use_dora':False,'use_rslora':False}
    save_file(mapped,str(stage/'adapter_model.safetensors'),metadata={'identity':json.dumps(identity,sort_keys=True)})
    write_json(stage/'adapter_config.json',config);write_json(stage/'packing-plan.json',packing)
    reloaded=load((stage/'adapter_model.safetensors').read_bytes())
    recovered={name[len(PREFIX):]:value for name,value in reloaded.items()}
    if state_sha(recovered)!=metadata['adapter_sha256'] or any(not torch.equal(recovered[k],v) for k,v in state.items()):
        raise RuntimeError('Export name mapping changed an original trained tensor')
    identity['files']={name:sha((stage/name).read_bytes()) for name in ('adapter_model.safetensors','adapter_config.json','packing-plan.json')}
    identity['reload_verified']=True;write_json(stage/'export-evidence.json',identity);stage.replace(output)
    return {'path':str(output),'sha256':sha((output/'export-evidence.json').read_bytes()),**identity}


def validate_export(output):
    import torch
    from safetensors.torch import load
    from shared_io.language_adapter import state_sha
    directory=Path(output).resolve()
    if not directory.is_relative_to((ROOT/'artifacts').resolve()):raise ValueError('Export must be a workspace artifact')
    evidence=json.loads((directory/'export-evidence.json').read_text())
    for name in ('adapter_model.safetensors','adapter_config.json','packing-plan.json'):
        if sha((directory/name).read_bytes())!=evidence['files'][name]:raise ValueError('Exported file identity changed: '+name)
    original,metadata,_=read_source(evidence['source']['checkpoint'],evidence['source']['training_evidence'])
    mapped=load((directory/'adapter_model.safetensors').read_bytes())
    if set(mapped)!={PREFIX+k for k in original} or any(not torch.equal(mapped[PREFIX+k],v) for k,v in original.items()):
        raise ValueError('Exported tensors differ from the actual trained source')
    if state_sha(mapped)!=evidence['exported_tensor_content_sha256']:raise ValueError('Exported content hash mismatch')
    config=json.loads((directory/'adapter_config.json').read_text())
    if any(config.get(k)!=v for k,v in {'r':4,'lora_alpha':8,'target_modules':['q_proj','v_proj'],'bias':'none','modules_to_save':None}.items()):
        raise ValueError('Exported adapter configuration changed')
    return {'status':'offline_export_verified','tensor_count':len(mapped),'source_adapter_sha256':metadata['adapter_sha256'],
        'mapped_tensors_bitexact':True,'original_weights_included':False,'deployed':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--training-evidence',required=True);p.add_argument('--evidence-sha256',required=True)
    p.add_argument('--alpha',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=export_adapter({'path':a.checkpoint,'sha256':a.sha256},{'path':a.training_evidence,'sha256':a.evidence_sha256},a.output,lora_alpha=a.alpha)
    result['validation']=validate_export(a.output);print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
