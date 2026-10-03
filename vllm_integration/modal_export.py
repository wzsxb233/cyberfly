"""Bounded capture of original merged perception rows, without altering them.

This helper is not active until explicitly connected to the model worker. The
runner's actual multimodal mask is required; printed placeholder text alone is
never accepted as evidence that a vision/audio encoder ran.
"""
from __future__ import annotations
from collections import OrderedDict
import queue
import threading
import time

from .latent_contract import ROOT,sha,write_json

STARTS={151669:('vision',151670),151679:('vision',151680),151697:('audio',151699)}
ENDS={151670,151680,151699}


def modal_token_positions(token_ids,actual_mm_mask,*,active=None):
    """Classify only actual encoder rows using original special-token bounds."""
    if len(token_ids)!=len(actual_mm_mask) or any(type(x)is not int for x in token_ids) or any(type(x)is not bool for x in actual_mm_mask):
        raise ValueError('Actual scheduled token IDs and boolean multimodal mask must align')
    indices={'vision':[],'audio':[]}
    for index,(token,selected) in enumerate(zip(token_ids,actual_mm_mask)):
        if token in STARTS:
            if active is not None:raise ValueError('Nested original modality bounds cannot be attributed safely')
            active=STARTS[token]
        elif token in ENDS:
            if active is None or token!=active[1]:raise ValueError('Original modality boundary mismatch')
            active=None
        if selected:
            if token!=128244 or active is None:
                raise ValueError('Actual multimodal row has no recognized original encoder boundary')
            indices[active[0]].append(index)
        elif token==128244 and active is not None:
            raise ValueError('A printed placeholder was not filled by the actual multimodal encoder')
    return indices,active


class ScheduledModalCapture:
    """Retain at most eight request prefixes, respecting real chunk offsets."""
    def __init__(self):self.pending=OrderedDict()

    def consume(self,request_id,offset,token_ids,actual_mm_mask,embeddings,*,complete):
        import torch
        if embeddings.ndim!=2 or embeddings.shape!=(len(token_ids),4096):raise ValueError('Actual scheduled embedding shape mismatch')
        if type(offset)is not int or offset<0:raise ValueError('Invalid actual scheduler offset')
        if request_id not in self.pending:
            if offset!=0:raise ValueError('Original modality capture must start at the real prompt beginning')
            self.pending[request_id]={'offset':0,'active':None,'parts':{'vision':[],'audio':[]},'positions':{'vision':[],'audio':[]}}
            if len(self.pending)>8:self.pending.popitem(last=False)
        entry=self.pending[request_id]
        if offset!=entry['offset']:raise ValueError('Original modality prefill has a gap or was replayed')
        indices,active=modal_token_positions(token_ids,actual_mm_mask,active=entry['active'])
        for name,positions in indices.items():
            if positions:
                chosen=embeddings.index_select(0,torch.tensor(positions,device=embeddings.device))
                entry['parts'][name].append(chosen.detach().float().cpu().clone())
                entry['positions'][name].extend(offset+i for i in positions)
        entry.update(offset=offset+len(token_ids),active=active)
        if sum(len(v) for v in entry['positions'].values())>4096:raise ValueError('Bounded original modality token budget exceeded')
        if not complete:return None
        if active is not None:raise ValueError('Original prompt ended inside a modality placeholder')
        self.pending.pop(request_id)
        arrays={name:torch.cat(parts,dim=0).numpy() if parts else torch.empty((0,4096),dtype=torch.float32).numpy()
            for name,parts in entry['parts'].items()}
        return {'arrays':arrays,'actual_input_positions':entry['positions'],
            'actual_multimodal_mask_verified':True,'prefill_tokens':entry['offset']}


class ModalTokenExporter:
    """One immutable file per completed original prefill; bounded I/O queue."""
    def __init__(self):
        self.queue=queue.Queue(maxsize=2)
        self.thread=threading.Thread(target=self._run,name='cyberfly-vllm-perception',daemon=True);self.thread.start()

    def submit(self,hidden,condition,step,captured):
        # Exactly the same actual prefill hidden accompanies its perception
        # rows. It is not a hidden from a later request or regenerated text.
        item=(hidden.detach().float().cpu().clone(),dict(condition),dict(step),captured)
        self.queue.put_nowait(item)

    def _run(self):
        from shared_io.native_speech import write_tensors
        while True:
            item=self.queue.get()
            if item is None:return
            hidden,condition,step,captured=item
            event_id=condition['event_id'];request_id=step['request_id']
            folder=ROOT/'artifacts/vllm_latent'/sha(event_id.encode())[:24]/sha(request_id.encode())[:24]
            try:
                arrays={**captured['arrays'],'language':hidden.numpy().reshape(1,4096)}
                identity={k:condition[k] for k in ('model_revision','weights_manifest_sha256','vllm_omni_commit')}
                descriptor=write_tensors(arrays,event_id,directory=ROOT/'artifacts/shared_modal_embeddings',identity={
                    **identity,'request_id':request_id,'prefill_sequence':step['sequence'],'published_at':time.time(),
                    'source':'actual vLLM original resampler and APM/projection/pool rows merged into thinker inputs_embeds',
                    'source_runtime':'vllm_omni','runtime':step.get('runtime'),
                    'actual_input_positions':captured['actual_input_positions'],
                    'actual_multimodal_mask_verified':True,'condition_file_sha256':condition['sha256'],
                    'original_modality_embeddings_modified':False,'same_original_latent_returned_to_native_tts':True})
                write_json(folder/'modal-latest.json',descriptor);write_json(folder.parent/'modal-latest.json',descriptor)
            except Exception as exc:
                write_json(folder/'modal-error.json',{'event_id':event_id,'request_id':request_id,
                    'status':'original_modality_export_failed','error':str(exc)})


class ModalCaptureBridge:
    """Small explicit worker hooks, bounded to its enforced one-request batch.

record_mask belongs after the original embed_input_ids call. capture belongs
after request/condition validation, before its actual forward. publish receives
that same forward's original last hidden. None of these replace tensors.
"""
    def __init__(self):
        self.scheduled=None;self.capture_state=ScheduledModalCapture();self.exporter=ModalTokenExporter()

    def record_mask(self,input_ids,is_multimodal):
        ids=input_ids.detach().cpu().reshape(-1).tolist()
        mask=is_multimodal.detach().cpu().reshape(-1).tolist() if is_multimodal is not None else [False]*len(ids)
        if len(ids)!=len(mask):raise ValueError('Original encoder mask does not match scheduled tokens')
        self.scheduled=(ids,mask)

    def capture(self,ids,embeds,kwargs):
        scheduled,self.scheduled=self.scheduled,None
        if not kwargs.get('_omni_is_prefill'):return None
        actual_ids=ids.detach().cpu().reshape(-1).tolist()
        if scheduled is None or scheduled[0]!=actual_ids:
            raise ValueError('Original per-request prefill cannot be matched to its real scheduler multimodal mask')
        offset=kwargs['_omni_num_computed_tokens'];prompt_len=kwargs['_omni_prompt_len']
        if type(prompt_len)is not int or not 0<prompt_len<=4096:raise ValueError('Invalid bounded original prompt length')
        return self.capture_state.consume(kwargs['request_id'],offset,actual_ids,scheduled[1],embeds,
            complete=offset+len(actual_ids)==prompt_len)

    def publish(self,hidden,condition,step,captured):
        if captured is not None:self.exporter.submit(hidden,condition,step,captured)
