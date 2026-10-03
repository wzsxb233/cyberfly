"""Same-generation MiniCPM hidden states -> original TTS -> original Token2wav.

This is a bounded, non-streaming native output path. It never re-prompts an LLM
with the decoded string. The original token-aligned generation tensors are saved
with the projector input, native audio codes and final waveform for inspection.
"""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
REVISION='503e754207c94da6bb26850b4469f367c9ea3582'


def digest(raw):return hashlib.sha256(raw).hexdigest()


def tensor_info(array):
    import numpy as np
    if not np.isfinite(array).all():raise RuntimeError('Nonfinite native module output')
    return {'shape':list(array.shape),'dtype':str(array.dtype),'sha256':digest(array.tobytes()),
        'min':float(array.min()) if array.size else None,'max':float(array.max()) if array.size else None,
        'l2':float(np.linalg.norm(array.astype(np.float64))) if array.size else 0.0}


def write_tensors(arrays, event_id, *, directory, identity):
    import numpy as np
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    path=directory/(digest(event_id.encode())[:16]+'-'+uuid.uuid4().hex[:12]+'.npz')
    details={name:tensor_info(value) for name,value in arrays.items()}
    with path.open('xb') as f:np.savez(f,**arrays)
    return {'schema':1,'event_id':event_id,'path':str(path),'sha256':digest(path.read_bytes()),
        'arrays':details,**identity}


class NativeSpeech:
    def __init__(self, host):
        self.host=host
        if not hasattr(host.model,'tts'):raise RuntimeError('Original TTS weights were not loaded; restart with init_tts enabled')
        manifest_path=ROOT/'model_training/native-speech-manifest.json'
        manifest=json.loads(manifest_path.read_text())
        if manifest.get('revision')!=REVISION or manifest.get('repo')!='openbmb/MiniCPM-o-4_5':
            raise ValueError('Unexpected native speech model identity')
        for entry in manifest['files']:
            path=ROOT/'model_training/models'/entry['path']
            raw=path.read_bytes()
            if len(raw)!=entry['size']:raise ValueError('Native speech asset size mismatch')
            if entry.get('lfs'):
                valid=digest(raw)==entry['lfs']['oid']
            else:
                valid=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==entry['oid']
            if not valid:raise ValueError('Native speech asset hash mismatch: '+entry['path'])
        self.assets_sha=digest(manifest_path.read_bytes())
        # Pinned and verified local asset path avoids the upstream implicit
        # unpinned download. FP32 flow avoids an upstream mixed-dtype path.
        from stepaudio2 import Token2wav
        self.token2wav=Token2wav(str(ROOT/'model_training/models/assets/token2wav'),float16=False,n_timesteps=10)
        self.host.model.tts.audio_tokenizer=self.token2wav
        self.reference=Path('/root/.cache/cyberfly/minicpm-runtime/llama.cpp-omni/tools/omni/assets/default_ref_audio/default_ref_audio.wav')
        if not self.reference.is_file():raise RuntimeError('Missing bundled official default voice reference WAV')
        self.reference_sha=digest(self.reference.read_bytes())

    def generate(self, fused, req):
        """One LLM generation; its actual last-layer tensors feed original TTS."""
        import numpy as np
        import soundfile as sf
        torch=self.host.torch;tok=self.host.processor.tokenizer;tts=self.host.model.tts
        started=time.monotonic();event_id=req['event_id']
        bos=tok.convert_tokens_to_ids('<|tts_bos|>')
        eos=tok.convert_tokens_to_ids('<|tts_eos|>')
        im_end=tok.convert_tokens_to_ids('<|im_end|>')
        # The speech template's final marker follows the assistant prefix. The
        # sensory and brain/body soft embeddings are preserved in this prefix.
        bos_emb=self.host.llm.get_input_embeddings()(torch.tensor([[bos]],device='cuda'))
        prefix=torch.cat([fused,bos_emb.to(fused.dtype)],dim=1)
        max_tokens=req.get('max_speech_tokens',750)
        if type(max_tokens) is not int or not 50<=max_tokens<=1500:raise ValueError('max_speech_tokens must be 50..1500')
        text_budget=req.get('max_new_tokens',128)
        if type(text_budget) is not int or not 1<=text_budget<=256:raise ValueError('Speech max_new_tokens must be 1..256')
        with torch.inference_mode():
            output=self.host.llm.generate(inputs_embeds=prefix,
                attention_mask=torch.ones(prefix.shape[:2],device='cuda',dtype=torch.long),
                do_sample=False,max_new_tokens=text_budget,use_cache=True,output_hidden_states=True,return_dict_in_generate=True,
                eos_token_id=[eos,im_end],pad_token_id=tok.pad_token_id)
            ids=output.sequences[0]
            # HF generate's first hidden block spans the prefix; subsequent
            # blocks span the previous generated token. No teacher forcing or
            # second model call is used to obtain the speech semantic tensors.
            aligned=torch.cat([step[-1][0] for step in output.hidden_states],dim=0)
            end=next((i for i,value in enumerate(ids.tolist()) if value in (eos,im_end)),len(ids))
            if end==0 or prefix.shape[1]+end>aligned.shape[0]:
                raise RuntimeError('Speech generation missing terminal token or token-aligned hidden states; no text-only fallback')
            text_ids=ids[:end]
            semantic_hidden=aligned[prefix.shape[1]:prefix.shape[1]+end]
            if semantic_hidden.shape!=(end,4096):raise RuntimeError('Invalid actual generation hidden shape')
            say=tok.decode(text_ids,skip_special_tokens=True).strip()
            if not say:raise RuntimeError('Original model produced no spoken text')
            semantic=tts.projector_semantic(semantic_hidden.to(tts.emb_text.weight.dtype))
            if tts.config.normalize_projected_hidden:semantic=torch.nn.functional.normalize(semantic,p=2,dim=-1)
            merged=tts.emb_text(text_ids)+semantic
            tail_ids=torch.tensor([tts.config.text_eos_token_id,tts.audio_bos_token_id],device='cuda')
            tts_inputs=torch.cat([merged,tts.emb_text(tail_ids)],dim=0).unsqueeze(0)
            from model_training.models.utils import TTSSamplingParams
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(17)
                speech=tts.generate(inputs_embeds=tts_inputs,
                    eos_token=torch.tensor([tts.config.num_audio_tokens-1],device='cuda'),
                    max_new_token=max_tokens,show_tqdm=False,sampling_params=TTSSamplingParams())
                codes=speech.new_ids.squeeze(0).squeeze(-1)
                if codes.numel()==0 or not torch.isfinite(semantic_hidden).all():raise RuntimeError('Invalid original TTS output')
                wav_bytes=self.token2wav(codes.tolist(),str(self.reference))
            arrays={'text_token_ids':text_ids.detach().cpu().numpy().astype(np.int64),
                'generation_hidden':semantic_hidden.detach().float().cpu().numpy(),
                'projected_hidden':semantic.detach().float().cpu().numpy(),
                'tts_inputs':tts_inputs.detach().float().cpu().numpy(),
                'audio_codes':codes.detach().cpu().numpy().astype(np.int32)}
        waveform,rate=sf.read(io.BytesIO(wav_bytes),dtype='float32')
        if rate!=24000 or waveform.ndim!=1 or len(waveform)<2400 or not np.isfinite(waveform).all() or float(np.abs(waveform).max())<1e-6:
            raise RuntimeError('Original Token2wav produced invalid/silent waveform')
        ident={'model_revision':REVISION,'weights_manifest_sha256':self.host.weights_manifest_sha,
            'speech_assets_manifest_sha256':self.assets_sha,**self.host.language_identity()}
        source=write_tensors(arrays,event_id,directory=ROOT/'artifacts/native_speech',identity=ident)
        path=Path(source['path']).with_suffix('.wav');path.write_bytes(wav_bytes)
        audio={'schema':1,'event_id':event_id,'path':str(path),'sha256':digest(wav_bytes),'sample_rate':rate,
            'samples':len(waveform),'duration_seconds':len(waveform)/rate,'peak':float(np.abs(waveform).max()),
            'rms':float(np.sqrt(np.mean(waveform**2))),**ident,
            'source':'same brain-conditioned LLM generation hidden -> original MiniCPM TTS -> original StepAudio2 flow/HiFT',
            'same_generation_hidden':True,'source_hidden_sha256':source['arrays']['generation_hidden']['sha256'],
            'default_voice_reference_sha256':self.reference_sha,'native_tensors_file':source,
            'audio_generation_finished':bool(speech.finished),'llm_generated_tokens':end,'elapsed_seconds':time.monotonic()-started,
            'generation_alignment':{'prefix_tokens':int(prefix.shape[1]),'hidden_rows':int(aligned.shape[0]),
                'generated_sequence_tokens':int(ids.numel()),'terminal_token_id':int(ids[end]),
                'speech_hidden_start':int(prefix.shape[1]),'speech_hidden_rows':end},
            'streaming':False}
        Path(source['path']).with_suffix('.json').write_text(json.dumps({'say':say,**audio},ensure_ascii=False,indent=2,allow_nan=False))
        return say,{'audio_file':audio,'native_speech':{'status':'actual_same_generation_audio','streaming':False,'same_generation_hidden':True}}
