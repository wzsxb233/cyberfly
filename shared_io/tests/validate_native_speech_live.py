"""Real native TTS and original perception tensor export; no optimizer updates.

This reuses separately recorded real brain/body/media fixtures to validate the
model-side numerical/audio paths. It does not advance a brain or establish a
simultaneous physical event; the interactive bus performs that separate test.
"""
from pathlib import Path
import hashlib,json,time,uuid
import numpy as np
import soundfile as sf
from .validate_neuron_link_live import call
ROOT=Path(__file__).resolve().parents[2]
def sha(raw):return hashlib.sha256(raw).hexdigest()

def main():
    out=ROOT/'artifacts/native-speech-validation'/uuid.uuid4().hex[:12];out.mkdir(parents=True)
    def save(name,obj):(out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))
    state={'status':'running','started_at':time.time(),'optimizer_steps':0,
        'scope':'Original recorded fixtures validate model-side IO; external brain not stepped here.'}
    save('summary.json',state)
    try:
        bodyroot=ROOT/'artifacts/body_sense_checks/interface_sample'
        packet_path=bodyroot/'body_packet.json';packet=json.loads(packet_path.read_text())
        layout=json.loads((bodyroot/'feature_layout.json').read_text())
        path=bodyroot/'features.npz'
        with np.load(path,allow_pickle=False) as z:features=z['features']
        body={'schema':1,'path':str(path),'sha256':sha(path.read_bytes()),'shape':list(features.shape),
            'dtype':'float32','layout_sha256':layout['feature_layout_sha256'],
            'simulation_time_s':packet['simulation_time_s'],'packet_path':str(packet_path),
            'packet_sha256':packet['packet_sha256'],'packet_file_sha256':sha(packet_path.read_bytes())}
        brain=json.loads((ROOT/'artifacts/neuron-link-validation/source-state.json').read_text())
        req={'connection_mode':'neuron_direct','io_scope':'sensory_motor','event_id':'native-speech-'+uuid.uuid4().hex,
            'text':'请看共同画面，只用一句简短中文说出果蝇眼睛的颜色。',
            'image_path':str(ROOT/'runs/ppo_3d/fly_3d.png'),
            'audio_path':str(ROOT/'artifacts/shared_io_modal_ab/980268a5fd7d/native_a_16k_3s.wav'),
            'body_features_file':body,'brain_state':brain}
        save('request.json',req);save('health-before.json',call('health'))
        encoded=call('encode',req);save('encode.json',encoded)
        d=encoded['modal_embeddings_file'];raw=Path(d['path']).read_bytes()
        assert sha(raw)==d['sha256'] and d['event_id']==req['event_id']
        with np.load(d['path'],allow_pickle=False) as z:
            assert set(z.files)=={'vision','audio','language'}
            for key in z.files:
                a=z[key];assert a.dtype==np.float32 and a.ndim==2 and a.shape[1]==4096 and len(a)>0
                assert sha(a.tobytes())==d['arrays'][key]['sha256']
            assert np.array_equal(z['language'][0],np.array(encoded['hidden'],dtype=np.float32))
        assert encoded['body_input']['body_soft_shape']==[1,1,4096]
        print('actual original vision/audio tensors + body numeric soft token verified: '+str(out/'encode.json'),flush=True)
        decoded=call('decode',{**req,'generate_audio':True,'max_speech_tokens':500,'body_feedback_file':body})
        save('decode.json',decoded)
        audio=decoded['audio_file'];wav=Path(audio['path']).read_bytes()
        assert sha(wav)==audio['sha256'] and audio['same_generation_hidden'] is True
        source=audio['native_tensors_file'];assert sha(Path(source['path']).read_bytes())==source['sha256']
        with np.load(source['path'],allow_pickle=False) as z:
            assert z['generation_hidden'].shape==(len(z['text_token_ids']),4096)
            assert z['projected_hidden'].shape==(len(z['text_token_ids']),768)
            assert z['tts_inputs'].shape==(1,len(z['text_token_ids'])+2,768)
            assert sha(z['generation_hidden'].tobytes())==audio['source_hidden_sha256']
            assert len(z['audio_codes'])>0
        wave,rate=sf.read(audio['path']);assert rate==24000 and wave.size>2400 and np.isfinite(wave).all() and np.max(np.abs(wave))>1e-5
        assert decoded['body_feedback']['body_soft_shape']==[1,1,4096]
        assert decoded['coupler_sha256']==encoded['coupler_sha256']
        save('health-after.json',call('health'))
        state.update(status='verified',completed_at=time.time(),say=decoded['say'],audio_file=audio,
            modal_embeddings_file=d,body_input=encoded['body_input'],body_feedback=decoded['body_feedback'],
            coupler_sha256=encoded['coupler_sha256'],brain_ablation=decoded['brain_ablation'])
        save('summary.json',state);print(str(out),flush=True)
    except Exception as exc:
        state.update(status='failed',error=str(exc));save('summary.json',state);raise
if __name__=='__main__':main()
