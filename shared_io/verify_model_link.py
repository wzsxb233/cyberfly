"""Run real local multimodal numerical coupling checks; save every actual result."""
import json
from pathlib import Path
import time
import urllib.request
import urllib.error
import wave
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'artifacts/lab-integration/shared-hidden-01'

def call(operation, data):
    req = urllib.request.Request('http://127.0.0.1:18647/' + operation,
        data=json.dumps(data, ensure_ascii=False, allow_nan=False).encode(), headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=600) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(exc.read().decode()) from None

def save(name, value):
    (OUTPUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))

def main():
    request = json.loads((OUTPUT/'request.json').read_text())
    record = json.loads((ROOT/'artifacts/full_brain/minicpm-state-source.json').read_text())
    request['facts'] = {'brain': record['neural'], 'interpretation':'真实全脑聚合记录；不等于主观思想。'}
    before = json.loads((OUTPUT/'encode-before.json').read_text())
    decoded = json.loads((OUTPUT/'decode-before.json').read_text())
    assert len(before['hidden']) == 4096 and len(before['currents_mv']) == 47
    assert before['image_token_count'] > 0
    assert decoded['brain_ablation']['logit_delta_l2'] > 0
    print('actual text+image numerical IO passed', flush=True)
    # The first run finished both optimizer steps, then NTFS mmap rename failed.
    # Recover saved live parameters and verify restart identity instead of silently
    # repeating training or overwriting the original failed record.
    original = json.loads((ROOT/'artifacts/model-training/brain-projector-smoke-01/progress.json').read_text())
    with urllib.request.urlopen('http://127.0.0.1:18647/health',timeout=60) as r:
        health=json.load(r)
    checkpoint = call('save', {})
    assert health['coupler_sha256'] == original['coupler_sha256']
    assert health['coupler_training_steps'] == 2 and checkpoint['reload_verified']
    trained = {**original,'status':'recovered','original_status':original['status'],
        'recovery':'Actual trained live tensors saved on Linux, service restarted from that exact embedded-identity checkpoint; original NTFS failure retained.',
        'reload_verified':True,'checkpoint':checkpoint}
    save('training-recovery.json', trained)
    assert trained['step'] == 2 and trained['adapter_delta_l2'] > 0
    assert trained['brain_to_model_grad_norm'] > 0 and trained['model_to_brain_grad_norm'] > 0
    request['event_id'] = 'shared-hidden-after-train'
    after = call('encode', request); save('encode-after.json', after)
    decoded_after = call('decode', request); save('decode-after.json', decoded_after)
    assert after['coupler_sha256'] != before['coupler_sha256']
    print('actual 2-step bidirectional projector gradients + reload passed', flush=True)
    # Actual native MiniCPM-generated audio, resampled once for its official 16k input.
    source = ROOT/'model_runtime/audio_verification/duplex_question.wav'
    with wave.open(str(source),'rb') as wav:
        sample_rate = wav.getframerate()
        samples = np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2').astype(np.float32)
    resampled = np.interp(np.arange(int(len(samples)*16000/sample_rate))*sample_rate/16000,np.arange(len(samples)),samples).astype('<i2')
    audio_path = OUTPUT/'shared-input-16k.wav'
    with wave.open(str(audio_path),'wb') as wav:
        wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(16000);wav.writeframes(resampled.tobytes())
    request.update(event_id='shared-raw-audio-image-text',audio_path=str(audio_path),text='请用中文回答录音中的问题，并简单说一下画面中昆虫眼睛的颜色。')
    audio_encoded = call('encode', request); save('encode-all-modalities.json',audio_encoded)
    audio_decoded = call('decode', request); save('decode-all-modalities.json',audio_decoded)
    assert audio_encoded['audio_token_count'] > 0 and audio_encoded['image_token_count'] > 0
    assert audio_encoded['modalities_encoded'] == ['text','image','audio']
    save('summary.json', {'status':'verified','original_model_revision':after['model_revision'],
        'raw_modalities_encoded':audio_encoded['modalities_encoded'], 'actual_hidden_dim':4096,
        'brain_features_dim':240,'soft_token_shape':decoded_after['brain_soft_shape'],
        'actual_current_dim':47,'training_steps':trained['step'],'reload_verified':True,
        'brain_ablation_logit_l2_before':decoded['brain_ablation']['logit_delta_l2'],
        'brain_ablation_logit_l2_after':decoded_after['brain_ablation']['logit_delta_l2'],
        'brain_step_executed_by_this_test':False,
        'limitations':'This test consumes previously recorded real brain measurements. Root validates the live full-brain/body loop separately. Two supervised optimization steps do not establish thought decoding or task competence.',
        'audio_source':str(source),'audio_resampling':'actual 24k PCM16 waveform linearly resampled to 16k PCM16'})
    print('actual raw audio+image+text numerical IO passed',flush=True)

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        save('failure.json', {'status':'failed','error':str(exc),'timestamp':time.time()})
        raise
