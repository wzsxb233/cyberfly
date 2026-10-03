"""Actual existing MiniCPM service: per-neuron IO, perturbation, training/reload."""
from __future__ import annotations
import json
from pathlib import Path
import time
import urllib.request
import urllib.error
import uuid
import numpy as np
from shared_io.neuron_link import NeuronStateCodec,IDS_SHA,N
from connectome_adapter.neuron_currents import load_neuron_currents
ROOT=Path(__file__).resolve().parents[2]
URL='http://127.0.0.1:18647'

def call(operation,payload=None):
    request=urllib.request.Request(URL+'/'+operation,
        data=json.dumps(payload,ensure_ascii=False,allow_nan=False).encode() if payload is not None else None,
        headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(request,timeout=600) as response:return json.load(response)
    except urllib.error.HTTPError as exc:raise RuntimeError(exc.read().decode()) from None


def main():
    output=ROOT/'artifacts/neuron-link-validation'/('live-'+uuid.uuid4().hex[:12]);output.mkdir(parents=True)
    def save(name,data):
        (output/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False))
    status={'status':'running','started_at':time.time(),'original_frozen_model':'MiniCPM-o-4.5','external_brain_step_executed':False}
    save('summary.json',status)
    try:
        health=call('health');save('health-before.json',health)
        assert 'neuron_direct' in health.get('supported_connection_modes',[])
        original_grouped_sha=health['coupler_sha256']
        descriptor=json.loads((ROOT/'artifacts/neuron-link-validation/source-state.json').read_text())
        codec=NeuronStateCodec();features,state=codec.read(descriptor)
        request={'connection_mode':'neuron_direct','io_scope':'sensory_motor','event_id':'native-before-'+uuid.uuid4().hex,
            'text':'看看共同画面，用中文简短描述昆虫眼睛的颜色。不要声称读到了主观想法。',
            'image_path':str(ROOT/'runs/ppo_3d/fly_3d.png'),'brain_state':descriptor}
        before=call('encode',request);save('encode-before.json',before)
        currents=load_neuron_currents(before['currents_file'],codec.ids,IDS_SHA)
        assert before['projected_read_neurons']==2129 and before['addressable_input_neurons']==17336
        assert np.all(currents[~codec.write_mask].view(np.uint32)==0)
        assert np.unique(currents[codec.write_mask]).size>17000 and np.max(currents)>10
        save('sensory-current-statistics.json',{'independent_unique_values':int(np.unique(currents[codec.write_mask]).size),
            'active_min_mv':float(currents[codec.write_mask].min()),'active_max_mv':float(currents.max()),
            'active_mean_mv':float(currents[codec.write_mask].mean()),'active_std_mv':float(currents[codec.write_mask].std()),
            'outside_scope_canonical_zero':True})
        decoded=call('decode',request);save('decode-before.json',decoded)
        neuron_id=int(codec.ids[np.flatnonzero(codec.read_mask)[0]])
        probe=call('probe_neuron',{**request,'neuron_id':neuron_id,'delta_mv':30.0});save('single-neuron-probe.json',probe)
        assert probe['perturbation']['changed_features']==1 and probe['soft_delta_l2']>0 and probe['hidden_delta_l2']>0 and probe['logit_delta_l2']>0
        print('native per-ID model IO + actual one-cell numeric ablation passed',flush=True)
        zero=json.loads((ROOT/'artifacts/neuron-link-validation/actual-zero-current-target.json').read_text())
        record={**request,'event_id':'native-supervision-'+uuid.uuid4().hex,
            'text':'根据脑状态数值简短解释这份真实记录，并说明能否读到主观想法。',
            'target_text':'这份记录覆盖166700个神经元，在1毫秒内记录到951次放电。这说明网络存在电活动，不能据此认定读到了主观想法。',
            'target_currents_file':zero,'source_scope':'Identical actual saved ids/v/counts from brain-agent final 1ms step. Its validation asserts zero additional general currents; no simulator state invented.'}
        train_request={'connection_mode':'neuron_direct','io_scope':'sensory_motor','steps':2,'records':[record],
            'output':str(output/'training')}
        trained=call('train_coupler',train_request);save('training-result.json',trained)
        assert trained['status']=='completed' and trained['step']==2 and trained['adapter_delta_l2']>0
        assert trained['brain_to_model_grad_norm']>0 and trained['model_to_brain_grad_norm']>0 and trained['reload_verified']
        request['event_id']='native-after-'+uuid.uuid4().hex
        after=call('encode',request);save('encode-after.json',after)
        after_decode=call('decode',request);save('decode-after.json',after_decode)
        assert after['coupler_sha256']==trained['coupler_sha256']!=before['coupler_sha256']
        print('native bidirectional gradients + saved tensors reload + real generation passed',flush=True)
        all_request={**request,'connection_mode':'neuron_direct','io_scope':'all_neurons','event_id':'native-all-'+uuid.uuid4().hex}
        full=call('encode',all_request);save('encode-all-neurons.json',full)
        all_current=load_neuron_currents(full['currents_file'],codec.ids,IDS_SHA)
        assert full['projected_read_neurons']==N and full['addressable_input_neurons']==N
        assert np.unique(all_current).size>N*.95 and np.count_nonzero(all_current)>N*.99
        saved=call('save',{});save('save-all-modes.json',saved)
        assert saved['reload_verified'] and set(saved['saved_connection_modes'])=={'grouped','neuron_direct/sensory_motor','neuron_direct/all_neurons'}
        final=call('health');save('health-after.json',final)
        assert final['coupler_sha256']==original_grouped_sha
        assert final['connection_modes']['neuron_direct/sensory_motor']['coupler_sha256']==after['coupler_sha256']
        status.update(status='verified',completed_at=time.time(),source_state=descriptor,
            sensory_unique_currents=int(np.unique(currents[codec.write_mask]).size),all_unique_currents=int(np.unique(all_current).size),
            sensory_write_neurons=17336,sensory_read_neurons=2129,all_neurons=N,rank=16,
            single_neuron_soft_delta_l2=probe['soft_delta_l2'],single_neuron_logit_delta_l2=probe['logit_delta_l2'],
            real_optimizer_steps=2,trained_native_sha=after['coupler_sha256'],grouped_sha_unchanged=original_grouped_sha,
            checkpoint_reload_verified=True,saved_connection_modes=list(saved['saved_connection_modes']),
            interpretation='Actual original-model hidden/soft-token computation and converter optimization. Single-cell probe is explicitly counterfactual, and this probe does not advance or differentiate the external brain. Root validates live brain/body execution separately.')
        save('summary.json',status);print(str(output),flush=True)
    except Exception as exc:
        status.update(status='failed',error=str(exc),failed_at=time.time());save('summary.json',status);raise

if __name__=='__main__':main()
