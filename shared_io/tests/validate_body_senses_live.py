"""Actual body packet, optimizer, reload and preserved-input neural integration."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import uuid
import numpy as np
from PIL import Image
import torch

from scenarios import create_scenario
from shared_io.body_senses import BodySenseAdapter, array_sha
from shared_io.parallel_inputs import prepare_parallel_stimulation

ROOT=Path(__file__).resolve().parents[2]


def main():
    torch.set_num_threads(1)
    output=ROOT/'artifacts/body_sense_checks'/uuid.uuid4().hex[:12];output.mkdir(parents=True)
    env=None;created_files=[];evidence={'status':'incomplete','isolation':'Own MuJoCo/full-brain worker; no live lab or model service requests'}
    try:
        env=create_scenario({'name':'fly_neural_link','env_kwargs':{'max_episode_steps':100,'learning':False}})
        env.reset(seed=20260913)
        semantic={'retina_left':.2,'retina_right':.2,'sugar':.5}
        env.apply_task({'neural_drive':semantic})
        pinned=env.shared_input_rgb();packet=env.shared_body_senses();qpos=env.body.unwrapped.sim.mj_data.qpos.copy()
        adapter=BodySenseAdapter(packet)
        first=adapter.prepare(packet,output/'before');created_files.append(first['currents_file']['path'])
        assert packet['vision']['rgb_sha256']==array_sha(pinned)
        assert packet['vision']['time_s']==packet['simulation_time_s']
        with np.load(first['body_features_file']['path'],allow_pickle=False) as f:
            assert f.files==['features'] and np.array_equal(f['features'],adapter.codec.encode(packet))
        assert np.array_equal(qpos,env.body.unwrapped.sim.mj_data.qpos)
        assert max(abs(first['minimum_current_mv']),abs(first['maximum_current_mv']))<.001
        source_event=json.loads((ROOT/'artifacts/shared_io/b40f4ead414e4ea4a08a7a84dafc380e/event.json').read_text())
        model_file=source_event['currents_file']
        persistent=[{'neuron_ids':[str(adapter.ids[adapter.indices[0]])],'current_mv':.125}]
        legacy=[*persistent,{'neuron_currents_file':model_file}]
        raw=env.brain.read_state(output/'brain_before')
        action=env.shared_passthrough_action()
        actual_drive=dict(zip(('retina_left','retina_right','sugar'),((action.astype(float)+1)/2).tolist()))
        budget=prepare_parallel_stimulation(env.brain,raw,first['currents_file'],legacy,actual_drive)
        created_files.append(budget['applied_currents_file']['path'])
        env.set_general_stimulation([*legacy,{'neuron_currents_file':budget['applied_currents_file']}])
        _,_,term,trunc,info=env.step(action);neural=info['neural']
        assert not term and not trunc
        assert neural['visual_input_enabled'] is True and neural['input_rgb_sha256']==packet['vision']['rgb_sha256']
        assert neural['general_stimulation']['current_vector_sha256']==budget['expected_general_currents_sha256']
        assert len(neural['general_stimulation']['neuron_current_components'])==2
        assert env.general_stimulation==persistent and env._shared_body_packet is None
        assert abs(neural['neural_drive']['additional_current_mv_equivalent']['sugar']-15)<1e-6
        second_packet=env.shared_body_senses()
        assert second_packet['simulation_time_s']>packet['simulation_time_s']
        assert second_packet['packet_sha256']!=packet['packet_sha256']
        Image.fromarray(env.render()).save(output/'body_senses_after_step.png')
        targets=np.vstack([np.full(6370,.08,dtype=np.float32),np.full(6370,-.04,dtype=np.float32)])
        training=adapter.fit_supervised([packet,second_packet],targets,steps=2,learning_rate=.001,
            label_source='Explicit engineering test labels +0.08/-0.04 mV on real measured body states; not recordings of natural proprioceptive neuron responses')
        assert training['changed_parameters']>0
        checkpoint=adapter.save(output/'checkpoint',packet);loaded=BodySenseAdapter.load(checkpoint['path'])
        with torch.no_grad():
            x=torch.from_numpy(adapter.codec.encode(second_packet));exact=bool(torch.equal(adapter(x),loaded(x)))
        assert exact and adapter.adapter_sha256()==loaded.adapter_sha256()
        after=loaded.prepare(second_packet,output/'trained');created_files.append(after['currents_file']['path'])
        assert after['non_target_nonzero']==0
        with np.load(after['currents_file']['path'],allow_pickle=False) as f:
            assert f['currents_mv'].shape==(166700,) and f['ids'].shape==(166700,)
        # Cleanup one-off additions never removes the original persistent input.
        _,_,_,_,next_info=env.step(env.shared_passthrough_action())
        assert env.general_stimulation==persistent and next_info['neural']['visual_input_enabled'] is True
        assert abs(next_info['neural']['neural_drive']['additional_current_mv_equivalent']['sugar']-15)<1e-6
        evidence.update(status='passed',feature_dim=adapter.codec.feature_dim,
            packet_shapes={k:packet['layout'][k] for k in ('nq','nv','nu','nbody','nsensor','nsensordata')},
            contact_count=packet['contact_count'],first_body_time_s=packet['simulation_time_s'],
            next_body_time_s=second_packet['simulation_time_s'],brain_interval_ms=neural['neural']['duration_ms'],
            body_interval_ms=packet['body_interval_s']*1000,
            shared_raw_packet_and_feature_identity_verified=True,sampling_did_not_advance_physics=True,
            untrained_encoder=first,trained_encoder=after,training=training,checkpoint=checkpoint,
            checkpoint_reload_exact=exact,parallel_budget=budget,actual_neural=neural['neural'],
            actual_stimulation=neural['general_stimulation'],original_visual_and_sugar_preserved=True,
            original_persistent_general_preserved=True,
            interpretation='Actual body state/contact forces encoded into extra current for real vnc_sensory cells; a trained engineering mapping, not reconstructed native body-to-neuron anatomy. Body and brain retain separate clocks.')
        print('PASS '+json.dumps({'output':str(output),'feature_dim':adapter.codec.feature_dim,'training':training,'initial_range':[first['minimum_current_mv'],first['maximum_current_mv']],'time_s':[packet['simulation_time_s'],second_packet['simulation_time_s']]}),flush=True)
    finally:
        if env is not None:
            pid=env.brain.process.pid;env.close();evidence['worker_closed']={'pid':pid,'returncode':env.brain.process.poll()}
        # Preserve descriptors and source files as immutable evidence for callers.
        evidence['created_current_files']=created_files
        (output/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
