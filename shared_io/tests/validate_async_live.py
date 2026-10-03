"""Independent real brain + 3D + candidate model; never touches the main lab."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import time
import uuid

ROOT=Path(__file__).resolve().parents[2]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--queued-file',action='store_true');args=parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ.setdefault('MUJOCO_GL','egl')
    import numpy as np
    import torch
    from scenarios import create_scenario
    from shared_io.async_loop import AsyncSharedLoop
    from shared_io.bus import SharedLinkClient
    torch.set_num_threads(1)
    output=ROOT/'artifacts/async-shared-validation'/uuid.uuid4().hex[:12];output.mkdir(parents=True)
    def save(value):(output/'evidence.json').write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
    print('OUTPUT='+str(output),flush=True)
    recorded=json.loads((ROOT/'artifacts/grounding_collection_20260913_sixseeds_v2/manifest.json').read_text())['records'][0]['trajectory_config']
    env=create_scenario({'name':'fly_neural_link','env_kwargs':{'max_episode_steps':300,'learning':False,
        'brain_checkpoint':recorded['brain_checkpoint'],'motor_readout_checkpoint':recorded['motor_checkpoint']}})
    loop=None;report={'status':'starting','main_lab_mutated':False,'candidate_endpoint':'http://127.0.0.1:18648'}
    save(report)
    try:
        env.reset(seed=7517);memory=env.brain.describe()['learning']['sha256'];start_position=env.task_state()['position_mm']
        loop=AsyncSharedLoop(env.brain,SharedLinkClient('http://127.0.0.1:18648'),max_new_tokens=16,max_delay_s=90)
        queued=None;queued_loads=0
        if args.queued_file:
            from connectome_adapter.neuron_currents import write_neuron_currents
            codec=loop.prototype.neuron_codec
            codec.read(loop.prototype._raw_state(output/'queued-input-snapshot'))
            current=np.zeros(166700,dtype=np.float32);current[np.flatnonzero(codec.write_mask)[0]]=1.25
            queued=write_neuron_currents(current,codec.ids)
            persistent={'group_id':env.brain.groups()['macro_groups'][0]['group_id'],'current_mv':0.05}
            env.set_general_stimulation([persistent,{'neuron_currents_file':queued}])
        action=env.shared_passthrough_action();ports=dict(zip(('retina_left','retina_right','sugar'),((action.astype(float)+1)/2).tolist()))
        original=list(env.general_stimulation);rgb=env.shared_input_rgb();body=env.shared_body_senses()
        begin=time.monotonic();accepted=loop.observe('用一句简短中文描述画面中的果蝇。',rgb,body_packet=body,
            legacy_general=original,legacy_neural_drive=ports,visual_input_enabled=True)
        assert accepted['accepted'];observe_return_s=time.monotonic()-begin
        counts={'encoding':0,'decoding':0};applied=0;started=time.monotonic();iterations=[]
        while time.monotonic()-started<150:
            loop.poll();phase=loop.phase
            if phase=='ready':
                original=list(env.general_stimulation)
                rgb=env.shared_input_rgb();body=env.shared_body_senses()
                stimulus=loop.take_stimulation(rgb,body_packet=body,legacy_general=original,
                    legacy_neural_drive=ports,visual_input_enabled=True)
                assert stimulus is not None and loop.take_stimulation(rgb,body_packet=body) is None
                try:
                    env.set_general_stimulation([*original,*stimulus['additional_stimulation']])
                    _,_,done,truncated,info=env.step(action)
                finally:env.set_general_stimulation([item for item in original if 'neuron_currents_file' not in item])
                applied+=1;loop.feedback(info['neural'],body_state=env.task_state(),body_packet=env.body_sensory_packet())
            elif phase in ('encoding','decoding'):
                t=time.monotonic();_,_,done,truncated,info=env.step(action)
                counts[phase]+=1;iterations.append({'stage':phase,'step_seconds':time.monotonic()-t,'body_time_s':env.task_state()['simulation_time_s']})
            elif phase=='idle':break
            else:raise RuntimeError('Unexpected async phase '+phase)
            if queued:
                queued_loads+=sum(x['file_sha256']==queued['sha256'] for x in info['neural']['general_stimulation']['neuron_current_components'])
                assert env.general_stimulation==[persistent]
            if done or truncated:raise RuntimeError('Independent validation episode ended early')
        event=loop.last
        assert event and event['phase']=='complete',event
        assert counts['encoding']>0 and counts['decoding']>0 and applied==1
        assert event['current_consumption_count']==1 and event['original_inputs_preserved'] and event['applied_currents_verified']
        assert event['actual_input_verified'] and event['shared_input_verified'] is False and event['model_facts'] is None
        assert event['actual_application']['neural_time_before_ms']>event['neural_time_before_ms']
        assert event['actual_application']['image_rgb_sha256']!=event['image_rgb_sha256']
        assert event['model_decode_evidence']['brain_ablation']['performed'] is False
        assert event['model_decode_evidence']['generation_token_budget']==16
        assert env.brain.describe()['learning']['sha256']==memory
        if queued:assert queued_loads==1
        report.update(status='passed',event_id=event['event_id'],event_path=str(ROOT/'artifacts/shared_io'/event['event_id']/'event.json'),
            actual_model=True,actual_full_brain=True,actual_3d=True,model_wait_body_steps=counts,application_count=applied,
            actor_observe_return_s=observe_return_s,model_observation=event['model_observation'],actual_application=event['actual_application'],
            say=event['say'],brain_memory_sha256=memory,brain_memory_preserved=True,model_facts=None,
            start_position_mm=start_position,final_position_mm=env.task_state()['position_mm'],iterations=iterations,
            queued_file=queued,queued_file_actual_loads=queued_loads if queued else None,
            file_stimulus_semantics='One interval, including failed steps; persistent group currents are preserved',
            scope='Actual delayed control and nonblocking body progress; response accuracy is not asserted')
        save(report);print(json.dumps({k:report[k] for k in ('status','event_id','model_wait_body_steps','actor_observe_return_s','say')},ensure_ascii=False),flush=True)
    except Exception as exc:
        report.update(status='failed',error=str(exc));save(report);raise
    finally:
        if loop:loop.close()
        env.close()

if __name__=='__main__':main()
