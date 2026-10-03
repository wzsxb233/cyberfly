"""Actual brain/physics one-interval file consumption, including a failed step."""
import json
import os
from pathlib import Path
import uuid

ROOT=Path(__file__).resolve().parents[2]

def main():
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ.setdefault('MUJOCO_GL','egl')
    import numpy as np
    import torch
    from scenarios import create_scenario
    from shared_io.neuron_link import NeuronStateCodec
    from connectome_adapter.neuron_currents import write_neuron_currents
    torch.set_num_threads(1)
    output=ROOT/'artifacts/async-file-queue-validation'/uuid.uuid4().hex[:12];output.mkdir(parents=True)
    record=json.loads((ROOT/'artifacts/grounding_collection_20260913_sixseeds_v2/manifest.json').read_text())['records'][0]
    env=create_scenario({'name':'fly_neural_link','env_kwargs':{'max_episode_steps':8,'learning':False,
        'brain_checkpoint':record['trajectory_config']['brain_checkpoint'],
        'motor_readout_checkpoint':record['trajectory_config']['motor_checkpoint']}})
    report={'status':'running','main_lab_mutated':False,'model_called':False,'actual_full_brain_and_3d':True}
    def save():(output/'evidence.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print('OUTPUT='+str(output),flush=True);save()
    try:
        env.reset(seed=7519);memory=env.brain.describe()['learning']['sha256']
        codec=NeuronStateCodec();codec.read(env.brain.read_state(output/'before'))
        current=np.zeros(166700,dtype=np.float32);current[np.flatnonzero(codec.write_mask)[0]]=1.25
        queued=write_neuron_currents(current,codec.ids)
        persistent={'group_id':env.brain.groups()['macro_groups'][0]['group_id'],'current_mv':0.05}
        env.set_general_stimulation([persistent,{'neuron_currents_file':queued}]);action=env.shared_passthrough_action()
        steps=[]
        for _ in range(2):
            original=list(env.general_stimulation)
            try:_,_,_,_,info=env.step(action)
            finally:env.set_general_stimulation([item for item in original if 'neuron_currents_file' not in item])
            steps.append(info['neural']);assert env.general_stimulation==[persistent]
        assert len(steps[0]['general_stimulation']['neuron_current_components'])==1
        assert steps[0]['general_stimulation']['neuron_current_components'][0]['file_sha256']==queued['sha256']
        assert steps[1]['general_stimulation']['neuron_current_components']==[]
        invalid={**queued,'sha256':'0'*64};env.set_general_stimulation([persistent,{'neuron_currents_file':invalid}])
        original=list(env.general_stimulation);failure=None
        try:env.step(action)
        except Exception as exc:failure=str(exc)
        finally:env.set_general_stimulation([item for item in original if 'neuron_currents_file' not in item])
        assert failure and env.general_stimulation==[persistent]
        _,_,_,_,info=env.step(action);steps.append(info['neural'])
        assert info['neural']['general_stimulation']['neuron_current_components']==[]
        assert env.brain.describe()['learning']['sha256']==memory
        report.update(status='passed',queued_file=queued,queued_actual_load_count=1,
            original_persistent_group=persistent,persistent_stimulus_preserved=True,
            failed_input_consumed=True,invalid_sha_error=failure,
            intervals=[{'neural':s['neural'],'actual_general':s['general_stimulation']} for s in steps],
            brain_memory_sha256=memory,brain_memory_preserved=True,
            note='Original file inputs are one interval; re-adding all original files would incorrectly repeat them. No LLM claim is made by this queue test.')
        save();print(json.dumps({'status':'passed','queued_actual_load_count':1,'failed_input_consumed':True}),flush=True)
    except Exception as exc:report.update(status='failed',error=str(exc));save();raise
    finally:env.close()

if __name__=='__main__':main()
