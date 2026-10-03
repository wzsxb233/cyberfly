"""The model adds one input while original vision, sugar and stimuli persist."""
from copy import deepcopy
import json
from pathlib import Path
import uuid
import numpy as np
from PIL import Image

from scenarios import create_scenario
from shared_io.motor_readout import GRAPH_PATH, array_sha256

ROOT=Path(__file__).resolve().parents[2]


def main():
    output=ROOT/'artifacts/preserved_input_checks'/uuid.uuid4().hex[:12]
    output.mkdir(parents=True)
    with np.load(GRAPH_PATH,allow_pickle=False) as graph:
        ids=graph['ids'];sensory=np.isin(graph['superclass'],('cb_sensory','ol_sensory','vnc_sensory'))
    indices=np.flatnonzero(sensory)
    baseline_index=int(indices[0])
    persistent=[{'neuron_ids':[str(ids[baseline_index])],'current_mv':1.25}]
    semantic={'retina_left':.2,'retina_right':.35,'sugar':.6}
    vector=np.zeros(len(ids),dtype=np.float32)
    vector[sensory]=np.linspace(.25,.75,int(sensory.sum()),dtype=np.float32)
    results=[];env=None;current_file=None
    try:
        for name in ('fly_neural_link','brain_sandbox','full_brain_sandbox'):
            env=create_scenario({'name':name,'env_kwargs':{'max_episode_steps':20,'visual_input_enabled':True,'learning':False}})
            env.reset(seed=20260913)
            env.apply_task({'neural_drive':semantic,'general_stimulation':deepcopy(persistent)})
            if name=='full_brain_sandbox':
                original_action=np.linspace(-.02,.02,47,dtype=np.float32)
                env.step(original_action)
                assert np.array_equal(env.shared_passthrough_action(),original_action)
                arrays,_=env.brain.read_arrays()
                macro_base=(30*original_action.astype(float)).astype(np.float32)[arrays['macro_group_index']]
            else:
                original_action=env.shared_passthrough_action()
                macro_base=np.zeros(len(ids),dtype=np.float32)
            expected_baseline=macro_base.copy();expected_baseline[baseline_index]+=np.float32(1.25)
            expected_sum=expected_baseline+vector
            pinned=env.shared_input_rgb()
            current_file=env.brain.write_neuron_currents(vector,ids=ids)
            # A prior one-off descriptor must be replaced, never queued twice.
            env.set_neuron_currents(current_file)
            env.append_neuron_currents(current_file)
            assert len(env.general_stimulation)==2
            assert env.general_stimulation[0]==persistent[0]
            assert env.visual_input_enabled is True and env.semantic_drive==semantic
            action=env.shared_passthrough_action()
            assert np.array_equal(action,original_action)
            _,_,term,trunc,first=env.step(action)
            actual=first['neural']
            assert not term and not trunc
            assert actual['visual_input_enabled'] is True
            assert actual['input_rgb_sha256']==array_sha256(pinned)
            assert actual['general_stimulation']['current_vector_sha256']==array_sha256(expected_sum)
            assert env.general_stimulation==persistent
            if name!='full_brain_sandbox':
                assert all(abs(actual['neural_drive']['normalized'][key]-value)<1e-7 for key,value in semantic.items())
                assert abs(actual['neural_drive']['additional_current_mv_equivalent']['sugar']-18)<1e-6
            Image.fromarray(env.render()).save(output/(name+'.png'))
            _,_,_,_,second=env.step(env.shared_passthrough_action())
            following=second['neural']
            assert following['visual_input_enabled'] is True
            assert following['general_stimulation']['current_vector_sha256']==array_sha256(expected_baseline)
            assert env.general_stimulation==persistent
            if name!='full_brain_sandbox':
                assert abs(following['neural_drive']['additional_current_mv_equivalent']['sugar']-18)<1e-6
            env.set_neuron_currents(None)
            assert env.general_stimulation==persistent and env.semantic_drive==semantic
            row={'scenario':name,'status':'passed','visual_projection_enabled_on_both_steps':True,
                'semantic_context_preserved':True,'original_action_preserved':True,
                'sugar_mv_first':actual['neural_drive']['additional_current_mv_equivalent']['sugar'],
                'sugar_mv_next':following['neural_drive']['additional_current_mv_equivalent']['sugar'],
                'sugar_note':'Full sandbox actions remain macro currents; three-port context was already observation-only' if name=='full_brain_sandbox' else 'Original sugar port remained active on both intervals',
                'persistent_stimulus':persistent,'independent_vector_sha256':array_sha256(vector),
                'expected_added_general_sha256':array_sha256(expected_sum),
                'actual_added_general':actual['general_stimulation'],
                'expected_next_general_sha256':array_sha256(expected_baseline),
                'actual_next_general':following['general_stimulation'],
                'descriptor_consumed_once':True,'pinned_raw_rgb_preserved':True,
                'first_neural':actual['neural'],'next_neural':following['neural']}
            Path(current_file['path']).unlink();current_file=None
            pid=env.brain.process.pid;env.close();row['worker_closed']={'pid':pid,'returncode':env.brain.process.poll()};env=None
            results.append(row)
            print('PASS '+name,flush=True)
    finally:
        if env is not None:env.close()
        if current_file:Path(current_file['path']).unlink(missing_ok=True)
        evidence={'status':'passed' if len(results)==3 else 'incomplete','scenarios':results,
            'input_contract':'New model currents add to existing inputs; original visual flag and contextual ports stay unchanged. Only the newest model file is one-shot.',
            'no_live_lab_mutation':True}
        (output/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        print('EVIDENCE '+str(output/'evidence.json'),flush=True)


if __name__=='__main__':main()
