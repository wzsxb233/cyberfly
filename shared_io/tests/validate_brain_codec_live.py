"""Real full-brain numerical IO and same-initial-state sensitivity validation.

No LLM execution is claimed here: model-hidden extraction/coupler training are
verified separately by the shared IO service. This audit verifies its numerical
brain boundary, full coverage, exact restored initial states, and actual pixels.
"""
from pathlib import Path
import hashlib,json,time,uuid
import numpy as np
from PIL import Image
from connectome_adapter import ConnectomeClient
from shared_io.brain_codec import BrainFeatureCodec
from scenarios import create_scenario

ROOT=Path(__file__).resolve().parents[2]

def digest(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()

def main():
    out=ROOT/'artifacts'/'shared_io_codec_checks'/uuid.uuid4().hex[:12]
    out.mkdir(parents=True)
    codec=BrainFeatureCodec()
    evidence={'status':'running','codec':codec.describe(),
        'scope':'Real numerical brain IO and same-initial-state sensitivity; not an LLM or trained coupling claim'}
    brain=ConnectomeClient(learning=False,timeout=180,log_dir=out/'brain')
    try:
        brain.reset(preserve_weights=True)
        initial_features=codec.encode_group_state(brain.group_state(),duration_ms=0,learning=False)
        initial_arrays,_=brain.read_arrays()
        saved=brain.save(out/'initial_brain')
        pixels=np.zeros((84,84,3),dtype=np.uint8);pixel_hash=digest(pixels)
        results=[];feature_vectors=[]
        for label,current in [('negative',-3.),('negative_repeat',-3.),('positive',3.)]:
            restored=brain.restore(out/'initial_brain')
            before,_=brain.read_arrays()
            equal_keys=['ids','v','counts','drive','cumulative_counts','group_index','macro_group_index']
            assert all(np.array_equal(before[k],initial_arrays[k]) for k in equal_keys),'Restored initial state differs'
            assert restored['learning']['sha256']==saved['memory_sha256']
            currents=np.full(47,current,dtype=np.float32)
            result=codec.step_brain(brain,pixels,currents,duration_ms=28.6)
            features=codec.encode_step(result)
            assert features.shape==(240,) and features.dtype==np.float32
            assert sum(result['group_state']['counts'])==166700
            assert result['general_stimulation']['driven_neurons']==166700
            assert result['general_stimulation']['topology_modified'] is False
            assert result['input_rgb_sha256']==pixel_hash
            assert result['learning']['sha256']==saved['memory_sha256']
            feature_vectors.append(features)
            np.savez_compressed(out/f'{label}.npz',features=features,currents_mv=currents)
            row={'condition':label,'current_mv':current,'initial_state_exact':True,'restored_keys':equal_keys,
                'spikes':result['neural']['spikes_this_step'],'counts_sha256':result['neural']['counts_sha256'],
                'feature_sha256':digest(features),'input_rgb_sha256':result['input_rgb_sha256'],
                'neural':result['neural'],'general_stimulation':result['general_stimulation']}
            results.append(row);print(label,row['spikes'],row['feature_sha256'],flush=True)
        repeat=float(np.linalg.norm(feature_vectors[0]-feature_vectors[1]))
        contrast=float(np.linalg.norm(feature_vectors[0]-feature_vectors[2]))
        assert contrast>max(1e-4,10*repeat),('Response not separable from repeat variation',repeat,contrast)
        evidence.update(conditions=results,repeat_feature_l2=repeat,negative_positive_feature_l2=contrast,
            changed_feature_values=int(np.count_nonzero(feature_vectors[0]!=feature_vectors[2])),
            initial_feature_shape=list(initial_features.shape))
    finally:brain.close()
    env_checks=[]
    for name in ['full_brain_sandbox','fly_neural_link']:
        env=create_scenario({'name':name,'env_kwargs':{'max_episode_steps':4,'learning':False}})
        try:
            _,initial=env.reset(seed=20260913)
            before_features=codec.encode_env_state(env.task_state())
            if name=='fly_neural_link':
                body=env.body.unwrapped
                body.show_feedback('food',.8,5000)
            pixels=env.shared_input_rgb();pixel_hash=digest(pixels)
            source=env.shared_input_metadata()
            if name=='fly_neural_link':
                # Change only the displayed props after pinning. The numerical
                # model's captured frame must remain the brain's exact input.
                env.show_feedback('aversive',.8,5000)
                preview=env.render()
                assert digest(preview)!=pixel_hash
            else:
                assert not np.any(pixels),'Bodyless shared input must be the actual black field'
                assert digest(env.render())!=pixel_hash,'Group chart is a separate display'
            currents=np.linspace(-.5,.5,47,dtype=np.float32)
            obs,reward,term,trunc,info=codec.step_env(env,currents)
            features=codec.encode_env_state(info)
            assert info['neural']['input_rgb_sha256']==pixel_hash
            assert features.shape==(240,) and env.observation_space.contains(obs)
            env.set_macro_group_currents(None)
            assert env.general_stimulation==[]
            env.set_macro_group_currents(np.zeros(47,dtype=np.float32))
            assert env.general_stimulation==[{'macro_group_currents_mv':[0.]*47}]
            env.set_macro_group_currents(None)
            Image.fromarray(pixels).save(out/f'{name}_shared_input.png')
            Image.fromarray(env.render()).save(out/f'{name}_display.png')
            row={'scenario':name,'original_observation_shape':list(obs.shape),'encoded_shape':list(features.shape),
                'input_shape':list(pixels.shape),'input_rgb_sha256':pixel_hash,
                'actual_brain_input_sha256':info['neural']['input_rgb_sha256'],
                'source':source,'feature_l2_change':float(np.linalg.norm(features-before_features)),
                'spikes':info['neural']['neural']['spikes_this_step'],'macro_setter_clear_verified':True}
            env_checks.append(row);print(name,row,flush=True)
        finally:env.close()
    evidence.update(environment_checks=env_checks,status='passed')
    (out/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
    print('PASS',out/'evidence.json',flush=True)

if __name__=='__main__':main()
