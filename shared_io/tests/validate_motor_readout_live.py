"""Real brain/body supervised readout plus one-interval independent-input check."""
from pathlib import Path
import json
import time
import uuid

import numpy as np
from PIL import Image
import torch

from scenarios import create_scenario
from shared_io.motor_readout import MotorReadout, GRAPH_PATH, array_sha256

ROOT = Path(__file__).resolve().parents[2]


def main():
    torch.set_num_threads(1)
    output = ROOT / "artifacts/motor_readout_checks" / uuid.uuid4().hex[:12]
    output.mkdir(parents=True)
    evidence = {"artifact_directory": str(output), "isolation": "Own environments; no live lab HTTP/session mutation"}
    env = None
    try:
        env = create_scenario({"name": "fly_neural_link", "env_kwargs": {"learning": False, "max_episode_steps": 100}})
        env.reset(seed=20260913)
        readout = MotorReadout(seed=20260913)
        features, teachers, samples = [], [], []
        for action in ([-1,-1,-1], [1,-1,-1], [-1,1,-1]):
            obs, reward, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
            assert not terminated and not truncated and obs.shape == (33,)
            arrays, descriptor = env.brain.read_arrays()
            features.append(readout.encode(arrays, descriptor, info["neural"]))
            teachers.append(info["body_action"])
            samples.append({"input": action, "body_action_teacher": info["body_action"],
                "simulation_ms": info["neural"]["neural"]["simulation_ms"],
                "spikes": info["neural"]["neural"]["spikes_this_step"],
                "counts_sha256": info["neural"]["neural"]["counts_sha256"],
                "features_sha256": array_sha256(features[-1])})
        x, y = np.asarray(features), np.asarray(teachers, dtype=np.float32)
        np.savez(output / "measured_teaching_examples.npz", features=x, target_actions=y)
        with torch.no_grad():
            heldout_before = float(torch.mean((readout(torch.from_numpy(x[2:]))-torch.from_numpy(y[2:]))**2))
        optimization = readout.fit_supervised(x[:2], y[:2], steps=2, learning_rate=1e-4,
            label_source="Explicit imitation labels: existing rule BCI's actual body commands on the same measured neural intervals; not natural motor ground truth")
        with torch.no_grad():
            heldout_after = float(torch.mean((readout(torch.from_numpy(x[2:]))-torch.from_numpy(y[2:]))**2))
        checkpoint = readout.save(output / "checkpoint")
        loaded = MotorReadout.load(output / "checkpoint")
        with torch.no_grad():
            exact = bool(torch.equal(readout(torch.from_numpy(x)), loaded(torch.from_numpy(x))))
        assert exact and optimization["changed_parameters"] > 0
        print("TRAINED " + json.dumps({**optimization, "checkpoint": checkpoint}), flush=True)
        selected = env.set_motor_readout(output / "checkpoint")
        before_qpos = env.body.unwrapped.sim.mj_data.qpos.copy()
        obs, reward, terminated, truncated, info = env.step(np.array([-1,-1,-1], dtype=np.float32))
        arrays, descriptor = env.brain.read_arrays()
        prediction, trace = loaded.predict_measurement(arrays, descriptor, info["neural"])
        assert np.array_equal(prediction, np.asarray(info["body_action"], dtype=np.float32))
        assert info["neural"]["motor_readout"]["features_sha256"] == trace["features_sha256"]
        assert obs.shape == (33,) and not np.array_equal(before_qpos, env.body.unwrapped.sim.mj_data.qpos)
        Image.fromarray(env.render()).save(output / "learned_readout_body.png")
        integration = {"selection": selected, "trace": trace, "obs_shape": list(obs.shape),
                       "body_qpos_changed": True, "body_action_exactly_learned_prediction": True}
        # Only sensory inputs receive explicit currents; output cells stay at
        # zero external stimulus and are measured downstream of the full graph.
        with np.load(GRAPH_PATH, allow_pickle=False) as graph:
            sensory = np.isin(graph['superclass'], ('cb_sensory','ol_sensory','vnc_sensory'))
            ids = graph['ids']
        currents = np.zeros(len(ids),dtype=np.float32)
        currents[sensory] = np.linspace(.5,3,int(sensory.sum()),dtype=np.float32)
        assert int(sensory.sum()) == 17336 and np.all(currents[readout.indices] == 0)
        direct = env.brain.write_neuron_currents(currents, ids=ids)
        env.set_visual_input_enabled(False)
        pinned = env.shared_input_rgb()
        pinned_hash = array_sha256(pinned)
        env.show_feedback('food', duration_ms=1000)
        preview = env.render()
        assert array_sha256(preview) != pinned_hash
        env.set_neuron_currents(direct)
        started = time.monotonic()
        obs, reward, terminated, truncated, stimulated = env.step(np.full(3,-1,dtype=np.float32))
        latency = time.monotonic()-started
        actual = stimulated['neural']
        assert actual['input_rgb_sha256'] == pinned_hash
        assert actual['visual_input_enabled'] is False
        assert actual['general_stimulation']['current_vector_sha256'] == direct['currents_sha256']
        assert env.general_stimulation == []
        _, _, _, _, following = env.step(np.full(3,-1,dtype=np.float32))
        assert following['neural']['general_stimulation']['driven_neurons'] == 0
        assert following['neural']['visual_input_enabled'] is False
        env.set_neuron_currents(None)
        env.set_motor_readout(None)
        _, _, _, _, restored = env.step(np.full(3,-1,dtype=np.float32))
        assert restored['motor_readout']['mode'] == 'rule_bci'
        assert np.array_equal(np.asarray(restored['body_action'],dtype=np.float32),env.decode_body_action(restored['neural']['native_action']))
        evidence.update(samples=samples,optimization=optimization,heldout_mse_before=heldout_before,
            heldout_mse_after=heldout_after,checkpoint=checkpoint,checkpoint_reload_exact=exact,
            learned_body_integration=integration,
            independent_input={'descriptor':direct,'sensory_cells_stimulated':int(sensory.sum()),
                'output_cells_directly_stimulated':0,'input_rgb_sha256':pinned_hash,
                'preview_rgb_sha256':array_sha256(preview),'pinned_frame_preserved':True,
                'one_interval_consumption':True,'current_vector_exact':True,
                'visual_input_enabled':False,'wall_seconds_including_readout_snapshot':latency,
                'actual_general_stimulation':actual['general_stimulation'],
                'following_driven_neurons':following['neural']['general_stimulation']['driven_neurons']},
            rule_bci_restored=True,
            interpretation='Real two-step supervised motor-adapter training and physical execution. Teacher is the prior engineering BCI; neither natural motor recovery nor task mastery is demonstrated.')
        Path(direct['path']).unlink()
        worker_pid = env.brain.process.pid
        env.close(); evidence['body_worker_closed']={'pid':worker_pid,'returncode':env.brain.process.poll()}; env=None
        # Same one-shot descriptor API on the independent bodyless branch.
        env = create_scenario({'name':'full_brain_sandbox','env_kwargs':{'max_episode_steps':8,'visual_input_enabled':False}})
        env.reset(seed=20260913)
        direct = env.brain.write_neuron_currents(currents,ids=ids)
        env.set_neuron_currents(direct)
        obs, _, _, _, bodyless = env.step(np.zeros(47,dtype=np.float32))
        assert obs.shape == (146,) and not bodyless['neural']['visual_input_enabled']
        assert bodyless['neural']['general_stimulation']['current_vector_sha256'] == direct['currents_sha256']
        assert env.general_stimulation == []
        _, _, _, _, bodyless_next = env.step(np.zeros(47,dtype=np.float32))
        assert bodyless_next['neural']['general_stimulation']['driven_neurons'] == 0
        evidence['bodyless']={'obs_shape':list(obs.shape),'descriptor_consumed_once':True,
            'current_vector_exact':True,'visual_projection_disabled':True,
            'next_interval_driven_neurons':bodyless_next['neural']['general_stimulation']['driven_neurons']}
        Path(direct['path']).unlink()
        evidence['status']='passed'
        print('PASSED '+str(output),flush=True)
    finally:
        if env is not None:
            pid=env.brain.process.pid if env.brain else None
            env.close()
            evidence['final_worker_closed']={'pid':pid,'returncode':env.brain.process.poll() if env.brain else None}
        (output/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':main()
