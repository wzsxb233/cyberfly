"""Mathematical/identity tests use synthetic state over real release IDs."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

from shared_io.motor_readout import MotorReadout, GRAPH_PATH, FEATURE_DIM, array_sha256
from scenarios.neural_link import FlyNeuralLink
from scenarios.brain_sandbox import BrainSandboxScenario
from scenarios.doom_neural_link import DoomNeuralLink
from scenarios.full_brain_sandbox import FullBrainSandboxScenario


class MotorReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.model = MotorReadout(seed=17)
        with np.load(GRAPH_PATH, allow_pickle=False) as data:
            cls.ids = data["ids"]

    def fixture(self):
        arrays = {"ids": self.ids.copy(), "v": np.full(len(self.ids), -65, dtype=np.float32),
                  "counts": np.zeros(len(self.ids), dtype=np.int32)}
        arrays["counts"][self.model.indices[0]] = 2
        summary = {"neural": {"neurons":166700,"edges":25582938,"duration_ms":100.,
                   "simulation_ms":100.,"spikes_this_step":2,"counts_sha256":array_sha256(arrays["counts"])}}
        descriptor = {"neurons":166700,"edges":25582938,"simulation_ms":100.}
        return arrays, descriptor, summary

    def test_exact_output_selection_and_rate_feature(self):
        x = self.model.encode(*self.fixture())
        self.assertEqual(x.shape, (FEATURE_DIM,))
        self.assertEqual(len(self.model.neuron_ids), 2129)
        self.assertEqual(self.model.selection['output_ids_sha256'], '8f49c6883c8f169575187f4e38e6512a96a5652e8902e3cd88fc2b7e57b902d0')
        self.assertAlmostEqual(float(x[0]), float(np.log1p(20)/np.log1p(200)), places=6)
        self.assertTrue(np.all(x[1::2] == 0))

    def test_measurement_identity_time_and_counts_are_bound(self):
        for kind in ('ids', 'time', 'spike_hash', 'negative_counts', 'nan_voltage', 'zero_interval'):
            a, d, s = self.fixture()
            if kind=='ids':a['ids'][0],a['ids'][1]=a['ids'][1],a['ids'][0]
            elif kind=='time':d['simulation_ms']=99
            elif kind=='spike_hash':s['neural']['counts_sha256']='incorrect'
            elif kind=='negative_counts':a['counts'][0]=-1
            elif kind=='nan_voltage':a['v'][0]=np.nan
            else:s['neural']['duration_ms']=0
            with self.subTest(kind=kind), self.assertRaises(ValueError):self.model.encode(a,d,s)

    def test_real_optimizer_and_safe_checkpoint_reload(self):
        model=MotorReadout(seed=24)
        x=model.encode(*self.fixture())
        features=np.stack([x,x+.01]);targets=np.array([[-.6,.4],[.2,-.7]],dtype=np.float32)
        result=model.fit_supervised(features,targets,label_source='Synthetic unit-test actions, not behavior evidence',steps=2)
        self.assertGreater(result['changed_parameters'],0)
        self.assertNotEqual(result['initial_weights_sha256'],result['final_weights_sha256'])
        with tempfile.TemporaryDirectory() as path:
            model.save(path);loaded=MotorReadout.load(path)
            self.assertTrue(torch.equal(model(torch.from_numpy(features)),loaded(torch.from_numpy(features))))
            weights=Path(path)/'weights.npz';weights.write_bytes(weights.read_bytes()+b'invalid')
            with self.assertRaises(ValueError):MotorReadout.load(path)

    def test_independent_current_setter_preserves_pinned_rgb(self):
        descriptor={'schema':1,'path':str(ROOT / 'artifacts/neuron_currents/test.npz'),
                    'sha256':'0'*64,'ids_sha256':'1'*64,'currents_sha256':'2'*64,'neurons':166700}
        for cls in (FlyNeuralLink,BrainSandboxScenario):
            env=object.__new__(cls);env.general_stimulation=[]
            env._shared_input_rgb=np.zeros((2,3,3),dtype=np.uint8);pinned=env._shared_input_rgb
            env.set_neuron_currents(descriptor)
            self.assertIs(env._shared_input_rgb,pinned)
            self.assertEqual(env.general_stimulation,[{'neuron_currents_file':descriptor}])
            env.set_neuron_currents(None)
            self.assertEqual(env.general_stimulation,[])
            self.assertIs(env._shared_input_rgb,pinned)

    def test_unsupported_motor_destination_rejected(self):
        bodyless=object.__new__(BrainSandboxScenario)
        with self.assertRaises(ValueError):bodyless.set_motor_readout('unused')
        doom=object.__new__(DoomNeuralLink);doom.scenario_spec={'embodiment':'game'}
        with self.assertRaises(ValueError):doom.set_motor_readout('unused')

    def test_model_input_preserves_other_stimuli_and_replaces_old_file(self):
        descriptor={'schema':1,'path':str(ROOT / 'artifacts/neuron_currents/test.npz'),
                    'sha256':'0'*64,'ids_sha256':'1'*64,'currents_sha256':'2'*64,'neurons':166700}
        previous={**descriptor,'sha256':'3'*64}
        persistent=[{'neuron_ids':['10001'],'current_mv':2.0}]
        for cls in (FlyNeuralLink,BrainSandboxScenario,FullBrainSandboxScenario):
            env=object.__new__(cls);env.general_stimulation=deepcopy(persistent)+[{'neuron_currents_file':previous}]
            env.semantic_drive={'retina_left':.2,'retina_right':.35,'sugar':.6}
            env._last_macro_action=np.linspace(-.1,.1,47,dtype=np.float32)
            env.append_neuron_currents(descriptor)
            self.assertEqual(env.general_stimulation, persistent+[{'neuron_currents_file':descriptor}])
            env.set_neuron_currents(None)
            self.assertEqual(env.general_stimulation,persistent)
            expected=env._last_macro_action if cls is FullBrainSandboxScenario else np.array([-.6,-.3,.2],dtype=np.float32)
            self.assertTrue(np.allclose(env.shared_passthrough_action(),expected,rtol=0,atol=1e-7))
            env.set_neuron_currents(descriptor,preserve_existing=False)
            self.assertEqual(env.general_stimulation,[{'neuron_currents_file':descriptor}])


if __name__ == '__main__':unittest.main()
