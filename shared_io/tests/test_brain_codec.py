"""Codec contract tests. Synthetic fixtures test math, not neural behavior."""
from copy import deepcopy
import unittest
import numpy as np
from shared_io.brain_codec import BrainFeatureCodec, NEURON_COVERAGE, GRAPH_EDGES, MACRO_PARTITION_SHA256


def zero_state(codec):
    return {'level':'macro','coverage':NEURON_COVERAGE,'partition_sha256':MACRO_PARTITION_SHA256,
        'group_ids':list(codec.group_ids),'counts':codec.group_counts.astype(int).tolist(),
        'spikes':[0]*47,'cumulative_spikes':[0]*47,'voltage_mean':[-65.]*47,
        'voltage_min':[-65.]*47,'voltage_max':[-65.]*47,'simulation_ms':0.}


def step_fixture(codec):
    state=zero_state(codec);state['simulation_ms']=100.;state['spikes'][0]=100
    return {'group_state':state,'neural':{'neurons':NEURON_COVERAGE,'edges':GRAPH_EDGES,
        'duration_ms':100.,'simulation_ms':100.,'spikes_this_step':100},'learning':{'enabled':False}}


class BrainCodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.codec=BrainFeatureCodec()

    def test_full_coverage_and_zero_state(self):
        x=self.codec.encode_group_state(zero_state(self.codec),duration_ms=0)
        self.assertEqual(x.shape,(240,));self.assertEqual(x.dtype,np.float32)
        matrix=x[:235].reshape(47,5)
        self.assertTrue(np.all(matrix[:,:4]==0))
        self.assertAlmostEqual(float(matrix[:,4].sum()),1.,places=6)
        self.assertEqual(len(self.codec.describe()['feature_names']),240)
        self.assertEqual(self.codec.describe()['neuron_coverage'],166700)

    def test_known_rate_and_total_binding(self):
        result=step_fixture(self.codec);x=self.codec.encode_step(result)
        expected=np.log1p(100/self.codec.group_counts[0]/.1)/np.log1p(200)
        self.assertAlmostEqual(float(x[0]),expected,places=6)
        self.assertAlmostEqual(float(x[-3]),.5,places=6)
        self.assertTrue(np.array_equal(x,self.codec.encode_env_state({'neural':result,'group_state':result['group_state']})))
        result['neural']['spikes_this_step']=101
        with self.assertRaises(ValueError):self.codec.encode_step(result)

    def test_no_missing_reordered_or_wrong_population_groups(self):
        for mutation in ['drop','reorder','counts','hash','coverage']:
            with self.subTest(mutation=mutation):
                state=zero_state(self.codec)
                if mutation=='drop':state['group_ids'].pop()
                elif mutation=='reorder':state['group_ids'][:2]=state['group_ids'][1::-1]
                elif mutation=='counts':state['counts'][0]-=1;state['counts'][1]+=1
                elif mutation=='hash':state['partition_sha256']='wrong'
                else:state['coverage']-=1
                with self.assertRaises(ValueError):self.codec.encode_group_state(state,duration_ms=0)

    def test_unknown_interval_or_inconsistent_summary_rejected(self):
        state=zero_state(self.codec);state['spikes'][0]=1
        with self.assertRaises(ValueError):self.codec.encode_group_state(state,duration_ms=0)
        state=zero_state(self.codec);state['simulation_ms']=1
        with self.assertRaises(ValueError):self.codec.encode_group_state(state,duration_ms=0)
        result=step_fixture(self.codec);result['neural']['simulation_ms']=99
        with self.assertRaises(ValueError):self.codec.encode_step(result)
        result=step_fixture(self.codec);result['neural']['neurons']=1000
        with self.assertRaises(ValueError):self.codec.encode_step(result)
        result=step_fixture(self.codec);result['neural'].pop('spikes_this_step')
        with self.assertRaises(ValueError):self.codec.encode_step(result)

    def test_invalid_measurements_fail_without_clipping_or_imputation(self):
        for key,value in [('voltage_mean',float('nan')),('voltage_min',-40),('spikes',-1),('spikes',.5)]:
            state=zero_state(self.codec);state[key][0]=value
            with self.assertRaises(ValueError):self.codec.encode_group_state(state,duration_ms=10)

    def test_numpy_output_bound_and_materialization(self):
        logits=np.linspace(-5,5,47,dtype=np.float32)
        currents=self.codec.currents_from_logits(logits)
        self.assertTrue(np.all(np.abs(currents)<=30))
        self.assertLess(currents[0],0);self.assertGreater(currents[-1],0)
        spec=self.codec.stimulation(currents)
        self.assertEqual(list(spec[0]),['macro_group_currents_mv'])
        self.assertEqual(len(spec[0]['macro_group_currents_mv']),47)
        for bad in [np.zeros(46),np.full(47,31.),np.full(47,np.nan),np.ones(47,dtype=bool)]:
            with self.assertRaises(ValueError):self.codec.stimulation(bad)

    def test_torch_decode_preserves_coupler_gradient(self):
        import torch
        torch.manual_seed(41)
        coupler=torch.nn.Linear(8,47)
        hidden=torch.arange(8,dtype=torch.float32).div(8).requires_grad_()
        currents=self.codec.currents_from_logits(coupler(hidden))
        loss=currents.square().mean();loss.backward()
        self.assertTrue(torch.isfinite(coupler.weight.grad).all())
        self.assertGreater(float(coupler.weight.grad.abs().sum()),0)
        self.assertGreater(float(hidden.grad.abs().sum()),0)
        self.assertTrue(currents.requires_grad)
        self.assertEqual(len(self.codec.stimulation(currents)[0]['macro_group_currents_mv']),47)

    def test_destination_identity_checked_before_write(self):
        codec=self.codec
        class Brain:
            info={'neurons':NEURON_COVERAGE,'edges':GRAPH_EDGES}
            def __init__(self):self.calls=[]
            def group_state(self):return zero_state(codec)
            def step(self,rgb,**kwargs):self.calls.append(kwargs);return kwargs
        brain=Brain();pixels=np.zeros((8,8,3),dtype=np.uint8)
        out=codec.step_brain(brain,pixels,np.ones(47),duration_ms=28.6)
        self.assertTrue(out['include_group_stats'])
        self.assertEqual(out['general_stimulation'][0]['macro_group_currents_mv'],[1.]*47)
        bad=Brain();bad.info={'neurons':1000,'edges':GRAPH_EDGES}
        with self.assertRaises(ValueError):codec.step_brain(bad,pixels,np.ones(47))
        self.assertEqual(bad.calls,[])

if __name__=='__main__':unittest.main()
