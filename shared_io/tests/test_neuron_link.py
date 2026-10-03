import json
from pathlib import Path
import unittest
import numpy as np
from shared_io.neuron_link import NeuronStateCodec,build_modules,N,IDS_SHA
ROOT=Path(__file__).resolve().parents[2]
DESCRIPTOR=ROOT/'artifacts/neuron-link-validation/source-state.json'

class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DESCRIPTOR.exists():raise unittest.SkipTest('Verified real full-neuron fixture not yet prepared')
        cls.descriptor=json.loads(DESCRIPTOR.read_text())

    def test_actual_id_order_and_scope(self):
        codec=NeuronStateCodec();features,info=codec.read(self.descriptor)
        self.assertEqual(features.shape,(N*2,))
        self.assertEqual(int(codec.read_mask.sum()),2129)
        self.assertEqual(int(codec.write_mask.sum()),17336)
        self.assertEqual(codec.ids.dtype,np.dtype('int64'))
        for bad in ({**self.descriptor,'ids_sha256':'0'*64},{**self.descriptor,'sha256':'0'*64},{**self.descriptor,'duration_ms':99}):
            with self.assertRaises(ValueError):codec.read(bad)

    def test_independent_rows_and_single_neuron_mask(self):
        import torch
        codec=NeuronStateCodec();features,_=codec.read(self.descriptor)
        to_brain,to_model=build_modules(torch,'sensory_motor',*codec.masks(),device='cpu')
        torch.manual_seed(20)
        with torch.no_grad():
            current=to_brain(torch.randn(4096)).numpy()
            self.assertGreater(np.unique(current[codec.write_mask]).size,17000)
            self.assertTrue(np.all(current[~codec.write_mask].view(np.uint32)==0))
            x=torch.from_numpy(features).unsqueeze(0);original=to_model(x)
            changed=x.clone();changed[0,int(np.flatnonzero(codec.read_mask)[0])]+=1
            self.assertGreater(float((to_model(changed)-original).norm()),0)
            blocked=x.clone();blocked[0,int(np.flatnonzero(~codec.read_mask)[0])]+=1
            self.assertTrue(torch.equal(to_model(blocked),original))

if __name__=='__main__':unittest.main()
