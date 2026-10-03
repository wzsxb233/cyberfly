import json
from pathlib import Path
import tempfile
import unittest
from shared_io.model_link import ModelLink, finite_vector

class ProtocolTests(unittest.TestCase):
    def test_numeric_features_reject_invalid_values(self):
        for bad in ([0]*239, [0]*239+[True], [0]*239+[float('nan')], [0]*239+[float('inf')], [0]*239+[1001]):
            with self.assertRaises(ValueError): finite_vector(bad,240,'brain_features')
        self.assertEqual(finite_vector([0]*240,240,'brain_features'),[0]*240)

    def test_workspace_raw_media_boundary(self):
        link=ModelLink.__new__(ModelLink)
        req={'event_id':'test','text':'你好','brain_features':[0]*240}
        self.assertEqual(link._request(req)[5],['text'])
        with self.assertRaises(ValueError): link._request({**req,'image_path':'/etc/passwd'})

    def test_atomic_checkpoint_embedded_identity_and_tamper(self):
        import torch
        from torch import nn
        from safetensors.torch import load_file,save_file
        from shared_io.brain_codec import BrainFeatureCodec
        link=ModelLink.__new__(ModelLink)
        link.torch,link.codec=torch,BrainFeatureCodec()
        link.to_brain=nn.Sequential(nn.LayerNorm(4096),nn.Linear(4096,47))
        link.to_model=nn.Sequential(nn.Linear(240,128),nn.Tanh(),nn.Linear(128,16384))
        link.training_steps=2;link.generation=3;link.weights_manifest_sha='test-manifest'
        link.update_sha()
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'coupler.safetensors';link.save(path)
            original=link.coupler_sha
            link.generation=0;link.load(path);link.update_sha()
            self.assertEqual(link.coupler_sha,original);self.assertEqual(link.generation,3)
            data=load_file(str(path));save_file(data,str(path))
            with self.assertRaises(ValueError):link.load(path)

if __name__=='__main__':unittest.main()
