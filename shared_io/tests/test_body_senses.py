"""Actual MuJoCo sampling checks and numerical encoder integrity tests."""
from copy import deepcopy
import importlib
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from rl.env import FlyNavigationEnv
from shared_io.body_senses import BodySenseAdapter, BodySenseCodec, array_sha


class BodySenseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.env=FlyNavigationEnv();cls.env.reset(seed=20260913)
        cls.packet=cls.env.body_sensory_packet()

    @classmethod
    def tearDownClass(cls):cls.env.close()

    def test_complete_raw_state_and_nonmutating_sampling(self):
        d=self.env.sim.mj_data
        before=(array_sha(d.qpos),array_sha(d.qvel),float(d.time))
        p=self.env.body_sensory_packet()
        self.assertTrue(np.array_equal(p['qpos'],d.qpos))
        self.assertTrue(np.array_equal(p['qvel'],d.qvel))
        self.assertEqual(len(p['contacts']),d.ncon)
        self.assertEqual(p['layout']['nsensor'],0)
        self.assertEqual(p['sensordata'],[])
        self.assertEqual(before,(array_sha(d.qpos),array_sha(d.qvel),float(d.time)))

    def test_contact_force_matches_official_flygym_reader(self):
        segments=self.env.fly.get_bodysegs_order()
        segment_forces=self.env.sim.get_bodysegment_contact_forces(self.env.fly.name,segments,ground_only=False)
        expected=np.zeros((self.env.sim.mj_model.nbody,3))
        # Static fused segments can have no named MuJoCo body (-1 in FlyGym's
        # body-name map). Associate each real geometry with its compiled body.
        geoms=self.env.sim._internal_geomid_by_bodyseg_by_fly[self.env.fly.name]
        for segment,force in zip(segments,segment_forces):
            expected[self.env.sim.mj_model.geom_bodyid[geoms[segment]]]+=force
        actual=np.asarray(self.packet['body_contact_force_world'])
        np.testing.assert_allclose(actual[1:],expected[1:],rtol=0,atol=1e-12)
        np.testing.assert_allclose(np.asarray(self.packet['body_contact_force_world']).sum(axis=0),0,rtol=0,atol=1e-12)

    def test_packet_integrity_and_layout_units(self):
        codec=BodySenseCodec(self.packet);x=codec.encode(self.packet)
        self.assertEqual(x.shape,(758,));self.assertEqual(x.dtype,np.float32)
        self.assertEqual(len(codec.dimensions),758)
        self.assertTrue(all('original_unit' in d and d['divide_by']>0 for d in codec.dimensions))
        altered=deepcopy(self.packet);altered['qpos'][0]+=.1
        with self.assertRaises(ValueError):codec.encode(altered)

    def test_real_data_fit_and_exact_reload(self):
        model=BodySenseAdapter(self.packet)
        target=np.full((1,6370),.08,dtype=np.float32)
        result=model.fit_supervised([self.packet],target,steps=2,label_source='Explicit constant engineering target for unit-test optimizer verification')
        self.assertGreater(result['changed_parameters'],0)
        self.assertNotEqual(result['initial_adapter_sha256'],result['final_adapter_sha256'])
        with tempfile.TemporaryDirectory() as path:
            model.save(path,self.packet);loaded=BodySenseAdapter.load(path)
            self.assertEqual(model.adapter_sha256(),loaded.adapter_sha256())
            x=torch.from_numpy(model.codec.encode(self.packet))
            self.assertTrue(torch.equal(model(x),loaded(x)))

    def test_codec_import_has_no_mujoco_dependency(self):
        import shared_io.body_senses as body_senses
        with patch.dict('sys.modules',{'mujoco':None}):
            reloaded=importlib.reload(body_senses)
            self.assertEqual(reloaded.BodySenseCodec(self.packet).feature_dim,758)


if __name__=='__main__':unittest.main()
