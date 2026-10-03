"""Task semantics are checked with real MuJoCo contacts and real camera output."""
import unittest
from unittest.mock import patch
import numpy as np
from scenarios import create_scenario

class FlyTaskTests(unittest.TestCase):
    def test_food_waypoints_follow_success_requires_real_steps(self):
        cases=[('fly_food',{'goal_mm':[.7,0]},1),
               ('fly_waypoints',{'waypoints_mm':[[.7,0],[.9,0]]},2),
               ('fly_follow',{'goal_mm':[.7,0],'target_speed_mm_s':0,'follow_steps':2},2)]
        for name,kwargs,steps in cases:
            with self.subTest(name=name):
                env=create_scenario({'name':name,'env_kwargs':{'max_episode_steps':8,**kwargs}})
                try:
                    obs,_=env.reset(seed=11);before=env.sim.mj_data.qpos.copy()
                    for _ in range(steps):obs,reward,term,trunc,info=env.step(np.zeros(2,dtype=np.float32))
                    self.assertTrue(term);self.assertTrue(info['is_success']);self.assertFalse(trunc)
                    self.assertTrue(env.observation_space.contains(obs))
                    self.assertFalse(np.array_equal(before,env.sim.mj_data.qpos))
                finally:env.close()

    def test_danger_zone_failure_overrides_nearby_goal(self):
        env=create_scenario({'name':'fly_hazard','env_kwargs':{'goal_mm':[.7,0],'hazards_mm':[[.6,0,1]]}})
        try:
            env.reset(seed=11)
            _,reward,term,trunc,info=env.step(np.zeros(2,dtype=np.float32))
            self.assertTrue(term);self.assertFalse(trunc);self.assertFalse(info['is_success'])
            self.assertEqual(info['failure_reason'],'danger_zone');self.assertLess(reward,0)
            self.assertEqual(env.sim.mj_model.geom_contype[env._task_geoms['task_hazard_0']],0)
        finally:env.close()

    def test_task_updates_are_staged_and_seeded_target_moves(self):
        env=create_scenario({'name':'fly_follow','env_kwargs':{'randomization':False}})
        try:
            env.reset(seed=11);position=env.sim.mj_data.qpos.copy();goal=env.goal.copy()
            self.assertTrue(env.apply_task({'difficulty':2})['requires_reset'])
            self.assertTrue(np.array_equal(position,env.sim.mj_data.qpos))
            self.assertEqual(env.task_state()['difficulty'],0)
            env.reset(seed=11);self.assertEqual(env.task_state()['difficulty'],2)
            goal=env.goal.copy();env.step(np.zeros(2,dtype=np.float32))
            self.assertFalse(np.array_equal(goal,env.goal))
            env.reset(seed=11);self.assertTrue(np.array_equal(goal,env.goal))
        finally:env.close()

    def test_visible_feedback_has_no_forces_or_body_teleport_and_expires(self):
        env=create_scenario({'name':'fly_food'})
        try:
            env.reset(seed=11);before=env.render();position=env.sim.mj_data.qpos.copy()
            with patch('rl.env.time.monotonic',return_value=100):
                env.show_feedback('aversive',1,1000);during=env.render()
                self.assertFalse(np.array_equal(before,during))
                self.assertTrue(np.array_equal(position,env.sim.mj_data.qpos))
                for geom in env._feedback_geom_ids.values():
                    self.assertEqual(env.sim.mj_model.geom_contype[geom],0)
                    self.assertEqual(env.sim.mj_model.geom_conaffinity[geom],0)
            with patch('rl.env.time.monotonic',return_value=102):
                env.render();self.assertIsNone(env.feedback_visual_state())
                self.assertTrue(all(env.sim.mj_model.geom_rgba[g,3]==0 for g in env._feedback_geom_ids.values()))
        finally:env.close()

if __name__=='__main__':unittest.main()
