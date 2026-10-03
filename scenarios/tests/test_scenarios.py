"""Real-engine integration checks; no fake physics/game backends."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import gymnasium as gym
import imageio.v3 as iio
import numpy as np

from scenarios import create_scenario, list_scenarios, register_scenario, load_plugins, GymScenarioAdapter


class ScenarioIntegrationTests(unittest.TestCase):
    output = Path(__file__).resolve().parents[2] / "test_artifacts"

    @classmethod
    def setUpClass(cls):
        cls.output.mkdir(exist_ok=True)

    def test_01_registry_and_explicit_validation(self):
        specs = list_scenarios()
        self.assertTrue({"fly_navigation", "doom_basic"}.issubset({s["name"] for s in specs}))
        specs[0]["name"] = "modified"
        self.assertNotEqual(list_scenarios()[0]["name"], "modified")
        with self.assertRaises(ValueError):
            create_scenario({"name": "module.to.execute"})
        with self.assertRaises(ValueError):
            create_scenario({"name": "doom_basic", "console_command": "anything"})

    def test_02_real_fly_reset_goal_step_and_render(self):
        env = create_scenario({"name": "fly_navigation", "env_kwargs": {"max_episode_steps": 3}})
        try:
            initial, _ = env.reset(seed=123)
            qpos = env.unwrapped.sim.mj_data.qpos.copy()
            env.apply_task({"goal_mm": [5, 2]})
            self.assertTrue(np.array_equal(qpos, env.unwrapped.sim.mj_data.qpos))
            self.assertFalse(np.array_equal(initial, env.observe()))
            obs, _, term, trunc, info = env.step(np.zeros(2, dtype=np.float32))
            self.assertTrue(env.observation_space.contains(obs))
            self.assertFalse(np.array_equal(qpos, env.unwrapped.sim.mj_data.qpos))
            frame = env.render()
            self.assertEqual(frame.shape, (480, 640, 3))
            self.assertGreater(float(frame.std()), 5)
            iio.imwrite(self.output / "fly_navigation.png", frame)
            env.reset(seed=123)
            self.assertEqual(env.task_state()["goal_mm"], [5.0, 2.0])
            with self.assertRaises(ValueError):
                env.apply_task({"goal_mm": [float("nan"), 1]})
        finally:
            env.close()

    def test_03_real_doom_pixels_actions_seed_and_render(self):
        env = create_scenario({"name": "doom_basic"})
        try:
            before, initial = env.reset(seed=123)
            first, _, _, _, info = env.step(0)
            self.assertEqual(first.shape, (3, 84, 84))
            self.assertEqual(first.dtype, np.uint8)
            self.assertTrue(env.observation_space.contains(first))
            self.assertEqual(info["game_tic"] - initial["game_tic"], 4)
            self.assertGreater(float(first.std()), 5)
            self.assertFalse(np.array_equal(before, first))
            again, _ = env.reset(seed=123)
            repeat, _, _, _, _ = env.step(0)
            self.assertTrue(np.array_equal(before, again))
            self.assertTrue(np.array_equal(first, repeat))
            frame = env.render()
            self.assertEqual(frame.shape, (240, 320, 3))
            iio.imwrite(self.output / "doom_basic.png", frame)
            json.dumps(env.task_state())
            with self.assertRaises(ValueError):
                env.step([1, 0, 0])
        finally:
            env.close()

    def test_04_doom_time_limit_and_task_rejection(self):
        env = create_scenario({"name": "doom_basic", "task": {"step_limit": 1}})
        try:
            env.reset(seed=456)
            _, _, terminated, truncated, info = env.step(0)
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertFalse(info["is_success"])
            self.assertEqual(env.task_state()["phase"], "finished")
            with self.assertRaises(RuntimeError):
                env.step(0)
            with self.assertRaises(ValueError):
                env.apply_task({"goal_mm": [4, 2]})
            with self.assertRaises(ValueError):
                env.apply_task({"target_kills": 2})
            self.assertEqual(env.max_episode_steps, 1)
        finally:
            env.close()

    def test_05_custom_gym_registration_and_plugin_entrypoint(self):
        spec = {"name": "test_cartpole", "observation_kind": "vector", "policy_type": "MlpPolicy", "render": "rgb_array"}
        def factory(kwargs):
            return GymScenarioAdapter(gym.make("CartPole-v1", **kwargs), spec)
        register_scenario("test_cartpole", factory, spec)
        with self.assertRaises(ValueError):
            register_scenario("test_cartpole", factory, spec)
        env = create_scenario({"name": "test_cartpole"})
        try:
            obs, _ = env.reset(seed=0)
            self.assertEqual(obs.shape, (4,))
            env.step(1)
            self.assertEqual(env.task_state()["step"], 1)
            with self.assertRaises(ValueError):
                env.apply_task({"goal_mm": [1, 1]})
        finally:
            env.close()
        plugin_spec = {**spec, "name": "test_plugin_cartpole"}
        class EntryPoint:
            name = "test_plugin"
            def load(self):
                return lambda register: register("test_plugin_cartpole", lambda kwargs: GymScenarioAdapter(gym.make("CartPole-v1", **kwargs), plugin_spec), plugin_spec)
        with patch("scenarios.registry.entry_points", return_value=[EntryPoint()]):
            load_plugins(["test_plugin"])
            env = create_scenario({"name": "test_plugin_cartpole", "plugins": ["test_plugin"]})
            try:
                self.assertEqual(env.reset(seed=1)[0].shape, (4,))
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main()
