"""Minimal adapter for an existing trusted Gymnasium environment."""

from copy import deepcopy

import gymnasium as gym
import numpy as np


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    return value


class GymScenarioAdapter(gym.Wrapper):
    """Give a Gymnasium environment the platform's minimal scenario contract.

    Task changes are rejected by default. Subclass and validate apply_task() to
    expose real environment-specific controls. Rendering is the wrapped
    environment's own renderer; this adapter does not synthesize 3D scenes.
    """

    def __init__(self, env, spec):
        super().__init__(env)
        self.scenario_spec = deepcopy(spec)
        self._steps = 0
        self._return = 0.0
        self._last_info = {}
        self._phase = "not_reset"
        self._observation = None

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._steps, self._return, self._phase = 0, 0.0, "running"
        self._last_info = _json_value(info)
        self._observation = deepcopy(obs)
        return obs, info

    def step(self, action):
        if self._phase != "running":
            raise RuntimeError("Call reset() before stepping a new or finished episode.")
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._steps += 1
        self._return += float(reward)
        self._last_info = _json_value(info)
        self._observation = deepcopy(obs)
        if terminated or truncated:
            self._phase = "finished"
        return obs, reward, terminated, truncated, info

    def observe(self):
        if self._phase == "not_reset":
            raise RuntimeError("Call reset() before observe().")
        return deepcopy(self._observation)

    def task_state(self):
        return {"scenario": self.scenario_spec["name"], "phase": self._phase,
                "step": self._steps, "episode_reward": self._return,
                "info": deepcopy(self._last_info)}

    def apply_task(self, task):
        raise ValueError(f"Scenario {self.scenario_spec['name']} does not support dynamic task updates.")
