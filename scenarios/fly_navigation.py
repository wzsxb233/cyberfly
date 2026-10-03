from __future__ import annotations

from copy import deepcopy

import gymnasium as gym
import numpy as np

from rl.env import FlyNavigationEnv
from .registry import FLY_SPEC


class FlyNavigationScenario(gym.Wrapper):
    """Platform adapter; the already-running RL environment remains unmodified."""

    def __init__(self, goal_mm=None, **env_kwargs):
        super().__init__(FlyNavigationEnv(**env_kwargs))
        self.scenario_spec = deepcopy(FLY_SPEC)
        self._fixed_goal = None
        self._started = False
        self._done = False
        self._last_info = {}
        if goal_mm is not None:
            self.apply_task({"goal_mm": goal_mm})

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        if self._fixed_goal is not None:
            options.setdefault("goal_xy", self._fixed_goal)
        obs, self._last_info = self.env.reset(seed=seed, options=options)
        self._started = True
        self._done = False
        return obs, deepcopy(self._last_info)

    def step(self, action):
        if not self._started or self._done:
            raise RuntimeError("Call reset() before stepping a new or finished episode.")
        obs, reward, terminated, truncated, self._last_info = self.env.step(action)
        self._done = bool(terminated or truncated)
        return obs, reward, terminated, truncated, deepcopy(self._last_info)

    def apply_task(self, task):
        if not isinstance(task, dict) or not task:
            raise ValueError("Task must be a nonempty dictionary.")
        unknown = set(task) - {"goal_mm"}
        if unknown:
            raise ValueError(f"fly_navigation supports only goal_mm, not {sorted(unknown)}.")
        goal = np.asarray(task["goal_mm"], dtype=np.float64)
        if goal.shape != (2,) or not np.isfinite(goal).all() or np.abs(goal).max() > 100:
            raise ValueError("goal_mm must contain two finite values within ±100 mm.")
        self._fixed_goal = goal.tolist()
        if self._started:
            self.env.set_goal(goal)
            self._last_info = self.env._info()
        return {"applied": {"goal_mm": self._fixed_goal}, "persists_across_resets": True}

    def task_state(self):
        return {"scenario": "fly_navigation", "phase": "finished" if self._done else "running" if self._started else "not_reset",
                "fixed_goal_mm": deepcopy(self._fixed_goal), **deepcopy(self._last_info),
                "feedback_visual": self.env.feedback_visual_state()}

    def observe(self):
        if not self._started:
            raise RuntimeError("Call reset() before observe().")
        return self.env.observe()

    def render(self):
        if not self._started:
            raise RuntimeError("Call reset() before render().")
        return self.env.render()

    def body_state(self):
        return self.env.body_state()

    def show_feedback(self, kind="aversive", intensity=1.0, duration_ms=800):
        return self.env.show_feedback(kind, intensity, duration_ms)
