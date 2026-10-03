"""Real pixel-observation ViZDoom task using the package's official basic assets."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image
import vizdoom as vzd

from .registry import DOOM_SPEC


class DoomBasicScenario(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 9}
    ASSET_NAME = "basic"
    SCENARIO_SPEC = DOOM_SPEC
    BUTTON_NAMES = ("MOVE_LEFT", "MOVE_RIGHT", "ATTACK")
    DEFAULT_TARGET_KILLS = 1

    def __init__(self, render_mode="rgb_array", frame_skip=4, max_episode_steps=75, reward_scale=0.01,
                 target_kills=None, doom_skill=None):
        super().__init__()
        if render_mode not in (None, "rgb_array"):
            raise ValueError("Only rgb_array rendering is supported.")
        if type(frame_skip) is not int or not 1 <= frame_skip <= 16:
            raise ValueError("frame_skip must be an integer from 1 to 16.")
        if type(max_episode_steps) is not int or not 1 <= max_episode_steps <= 1000:
            raise ValueError("max_episode_steps must be an integer from 1 to 1000.")
        if not np.isfinite(reward_scale) or not 0 < reward_scale <= 10:
            raise ValueError("reward_scale must be finite and in (0, 10].")
        target_kills = self.DEFAULT_TARGET_KILLS if target_kills is None else target_kills
        if type(target_kills) is not int or not 1 <= target_kills <= (1 if self.ASSET_NAME == "basic" else 100):
            raise ValueError("target_kills must be 1 for basic, or 1..100 for defense scenarios.")
        if doom_skill is not None and (type(doom_skill) is not int or not 1 <= doom_skill <= 5):
            raise ValueError("doom_skill must be an integer from 1 to 5.")
        self.target_kills = target_kills
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.reward_scale = float(reward_scale)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(0, 255, shape=(3, 84, 84), dtype=np.uint8)
        self.scenario_spec = deepcopy(self.SCENARIO_SPEC)
        self.scenario_spec["frame_skip"] = frame_skip
        self.metadata = {**self.metadata, "render_fps": max(1, round(35 / frame_skip))}
        self._actions = np.eye(3, dtype=np.int32).tolist()
        self._temp_dir = tempfile.TemporaryDirectory(prefix="cyberfly-doom-")
        self.game = vzd.DoomGame()
        self._started = False
        self._closed = False
        self._done = False
        self._steps = 0
        self._return = 0.0
        self._last_info = {}
        self._rgb = np.zeros((240, 320, 3), dtype=np.uint8)
        self._obs = np.zeros((3, 84, 84), dtype=np.uint8)
        try:
            assets = Path(vzd.scenarios_path)
            self.game.load_config(str(assets / f"{self.ASSET_NAME}.cfg"))
            self.game.set_doom_scenario_path(str(assets / f"{self.ASSET_NAME}.wad"))
            if doom_skill is not None:
                self.game.set_doom_skill(doom_skill)
            self.game.set_doom_config_path(str(Path(self._temp_dir.name) / "vizdoom.ini"))
            self.game.set_window_visible(False)
            self.game.set_console_enabled(False)
            self.game.set_sound_enabled(False)
            self.game.set_mode(vzd.Mode.PLAYER)
            self.game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
            self.game.set_screen_format(vzd.ScreenFormat.RGB24)
            self.game.set_available_buttons([getattr(vzd.Button, name) for name in self.BUTTON_NAMES])
            self.game.set_available_game_variables([vzd.GameVariable.HEALTH, vzd.GameVariable.AMMO2, vzd.GameVariable.KILLCOUNT])
            # The wrapper supplies the time limit and correct Gymnasium truncation.
            self.game.set_episode_timeout(0)
            self.game.init()
        except BaseException:
            self.close()
            raise

    def _capture(self):
        state = self.game.get_state()
        if state is not None:
            self._rgb = np.asarray(state.screen_buffer, dtype=np.uint8).copy()
            if self._rgb.shape != (240, 320, 3):
                raise RuntimeError(f"Unexpected ViZDoom RGB shape: {self._rgb.shape}")
            resized = Image.fromarray(self._rgb).resize((84, 84), Image.Resampling.BILINEAR)
            self._obs = np.asarray(resized, dtype=np.uint8).transpose(2, 0, 1).copy()
        # ViZDoom exposes no GameState after natural termination. Preserve the
        # last real image for terminal rendering and terminal_observation.
        return self._obs.copy()

    def _info(self, *, raw_reward=0.0, success=False, terminated=False, truncated=False):
        return {
            "scenario": self.scenario_spec["name"], "step": self._steps,
            "game_tic": int(self.game.get_episode_time()),
            "health": float(self.game.get_game_variable(vzd.GameVariable.HEALTH)),
            "ammo": float(self.game.get_game_variable(vzd.GameVariable.AMMO2)),
            "kills": int(self.game.get_game_variable(vzd.GameVariable.KILLCOUNT)),
            "raw_reward": float(raw_reward), "episode_reward": self._return,
            "is_success": bool(success), "dead": bool(self.game.is_player_dead()),
            "terminated": bool(terminated), "truncated": bool(truncated),
            "step_limit": self.max_episode_steps, "target_kills": self.target_kills,
        }

    def reset(self, *, seed=None, options=None):
        if self._closed:
            raise RuntimeError("This Doom environment has been closed.")
        super().reset(seed=seed)
        if options:
            self.apply_task(options)
        self.game.set_seed(int(self.np_random.integers(0, 2**31 - 1)))
        self.game.new_episode()
        self._started, self._done = True, False
        self._steps, self._return = 0, 0.0
        obs = self._capture()
        self._last_info = self._info()
        return obs, deepcopy(self._last_info)

    def step(self, action):
        if not self._started or self._done:
            raise RuntimeError("Call reset() before stepping a new or finished episode.")
        if not self.action_space.contains(action):
            raise ValueError(f"Doom action must be an integer 0..2: {self.scenario_spec['action_labels']}.")
        raw_reward = float(self.game.make_action(self._actions[int(action)], self.frame_skip))
        reward = raw_reward * self.reward_scale
        self._steps += 1
        self._return += reward
        finished = self.game.is_episode_finished()
        dead = self.game.is_player_dead()
        kills = int(self.game.get_game_variable(vzd.GameVariable.KILLCOUNT))
        native_timeout = bool(self.game.is_episode_timeout_reached())
        success = kills >= self.target_kills and not dead
        terminated = bool(success or dead or (finished and not native_timeout))
        truncated = bool((self._steps >= self.max_episode_steps or native_timeout) and not terminated)
        self._done = terminated or truncated
        obs = self._capture()
        self._last_info = self._info(raw_reward=raw_reward, success=success, terminated=terminated, truncated=truncated)
        return obs, float(reward), terminated, truncated, deepcopy(self._last_info)

    def apply_task(self, task):
        if not isinstance(task, dict) or not task:
            raise ValueError("Task must be a nonempty dictionary.")
        unknown = set(task) - {"step_limit", "target_kills"}
        if unknown:
            raise ValueError(f"This Doom task supports only step_limit and target_kills, not {sorted(unknown)}.")
        if "target_kills" in task and (type(task["target_kills"]) is not int or not 1 <= task["target_kills"] <= (1 if self.ASSET_NAME == "basic" else 100)):
            raise ValueError("target_kills must be 1 for basic, or 1..100 for defense scenarios.")
        if "step_limit" in task:
            limit = task["step_limit"]
            if type(limit) is not int or not 1 <= limit <= 1000:
                raise ValueError("step_limit must be an integer from 1 to 1000.")
            self.max_episode_steps = limit
        self.target_kills = task.get("target_kills", self.target_kills)
        return {"applied": {"step_limit": self.max_episode_steps, "target_kills": self.target_kills}, "persists_across_resets": True}

    def task_state(self):
        return {"scenario": self.scenario_spec["name"], "phase": "finished" if self._done else "running" if self._started else "not_reset",
                **deepcopy(self._last_info), "step_limit": self.max_episode_steps, "target_kills": self.target_kills,
                "observation_source": "RGB pixels; telemetry is planner-only"}

    def observe(self):
        if not self._started:
            raise RuntimeError("Call reset() before observe().")
        return self._obs.copy()

    def render(self):
        if not self._started:
            raise RuntimeError("Call reset() before render().")
        return self._rgb.copy()

    def close(self):
        if not self._closed:
            self.game.close()
            self._temp_dir.cleanup()
            self._closed = True
