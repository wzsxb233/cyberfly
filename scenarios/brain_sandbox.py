"""An embodied-free, genuine full-connectome stimulus-control experiment."""
from __future__ import annotations

from copy import deepcopy
import math

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image, ImageDraw

from connectome_adapter import ConnectomeClient
from connectome_adapter.ports import validate_neural_drive, validate_general_stimulation


PORT_NAMES = ("retina_left", "retina_right", "sugar")
BRAIN_SANDBOX_SPEC = {
    "name": "brain_sandbox", "title": "无身体：真实果蝇脑实验",
    "observation_kind": "vector", "observation_shape": [12], "policy_type": "MlpPolicy",
    "action_kind": "continuous", "action_shape": [3], "action_labels": list(PORT_NAMES),
    "render": "rgb_array", "render_shape": [480, 640, 3],
    "engine": "MaleCNS full graph / DOOMFLY v6 native neural kernel; no physical body",
    "neural_input": True, "embodiment": "none",
    "control_modes": ["manual", "policy", "llm_brain"], "supports_demonstration": True,
    "task_updates": {"neural_drive": "dict retina_left/retina_right/sugar, each 0..1; semantic context consumed by the selected controller",
                     "general_stimulation": "Explicit actual group or neuron IDs with bounded engineering currents; all-neuron groups available",
                     "target_rate_hz": "finite full-population mean firing-rate target within 0.1..20 Hz",
                     "step_limit": "integer from 1 to 1000"},
    "limitations": ["No body, locomotion, game, natural reward, or innate language understanding",
                    "Reward measures an engineered population-rate tracking task with a stimulation cost",
                    "PPO/BC trains the three-dimensional input adapter; optional v6 synaptic plasticity is separately controlled",
                    "A black RGB field accompanies direct currents; metric graphics are observations for the user, not fabricated neural anatomy"],
}


class BrainSandboxScenario(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 35}

    def __init__(self, render_mode="rgb_array", max_episode_steps=64, neural_duration_ms=28.6,
                 target_rate_hz=2.0, learning=False, brain_checkpoint=None, brain_timeout=180,
                 visual_input_enabled=True):
        super().__init__()
        if render_mode not in (None, "rgb_array"):
            raise ValueError("Only rgb_array rendering is supported")
        if type(max_episode_steps) is not int or not 1 <= max_episode_steps <= 1000:
            raise ValueError("max_episode_steps must be an integer from 1 to 1000")
        if type(neural_duration_ms) not in (int, float) or not math.isfinite(neural_duration_ms) or not 0.1 <= neural_duration_ms <= 200:
            raise ValueError("neural_duration_ms must be within 0.1..200 ms")
        if type(learning) is not bool:
            raise ValueError("learning must be bool")
        self.render_mode, self.max_episode_steps = render_mode, max_episode_steps
        self.neural_duration_ms, self.learning = float(neural_duration_ms), learning
        self.brain_checkpoint, self.brain_timeout = brain_checkpoint, brain_timeout
        self.brain = None
        self.semantic_drive = validate_neural_drive()
        self.general_stimulation = []
        self.set_visual_input_enabled(visual_input_enabled)
        self.target_rate_hz = 2.0
        self.apply_task({"target_rate_hz": target_rate_hz})
        self.action_space = spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(-10, 10, shape=(12,), dtype=np.float32)
        self.scenario_spec = deepcopy(BRAIN_SANDBOX_SPEC)
        self._blank_rgb = np.zeros((84, 84, 3), dtype=np.uint8)
        self._started = self._done = self._closed = self._pending_aversive = False
        self._steps, self._return = 0, 0.0
        self._last_neural, self._last_info, self._input_ports = None, {}, {}
        self._applied_drive = validate_neural_drive()
        self._obs = np.zeros(12, dtype=np.float32)

    def _ensure_brain(self):
        if self.brain is None:
            self.brain = ConnectomeClient(learning=self.learning, timeout=self.brain_timeout,
                                          checkpoint=self.brain_checkpoint)
            self.brain.set_learning(self.learning)
            self._input_ports = {name: {key: value for key, value in spec.items() if key != "neuron_ids"}
                                 for name, spec in self.brain.info["input_ports"].items()}

    def _update(self, result):
        self._last_neural = result
        self._applied_drive = result["neural_drive"]["normalized"].copy()
        native, neural = result["native_action"], result["neural"]
        # Raw interval counts per cell match the embodied adapter's telemetry.
        telemetry = [neural["spikes_this_step"] / neural["neurons"]]
        telemetry += [result["neural_drive"]["port_spikes"][name] / self._input_ports[name]["count"] for name in PORT_NAMES]
        telemetry += [native["turn"] / 6.0, native["forward"] / 10.0 - 1.0,
                      float(native["attack"]), float(self.learning)]
        values = telemetry + [self.semantic_drive[name] for name in PORT_NAMES] + [self.target_rate_hz / 20.0]
        self._obs = np.clip(values, -10, 10).astype(np.float32)
        if not self.observation_space.contains(self._obs) or not np.isfinite(self._obs).all():
            raise RuntimeError("Actual neural observation is invalid")

    def _mean_rate(self):
        neural = self._last_neural["neural"]
        return neural["spikes_this_step"] / neural["neurons"] / (neural["duration_ms"] / 1000.0)

    def _info(self, reward=0.0, *, truncated=False):
        rate = self._mean_rate()
        return {"scenario": "brain_sandbox", "embodiment": "none", "step": self._steps,
                "step_limit": self.max_episode_steps, "population_mean_rate_hz": rate,
                "target_rate_hz": self.target_rate_hz, "absolute_rate_error_hz": abs(rate - self.target_rate_hz),
                "episode_reward": self._return, "last_reward": float(reward),
                "is_success": False, "terminated": False, "truncated": truncated,
                "neural_drive": self.semantic_drive.copy(), "applied_neural_drive": self._applied_drive.copy(),
                "neural": self._last_neural, "input_ports": self._input_ports,
                "general_stimulation": deepcopy(self.general_stimulation),
                "group_state": deepcopy(self._last_neural.get("group_state")),
                "reward_definition": "-abs(measured_population_hz-target_hz)/max(target_hz,1) - 0.01*mean(normalized_current**2)",
                "reward_interpretation": "Engineered neural-rate tracking assay; no natural fly preference or task mastery claim",
                "learning": self.learning}

    def reset(self, *, seed=None, options=None):
        if self._closed:
            raise RuntimeError("This brain sandbox has been closed")
        super().reset(seed=seed)
        if options:
            self.apply_task(options)
        self._ensure_brain()
        self.brain.reset(preserve_weights=True)
        self.general_stimulation = [item for item in self.general_stimulation if "neuron_currents_file" not in item]
        self._steps, self._return, self._pending_aversive = 0, 0.0, False
        self._started, self._done = True, False
        # A real initial neural interval supplies reset observation; no fake zero activity.
        self._update(self.brain.step(self._blank_rgb, duration_ms=self.neural_duration_ms,
                                    visual_input_enabled=self.visual_input_enabled, include_group_stats=True))
        self._last_info = self._info()
        return self._obs.copy(), deepcopy(self._last_info)

    def step(self, action):
        if not self._started or self._done:
            raise RuntimeError("Call reset() before stepping a new or finished episode")
        action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action) or not np.isfinite(action).all():
            raise ValueError("Action must be three finite amplitudes within [-1,1]")
        normalized = (action.astype(np.float64) + 1.0) / 2.0
        drive = {name: float(normalized[index]) for index, name in enumerate(PORT_NAMES)}
        stimulation = self.general_stimulation
        self.general_stimulation = [item for item in stimulation if "neuron_currents_file" not in item]
        result = self.brain.step(self._blank_rgb, duration_ms=self.neural_duration_ms,
                                 neural_drive=drive, reward_event={"aversive": self._pending_aversive},
                                 general_stimulation=stimulation, visual_input_enabled=self.visual_input_enabled,
                                 include_group_stats=True)
        self._pending_aversive = False
        self._update(result)
        reward = -abs(self._mean_rate() - self.target_rate_hz) / max(self.target_rate_hz, 1.0) - 0.01 * float(np.mean(normalized ** 2))
        self._steps += 1
        self._return += reward
        self._done = self._steps >= self.max_episode_steps
        self._last_info = self._info(reward, truncated=self._done)
        return self._obs.copy(), float(reward), False, self._done, deepcopy(self._last_info)

    def set_learning(self, enabled):
        if type(enabled) is not bool:
            raise ValueError("enabled must be bool")
        self.learning = enabled
        if self.brain is not None:
            self.brain.set_learning(enabled)
        if self._last_neural:
            self._last_neural["learning"]["enabled"] = enabled
            self._update(self._last_neural)
            self._last_info = self._info(truncated=self._done)
        return {"learning": enabled, "initialized": self.brain is not None}

    def queue_aversive(self):
        if not self._started or self._done or not self.learning:
            raise ValueError("Aversive input requires a running episode with synaptic learning enabled")
        self._pending_aversive = True
        return {"queued": True, "source": "engineered_external_event", "duration_ms": 200}

    @property
    def pending_aversive(self):
        return self._pending_aversive

    def set_general_stimulation(self, value):
        self.general_stimulation = validate_general_stimulation(value)
        return {"general_stimulation": deepcopy(self.general_stimulation), "delivery": "next step; actual targets validated by the live worker"}

    def set_macro_group_currents(self, currents_mv):
        """Replace additional stimuli with one signed 47-group current vector."""
        if currents_mv is None:
            return self.set_general_stimulation([])
        values = np.asarray(currents_mv)
        if values.dtype.kind not in "fiu" or values.shape != (47,) or not np.isfinite(values).all() or np.any(np.abs(values) > 30):
            raise ValueError("Macro currents require 47 finite values within [-30,30] mV-equivalent.")
        return self.set_general_stimulation([{"macro_group_currents_mv": values.astype(float).tolist()}])

    def set_neuron_currents(self, descriptor, *, preserve_existing=True):
        if type(preserve_existing) is not bool:
            raise ValueError("preserve_existing must be bool")
        stimuli = [item for item in self.general_stimulation if "neuron_currents_file" not in item] if preserve_existing else []
        if descriptor is not None:
            stimuli = [*stimuli, {"neuron_currents_file": descriptor}]
        return self.set_general_stimulation(stimuli)

    def append_neuron_currents(self, descriptor):
        return self.set_neuron_currents(descriptor, preserve_existing=True)

    def shared_passthrough_action(self):
        drive = validate_neural_drive(self.semantic_drive)
        return np.asarray([2 * drive[name] - 1 for name in PORT_NAMES], dtype=np.float32)

    def set_visual_input_enabled(self, enabled):
        if type(enabled) is not bool:
            raise ValueError("visual_input_enabled must be bool")
        self.visual_input_enabled = enabled
        return {"visual_input_enabled": enabled}

    def set_motor_readout(self, checkpoint):
        if checkpoint is not None:
            raise ValueError("A bodyless brain has no motor action destination")
        return {"mode": "none", "reason": "No physical body in this scenario"}

    def apply_task(self, task):
        if not isinstance(task, dict) or not task or set(task) - {"neural_drive", "general_stimulation", "target_rate_hz", "step_limit"}:
            raise ValueError("brain_sandbox task supports neural_drive, general_stimulation, target_rate_hz, step_limit")
        drive = validate_neural_drive(task["neural_drive"]) if "neural_drive" in task else None
        general = validate_general_stimulation(task["general_stimulation"]) if "general_stimulation" in task else None
        target = task.get("target_rate_hz", self.target_rate_hz)
        if type(target) not in (int, float) or not math.isfinite(target) or not 0.1 <= target <= 20:
            raise ValueError("target_rate_hz must be finite within 0.1..20")
        limit = task.get("step_limit", self.max_episode_steps)
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("step_limit must be an integer from 1 to 1000")
        if drive is not None:
            self.semantic_drive = drive
        if general is not None:
            self.general_stimulation = general
        self.target_rate_hz, self.max_episode_steps = float(target), limit
        if getattr(self, "_last_neural", None):
            self._update(self._last_neural)
            self._last_info = self._info(truncated=self._done)
        return {"applied": deepcopy(task), "persists_across_resets": True,
                "neural_drive_semantics": "Context only; the selected manual, LLM or learned controller supplies the actual step action"}

    def task_state(self):
        return {"scenario": "brain_sandbox", "embodiment": "none",
                "phase": "finished" if self._done else "running" if self._started else "not_reset",
                **deepcopy(self._last_info), "neural_drive": self.semantic_drive.copy(),
                "applied_neural_drive": self._applied_drive.copy(), "target_rate_hz": self.target_rate_hz,
                "step_limit": self.max_episode_steps, "learning": self.learning,
                "neural": deepcopy(self._last_neural), "input_ports": deepcopy(self._input_ports),
                "general_stimulation": deepcopy(self.general_stimulation),
                "visual_input_enabled": self.visual_input_enabled,
                "group_state": deepcopy(self._last_neural.get("group_state")) if self._last_neural else None}

    def observe(self):
        if not self._started:
            raise RuntimeError("Call reset() before observe()")
        return self._obs.copy()

    def shared_input_rgb(self):
        """The actual external visual field is black; the activity chart is a separate display."""
        if not self._started or self._done:
            raise RuntimeError("Shared input requires an active, unfinished episode.")
        return self._blank_rgb.copy()

    def shared_input_metadata(self):
        return {"image_source": "no_external_visual_input", "shared_with_brain": True,
                "note": "Black RGB field; rendered group-activity charts are not visual input to the brain."}

    def render(self):
        if not self._started:
            raise RuntimeError("Call reset() before render()")
        image = Image.new("RGB", (640, 480), "#101923")
        draw = ImageDraw.Draw(image)
        draw.text((24, 18), "CYBERFLY / REAL CONNECTOME / NO BODY", fill="#e2e9ef")
        neural = self._last_neural["neural"]
        draw.text((24, 45), f"{neural['neurons']:,} neurons   {neural['edges']:,} edges", fill="#93b7cd")
        draw.text((24, 69), f"Neural time {neural['simulation_ms']:.1f} ms | interval spikes {neural['spikes_this_step']:,}", fill="#93b7cd")
        draw.text((24, 104), f"Measured mean {self._mean_rate():.3f} Hz | target {self.target_rate_hz:.3f} Hz", fill="#71deae")
        for index, name in enumerate(PORT_NAMES):
            y = 150 + 70 * index
            drive = self._applied_drive[name]
            spikes = self._last_neural["neural_drive"]["port_spikes"][name]
            count = self._input_ports[name]["count"]
            draw.text((24, y), f"{name} | {count} identified neurons | {spikes} actual spikes", fill="#d2dce4")
            draw.rectangle((24, y + 24, 610, y + 40), fill="#223442")
            if drive > 0:
                draw.rectangle((24, y + 24, 24 + int(586 * drive), y + 40), fill="#4fa3cf")
        draw.text((24, 379), f"Step {self._steps}/{self.max_episode_steps} | plasticity {'ON' if self.learning else 'FROZEN'}", fill="#e2e9ef")
        draw.text((24, 405), "Reward: engineered rate tracking and stimulation cost", fill="#a9b6c0")
        draw.text((24, 431), "Measured indicators; no simulated body or invented neuron animation", fill="#a9b6c0")
        return np.asarray(image).copy()

    def close(self):
        if not self._closed:
            if self.brain is not None:
                self.brain.close()
            self._closed = True
