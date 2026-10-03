"""Neural-input RL: every policy action enters the actual full connectome first."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from connectome_adapter import ConnectomeClient
from connectome_adapter.ports import validate_neural_drive
from .fly_navigation import FlyNavigationScenario
from .neural_inputs import validate_general_stimulation


PORT_ORDER = ("retina_left", "retina_right", "sugar")
NEURAL_LINK_SPEC = {
    "name": "fly_neural_link", "title": "神经直连 · 三维果蝇",
    "observation_kind": "vector", "observation_shape": [33], "policy_type": "MlpPolicy",
    "action_kind": "continuous", "action_shape": [3], "action_labels": list(PORT_ORDER),
    "render": "rgb_array", "render_shape": [480, 640, 3],
    "engine": "Real MaleCNS full graph → engineered BCI decoder → MuJoCo NeuroMechFly",
    "neural_input": True, "embodiment": "physical_3d", "control_modes": ["manual", "policy", "llm_brain"],
    "supports_demonstration": True,
    "supports_3d_feedback": True,
    "task_updates": {"neural_drive": "retina_left, retina_right, sugar: normalized 0..1 semantic context",
                     "general_stimulation": "Fine-group, macro-group or original neuron-ID bounded currents; next neural step"},
    "limitations": ["PPO trains a neural-input adapter; synaptic plasticity is a separate optional mode",
                   "Body camera projection, sensory currents and motor decoder are engineering interfaces",
                   "Sugar input is not established positive reinforcement", "No automatic terminal punishment: explicit manual aversion only"],
}


def compact_description(description):
    """Preserve port identity/count/current scale without copying thousands of IDs."""
    return {key: deepcopy(value) for key, value in description.items() if key != "input_ports"} | {
        "input_ports": {name: {key: deepcopy(value) for key, value in port.items() if key != "neuron_ids"}
                        for name, port in description.get("input_ports", {}).items()}
    }


def normalize_context(value):
    if value is None:
        return validate_neural_drive()
    if isinstance(value, (list, tuple, np.ndarray)):
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (3,):
            raise ValueError("semantic_drive must have three normalized values.")
        value = dict(zip(PORT_ORDER, array.tolist()))
    return validate_neural_drive(value)


class FlyNeuralLink(gym.Env):
    """Body state + context -> policy -> currents -> brain -> decoder -> body.

    The action cannot directly access leg control, target position, or body
    velocity. semantic_drive appears in the observation as task context; it is
    not silently mixed into an action. A language-controller executes its plan
    by explicitly taking action = 2 * semantic_drive - 1.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}
    SCENARIO_SPEC = NEURAL_LINK_SPEC

    def __init__(self, *, semantic_drive=None, general_stimulation=None, brain_checkpoint=None, learning=False,
                 max_episode_steps=150, render_mode="rgb_array", visual_input_enabled=True,
                 motor_readout_checkpoint=None, **body_kwargs):
        super().__init__()
        if type(learning) is not bool:
            raise ValueError("learning must be a boolean.")
        self.scenario_spec = deepcopy(self.SCENARIO_SPEC)
        self.render_mode = render_mode
        self.action_space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
        self.observation_space = spaces.Box(-10.0, 10.0, tuple(self.scenario_spec["observation_shape"]), dtype=np.float32)
        self.semantic_drive = normalize_context(semantic_drive)
        self.general_stimulation = validate_general_stimulation(general_stimulation or [])
        self.set_visual_input_enabled(visual_input_enabled)
        self.motor_readout = None
        self._motor_trace = None
        self.group_state = {}
        self.applied_neural_drive = validate_neural_drive()
        self.learning = learning
        self._started = False
        self._done = False
        self._closed = False
        self._pending_aversive = False
        self._cancelled_aversive = 0
        self._last_reset_cancelled_event = False
        self.neural = None
        self._body_info = {}
        self._body_action = self._zero_body_action()
        self._brain_vector = np.zeros(8, dtype=np.float32)
        self._frame = None
        self._shared_input_rgb = None
        self._shared_body_packet = None
        self.brain = None
        self.body = self._make_body(max_episode_steps, render_mode, body_kwargs)
        try:
            self.brain = ConnectomeClient(learning=learning, checkpoint=brain_checkpoint, timeout=180)
            # Restoring a checkpoint may restore its old plasticity setting.
            # The explicit scenario configuration always controls this run.
            description = self.brain.set_learning(learning)
            self.brain_description = compact_description(description)
            if motor_readout_checkpoint is not None:
                self.set_motor_readout(motor_readout_checkpoint)
        except BaseException:
            self.close()
            raise

    def _make_body(self, max_episode_steps, render_mode, kwargs):
        return FlyNavigationScenario(max_episode_steps=max_episode_steps, render_mode=render_mode, **kwargs)

    def _zero_body_action(self):
        return np.zeros(2, dtype=np.float32)

    def _body_observation(self):
        return self.body.observe()

    def _body_interval_ms(self):
        return self.body.unwrapped.action_dt * 1000

    @property
    def pending_aversive(self):
        return self._pending_aversive

    def set_learning(self, enabled):
        if type(enabled) is not bool:
            raise ValueError("learning must be a boolean.")
        self.brain_description = compact_description(self.brain.set_learning(enabled))
        self.learning = enabled
        self._brain_vector[-1] = float(enabled)
        if not enabled and self._pending_aversive:
            self._pending_aversive = False
            self._cancelled_aversive += 1
        return {"brain_learning": enabled, "cancelled_manual_events": self._cancelled_aversive}

    def queue_aversive(self):
        """One explicit event, delivered on the next real neural interval.

        Terminal or pre-reset events are rejected, not silently dropped. A reset
        before delivery reports cancellation in task_state and the brain itself
        logs any already-scheduled pulse cancelled by its reset.
        """
        if not self.learning:
            raise ValueError("Enable neural plasticity before submitting an aversive event.")
        if not self._started or self._done:
            raise ValueError("Manual aversion requires an active, unfinished episode; reset first.")
        if self._pending_aversive:
            raise ValueError("A manual aversive event is already waiting for the next neural step.")
        self._pending_aversive = True
        return {"queued": True, "delivery": "next neural interval", "source": "explicit_manual_event"}

    def reset(self, *, seed=None, options=None):
        if self._closed:
            raise RuntimeError("This neural-input environment has been closed.")
        super().reset(seed=seed)
        self._last_reset_cancelled_event = self._pending_aversive
        self._cancelled_aversive += int(self._pending_aversive)
        self._pending_aversive = False
        self._shared_input_rgb = None
        self._shared_body_packet = None
        self.general_stimulation = [item for item in self.general_stimulation if "neuron_currents_file" not in item]
        self._motor_trace = None
        self.brain_description = compact_description(self.brain.reset(preserve_weights=True))
        self.group_state = self.brain.group_state()
        self.body.reset(seed=seed, options=options)
        self._body_info = self.body.task_state()
        self.neural = None
        self.applied_neural_drive = validate_neural_drive()
        self._body_action = self._zero_body_action()
        self._brain_vector = np.zeros(8, dtype=np.float32)
        self._brain_vector[-1] = float(self.learning)
        self._frame = self.body.render()
        self._started, self._done = True, False
        return self.observe(), self.task_state()

    @staticmethod
    def decode_body_action(native):
        """Fixed engineering decoder. Only actual brain readouts enter it."""
        turn = float(np.clip(float(native["turn"]) / 6.0, -1.0, 1.0))
        forward = float(np.clip(float(native["forward"]) / 10.0 - 1.0, -1.0, 1.0))
        return np.clip([forward - turn, forward + turn], -1.0, 1.0).astype(np.float32)

    def _telemetry_vector(self, summary):
        neural = summary["neural"]
        port_spikes = summary["neural_drive"]["port_spikes"]
        ports = self.brain_description["input_ports"]
        native = summary["native_action"]
        values = [neural["spikes_this_step"] / max(1, neural["neurons"])]
        values += [port_spikes[name] / max(1, ports[name]["count"]) for name in PORT_ORDER]
        values += [float(native["turn"]) / 6.0, float(native["forward"]) / 10.0 - 1.0,
                   float(native["attack"]), float(self.learning)]
        return np.clip(values, -10.0, 10.0).astype(np.float32)

    def step(self, action):
        if not self._started or self._done:
            raise RuntimeError("Call reset() before stepping a new or finished episode.")
        action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action):
            raise ValueError("Neural action must be three finite float values in [-1,1].")
        normalized = (action.astype(np.float64) + 1.0) / 2.0
        drive = dict(zip(PORT_ORDER, normalized.tolist()))
        # This is the only policy-to-body path. No direct action component,
        # semantic context or target coordinate reaches decode_body_action().
        input_rgb = self._frame if self._shared_input_rgb is None else self._shared_input_rgb
        stimulation = self.general_stimulation
        # File-backed independent-neuron currents are one interval, including
        # failures; a later retry must explicitly resubmit its own descriptor.
        self.general_stimulation = [item for item in stimulation if "neuron_currents_file" not in item]
        summary = self.brain.step(input_rgb, duration_ms=28.6,
                                  neural_drive=drive,
                                  general_stimulation=stimulation,
                                  visual_input_enabled=self.visual_input_enabled,
                                  include_group_stats=True,
                                  reward_event={"aversive": self._pending_aversive})
        self._shared_input_rgb = None
        self._shared_body_packet = None
        self._pending_aversive = False
        self.neural = summary
        self.group_state = deepcopy(summary["group_state"])
        self.applied_neural_drive = drive
        self._brain_vector = self._telemetry_vector(summary)
        self._body_action = self._decode_brain_to_body(summary)
        _, reward, terminated, truncated, self._body_info = self.body.step(self._body_action)
        self._done = bool(terminated or truncated)
        self._frame = self.body.render()
        info = self.task_state()
        info.update(terminated=bool(terminated), truncated=bool(truncated))
        return self.observe(), float(reward), bool(terminated), bool(truncated), info

    def apply_task(self, task):
        if not isinstance(task, dict) or not task or set(task) - {"neural_drive", "general_stimulation"}:
            raise ValueError("Neural tasks support neural_drive and general_stimulation.")
        context = validate_neural_drive(task["neural_drive"]) if "neural_drive" in task else self.semantic_drive
        general = validate_general_stimulation(task["general_stimulation"]) if "general_stimulation" in task else self.general_stimulation
        self.semantic_drive, self.general_stimulation = context, general
        return {"applied": {"neural_drive": deepcopy(context), "general_stimulation": deepcopy(general)},
                "effect": "3-port context is executed by the controller; persistent general stimulation is delivered on the next real brain step"}

    def set_general_stimulation(self, stimulation):
        self.general_stimulation = validate_general_stimulation(stimulation)
        return {"general_stimulation": deepcopy(self.general_stimulation), "delivery": "next neural interval"}

    def set_macro_group_currents(self, currents_mv):
        """Replace additional stimuli with a signed dense macro current vector."""
        if currents_mv is None:
            return self.set_general_stimulation([])
        values = np.asarray(currents_mv)
        if values.dtype.kind not in "fiu" or values.shape != (47,) or not np.isfinite(values).all() or np.any(np.abs(values) > 30):
            raise ValueError("Macro currents require 47 finite values within [-30,30] mV-equivalent.")
        return self.set_general_stimulation([{"macro_group_currents_mv": values.astype(float).tolist()}])

    def set_neuron_currents(self, descriptor, *, preserve_existing=True):
        """Add one interval of model input while preserving the other senses.

        Replacing/removing an old file prevents accidental replay. Existing
        group/ID currents, semantic drives and the pinned camera are untouched.
        """
        if type(preserve_existing) is not bool:
            raise ValueError("preserve_existing must be bool")
        stimuli = [item for item in self.general_stimulation if "neuron_currents_file" not in item] if preserve_existing else []
        if descriptor is not None:
            stimuli = [*stimuli, {"neuron_currents_file": descriptor}]
        return self.set_general_stimulation(stimuli)

    def append_neuron_currents(self, descriptor):
        return self.set_neuron_currents(descriptor, preserve_existing=True)

    def shared_passthrough_action(self):
        """Keep the currently configured sensory-port context during sharing."""
        drive = validate_neural_drive(self.semantic_drive)
        return np.asarray([2 * drive[name] - 1 for name in PORT_ORDER], dtype=np.float32)

    def set_visual_input_enabled(self, enabled):
        if type(enabled) is not bool:
            raise ValueError("visual_input_enabled must be bool")
        self.visual_input_enabled = enabled
        return {"visual_input_enabled": enabled,
                "effect": "Legacy retinal image projection enabled" if enabled else "Legacy retinal image projection bypassed; explicit currents and original tonic dynamics remain"}

    def set_motor_readout(self, checkpoint):
        """Select an explicitly trained physical-body adapter; default stays BCI."""
        if checkpoint is None:
            self.motor_readout, self._motor_trace = None, None
            return {"mode": "rule_bci", "default": True}
        if self.scenario_spec.get("embodiment") != "physical_3d" or self.body.action_space.shape != (2,):
            raise ValueError("This learned motor readout supports only the two-command MuJoCo body")
        from shared_io.motor_readout import MotorReadout
        candidate = MotorReadout.load(checkpoint)
        if candidate.optimizer_updates < 1:
            raise ValueError("Train this motor readout before applying it to the body")
        self.motor_readout, self._motor_trace = candidate, None
        return candidate.describe()

    def _decode_brain_to_body(self, summary):
        if self.motor_readout is None:
            self._motor_trace = None
            return self.decode_body_action(summary["native_action"])
        arrays, descriptor = self.brain.read_arrays()
        action, self._motor_trace = self.motor_readout.predict_measurement(arrays, descriptor, summary)
        summary["motor_readout"] = deepcopy(self._motor_trace)
        return action

    def observe(self):
        if not self._started:
            raise RuntimeError("Call reset() before observe().")
        return np.concatenate([self._body_observation(),
                               np.array([self.semantic_drive[name] for name in PORT_ORDER], dtype=np.float32),
                               self._brain_vector]).astype(np.float32)

    def task_state(self):
        return {**deepcopy(self._body_info), "scenario": self.scenario_spec["name"],
                "phase": "finished" if self._done else "running" if self._started else "not_reset",
                "neural": deepcopy(self.neural), "neural_drive": deepcopy(self.semantic_drive),
                "applied_neural_drive": deepcopy(self.applied_neural_drive),
                "general_stimulation": deepcopy(self.general_stimulation),
                "visual_input_enabled": self.visual_input_enabled,
                "motor_readout": deepcopy(self._motor_trace) if self._motor_trace else self.motor_readout.describe() if self.motor_readout else {"mode": "rule_bci"},
                "group_state": deepcopy(self.group_state),
                "body_action": self._body_action.tolist(), "brain_learning": self.learning,
                "feedback_visual": self.body.task_state().get("feedback_visual"),
                "stimulus_pending": self._pending_aversive,
                "cancelled_manual_events": self._cancelled_aversive,
                "last_reset_cancelled_manual_event": self._last_reset_cancelled_event,
                "input_ports": deepcopy(self.brain_description["input_ports"]),
                "input_ports_sha256": self.brain_description.get("input_ports_sha256"),
                "brain_backend": self.brain_description["backend"],
                "brain_neurons": self.brain_description["neurons"],
                "brain_edges": self.brain_description["edges"],
                "neural_event_log": str(self.brain.event_log_path),
                "feedback_mode": "explicit manual aversion only; terminal outcomes are not silently converted to punishment",
                "decoder_note": "Learned engineering readout: 2,129 measured output cells to two CPG commands" if self.motor_readout else "Engineered: turn/6 and forward/10-1 map actual neural BCI readouts to left/right CPG commands",
                "neural_interval_ms": 28.6, "body_interval_ms": self._body_interval_ms()}

    def render(self):
        if self._frame is None:
            raise RuntimeError("Call reset() before render().")
        # Refresh MuJoCo prop lifetimes even while physics is paused. These are
        # real camera pixels, and never a fabricated neural-feedback event.
        self._frame = self.body.render()
        return self._frame.copy()

    def shared_input_rgb(self):
        """Capture and pin exactly the real RGB delivered on the next brain step."""
        if not self._started or self._done:
            raise RuntimeError("Shared input requires an active, unfinished episode.")
        if self._shared_input_rgb is not None:
            if (self._shared_body_packet is not None and
                    self._shared_body_packet["simulation_time_s"] != float(self.body.unwrapped.sim.mj_data.time)):
                raise RuntimeError("The shared camera is pinned to another physics instant")
            # Concurrent consumers on the owner thread (model dispatch and
            # online recording) must see the exact same upcoming brain image.
            # Rendering again could expire an already-delivered 3D feedback
            # effect while physics has not advanced. step/reset releases it.
            return self._shared_input_rgb.copy()
        self._shared_input_rgb = self.body.render().copy()
        self._shared_body_packet = self.body.unwrapped.body_sensory_packet(rgb=self._shared_input_rgb) if self.scenario_spec.get("embodiment") == "physical_3d" else None
        return self._shared_input_rgb.copy()

    def body_sensory_packet(self):
        if self.scenario_spec.get("embodiment") != "physical_3d":
            raise ValueError("This scene has no MuJoCo body sensory packet")
        return self.body.unwrapped.body_sensory_packet()

    def shared_body_senses(self):
        """Return the body packet sampled at the exact pinned camera instant."""
        if self.scenario_spec.get("embodiment") != "physical_3d":
            raise ValueError("This scene has no MuJoCo body sensory packet")
        if self._shared_body_packet is None:
            self.shared_input_rgb()
        if self._shared_body_packet["simulation_time_s"] != float(self.body.unwrapped.sim.mj_data.time):
            raise RuntimeError("The pinned body packet is from another physics instant")
        return deepcopy(self._shared_body_packet)

    def shared_input_metadata(self):
        return {"image_source": "physical_camera" if self.scenario_spec.get("embodiment") == "physical_3d" else "game_camera",
                "shared_with_brain": True, "lifetime": "next neural step"}

    def show_feedback(self, kind="aversive", intensity=1.0, duration_ms=800):
        if not hasattr(self.body, "show_feedback"):
            raise ValueError("This scene has no 3D feedback prop implementation.")
        result = self.body.show_feedback(kind, intensity, duration_ms)
        if self._started:
            self._frame = self.body.render()
        return result

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self.brain is not None:
                self.brain.close()
        finally:
            self.body.close()
