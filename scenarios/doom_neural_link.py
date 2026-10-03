"""The same neural-current action contract, coupled to the real Doom game."""

from copy import deepcopy

import numpy as np
from PIL import Image

from .neural_link import FlyNeuralLink, NEURAL_LINK_SPEC


DOOM_NEURAL_LINK_SPEC = {
    **deepcopy(NEURAL_LINK_SPEC), "name": "doom_neural_link", "title": "神经直连 · Doom",
    "observation_shape": [81], "render_shape": [240, 320, 3], "embodiment": "game",
    "supports_3d_feedback": False,
    "engine": "Real full MaleCNS graph → engineered BCI decoder → ViZDoom basic",
    "limitations": ["PPO trains a neural-input adapter; graph plasticity is a separate mode",
                   "Adapter observation includes an 8x8 grayscale view and engine telemetry; the brain receives the full real RGB image",
                   "Attack/turn-to-game mapping is an engineering decoder, not innate Doom knowledge",
                   "No automatic terminal punishment: explicit manual aversion only"],
}


class DoomNeuralLink(FlyNeuralLink):
    """Every action still goes through neural currents and the complete graph.

    Observation: 64 real downsampled grayscale pixels, health/ammo/kills/time/
    reward/done (6), semantic context (3), and actual brain telemetry (8).
    This is a vector MLP adapter, distinct from doom_basic's direct CNN policy.
    """

    SCENARIO_SPEC = DOOM_NEURAL_LINK_SPEC

    def _make_body(self, max_episode_steps, render_mode, kwargs):
        from .doom_basic import DoomBasicScenario
        return DoomBasicScenario(max_episode_steps=max_episode_steps, render_mode=render_mode, **kwargs)

    def _zero_body_action(self):
        return np.int64(0)

    def _body_interval_ms(self):
        return self.body.frame_skip / 35.0 * 1000.0

    def _body_observation(self):
        grayscale = Image.fromarray(self._frame).convert("L").resize((8, 8), Image.Resampling.BILINEAR)
        pixels = np.asarray(grayscale, dtype=np.float32).reshape(-1) / 255.0
        info = self._body_info
        state = np.array([info.get("health", 0) / 100.0, info.get("ammo", 0) / 50.0,
                          info.get("kills", 0), info.get("step", 0) / self.body.max_episode_steps,
                          info.get("episode_reward", 0) / 10.0, float(self._done)], dtype=np.float32)
        return np.clip(np.concatenate([pixels, state]), -10.0, 10.0).astype(np.float32)

    @staticmethod
    def decode_body_action(native):
        return np.int64(2 if native["attack"] else (0 if float(native["turn"]) < 0 else 1))

    def task_state(self):
        result = super().task_state()
        result["decoder_note"] = "Engineered: actual neural attack readout selects shoot; otherwise neural turn sign selects left/right strafe"
        return result
