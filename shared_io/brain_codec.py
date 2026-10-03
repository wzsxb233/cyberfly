"""Strict numerical interface between real full-brain measurements and a coupler.

This module does not invent a neural state, run an LLM, or train the coupler.
It encodes measured macro-group statistics and turns learned 47-value logits
into actual bounded currents. NumPy is sufficient for IO; torch logits retain
their autograd graph until explicitly serialized for the external brain worker.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import weakref

import numpy as np

FEATURE_DIM = 240
CURRENT_DIM = 47
NEURON_COVERAGE = 166700
GRAPH_EDGES = 25582938
MACRO_PARTITION_SHA256 = "185289a5e45fa8df33b088828cc92543c16bfdf76d6c728ca596ccf8b22f026c"
CATALOG_PATH = Path(__file__).resolve().parents[1] / "artifacts/full_brain/catalog.json"


def _number(value, name, *, minimum=None, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"{name} is outside its permitted range.")
    return float(value)


def _vector(value, size, name, *, integer=False, nonnegative=False):
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf":
            raise ValueError
        array = raw.astype(np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a numeric vector.") from None
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain exactly {size} finite numbers.")
    if integer and np.any(array != np.floor(array)) or nonnegative and np.any(array < 0):
        raise ValueError(f"{name} contains invalid counts.")
    return array


def _torch_tensor(value):
    # Do not import a second heavyweight runtime for plain numerical encoding.
    import sys
    torch = sys.modules.get("torch")
    return torch if torch is not None and isinstance(value, torch.Tensor) else None


class BrainFeatureCodec:
    """240 continuous features, preserving the complete 47-group partition.

    Group-major features, five per group: log-rate, mean/min/max voltage,
    and population fraction. Five global features follow: population log-rate,
    population mean voltage, interval length, elapsed neural time, plasticity.
    A supplied catalog is trusted metadata, but every observed group ID, count,
    partition hash and total count must match it exactly before encoding.
    """
    feature_dim = FEATURE_DIM
    current_dim = CURRENT_DIM
    neuron_coverage = NEURON_COVERAGE
    macro_partition_sha256 = MACRO_PARTITION_SHA256
    current_limit_mv = 30.0

    def __init__(self, catalog: dict | None = None):
        if catalog is None:
            catalog = json.loads(CATALOG_PATH.read_text())
        if not isinstance(catalog, dict):
            raise ValueError("Catalog must be an object.")
        if catalog.get("macro_partition_sha256") != MACRO_PARTITION_SHA256 or catalog.get("coverage") != NEURON_COVERAGE:
            raise ValueError("The shared codec requires the verified complete MaleCNS macro partition.")
        groups = deepcopy(catalog.get("macro_groups"))
        if not isinstance(groups, list) or len(groups) != CURRENT_DIM:
            raise ValueError("The codec requires all 47 macro groups.")
        if [group.get("index") for group in groups] != list(range(CURRENT_DIM)):
            raise ValueError("Catalog macro groups must be in canonical index order.")
        self.group_ids = tuple(group.get("group_id") for group in groups)
        if len(set(self.group_ids)) != CURRENT_DIM or any(not isinstance(gid, str) for gid in self.group_ids):
            raise ValueError("Catalog macro group identities must be unique strings.")
        self.group_counts = _vector([group.get("count") for group in groups], CURRENT_DIM, "catalog counts", integer=True, nonnegative=True)
        if np.any(self.group_counts <= 0) or self.group_counts.sum() != NEURON_COVERAGE:
            raise ValueError("Catalog groups must partition all 166700 retained neurons.")
        self.group_counts.setflags(write=False)
        self.groups = groups
        self._verified_brains = weakref.WeakKeyDictionary()
        components = ("log_rate", "mean_voltage", "min_voltage", "max_voltage", "population_fraction")
        self.feature_names = tuple(f"{gid}.{feature}" for gid in self.group_ids for feature in components) + (
            "population.log_rate", "population.mean_voltage", "interval_ms_scaled", "elapsed_seconds_log", "plasticity_enabled")
        description = {"schema": 1, "feature_dim": FEATURE_DIM, "current_dim": CURRENT_DIM,
            "macro_partition_sha256": MACRO_PARTITION_SHA256, "neuron_coverage": NEURON_COVERAGE,
            "graph_edges": GRAPH_EDGES,
            "group_ids": list(self.group_ids), "group_counts": self.group_counts.astype(int).tolist(),
            "feature_names": list(self.feature_names),
            "normalization": {"log_rate": "log1p(spikes / neurons / interval_seconds) / log1p(200)",
                "voltage": "(measured_mV + 65) / 30, without clipping", "population_fraction": "group_count / 166700",
                "interval_ms_scaled": "actual interval_ms / 200", "elapsed_seconds_log": "log1p(actual simulation_ms / 1000)",
                "plasticity_enabled": "0 or 1 from actual brain learning state"},
            "current_decode": "30 * tanh(coupler_logits), model membrane mV-equivalent",
            "interpretation": "Measured aggregate numerical IO over every retained neuron; no neuron-state synthesis or biological-semantic equivalence claim"}
        self.codec_sha256 = hashlib.sha256(json.dumps(description, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self._description = {**description, "codec_sha256": self.codec_sha256}

    def describe(self):
        return deepcopy(self._description)

    def _validated_groups(self, state):
        if not isinstance(state, dict) or state.get("level") != "macro":
            raise ValueError("Encode actual macro group_state, not a sampled or fine-group subset.")
        if state.get("partition_sha256") != MACRO_PARTITION_SHA256 or state.get("coverage") != NEURON_COVERAGE:
            raise ValueError("Measured brain coverage or partition differs from this codec.")
        if state.get("group_ids") != list(self.group_ids):
            raise ValueError("Measured macro group order or identity changed.")
        counts = _vector(state.get("counts"), CURRENT_DIM, "group counts", integer=True, nonnegative=True)
        if not np.array_equal(counts, self.group_counts):
            raise ValueError("Measured group counts do not cover the verified neuron partition.")
        spikes = _vector(state.get("spikes"), CURRENT_DIM, "group spikes", integer=True, nonnegative=True)
        means = _vector(state.get("voltage_mean"), CURRENT_DIM, "mean voltage")
        minima = _vector(state.get("voltage_min"), CURRENT_DIM, "minimum voltage")
        maxima = _vector(state.get("voltage_max"), CURRENT_DIM, "maximum voltage")
        if np.any(minima > means + 1e-4) or np.any(means > maxima + 1e-4):
            raise ValueError("Measured voltage extrema do not contain group means.")
        return counts, spikes, means, minima, maxima

    def encode_group_state(self, state: dict, *, duration_ms: float, learning: bool = False,
                           simulation_ms: float | None = None, expected_total_spikes: int | None = None):
        """Encode actual measurements; callers must supply the measured interval.

        A zero interval is permitted only for the true zero-spike reset state.
        It never fabricates a firing rate for an unknown measurement duration.
        """
        if type(learning) is not bool:
            raise ValueError("learning must reflect an actual boolean brain setting.")
        duration = _number(duration_ms, "duration_ms", minimum=0, maximum=200)
        counts, spikes, means, minima, maxima = self._validated_groups(state)
        actual_time = _number(state.get("simulation_ms"), "group simulation_ms", minimum=0)
        if simulation_ms is not None:
            supplied_time = _number(simulation_ms, "simulation_ms", minimum=0)
            if not math.isclose(supplied_time, actual_time, rel_tol=1e-9, abs_tol=1e-5):
                raise ValueError("Neural summary and group measurement timestamps differ.")
        total = int(spikes.sum())
        if expected_total_spikes is not None:
            expected = _number(expected_total_spikes, "total spikes", minimum=0)
            if expected != int(expected) or total != expected:
                raise ValueError("Group spike sum differs from the actual full-brain spike total.")
        if duration == 0:
            if total != 0 or actual_time != 0:
                raise ValueError("An unknown/zero interval cannot encode post-step rates.")
            rates = np.zeros(CURRENT_DIM)
        else:
            rates = spikes / counts / (duration / 1000)
        log_normalizer = np.log1p(200.0)
        group_features = np.column_stack((np.log1p(rates) / log_normalizer,
            (means + 65) / 30, (minima + 65) / 30, (maxima + 65) / 30, counts / NEURON_COVERAGE))
        population_rate = total / NEURON_COVERAGE / (duration / 1000) if duration else 0
        population_voltage = float(np.average(means, weights=counts))
        features = np.r_[group_features.reshape(-1), np.log1p(population_rate) / log_normalizer,
            (population_voltage + 65) / 30, duration / 200, np.log1p(actual_time / 1000), float(learning)].astype(np.float32)
        if features.shape != (FEATURE_DIM,) or not np.isfinite(features).all():
            raise RuntimeError("Non-finite numerical brain features; refusing shared IO.")
        return features

    def encode_step(self, result: dict):
        """Encode the actual ConnectomeClient.step(..., include_group_stats=True) result."""
        if not isinstance(result, dict) or not isinstance(result.get("neural"), dict):
            raise ValueError("Expected a real brain step summary.")
        neural = result["neural"]
        if neural.get("neurons") != NEURON_COVERAGE or neural.get("edges") != GRAPH_EDGES:
            raise ValueError("The neural step differs from the verified complete graph.")
        duration = _number(neural.get("duration_ms"), "duration_ms", minimum=0.1, maximum=200)
        simulation_ms = _number(neural.get("simulation_ms"), "simulation_ms", minimum=0)
        total_spikes = _number(neural.get("spikes_this_step"), "full-brain spikes", minimum=0)
        learning = result.get("learning", {}).get("enabled")
        return self.encode_group_state(result.get("group_state"), duration_ms=duration, learning=learning,
            simulation_ms=simulation_ms, expected_total_spikes=total_spikes)

    def encode_env_state(self, state: dict):
        """Accept full_brain_sandbox or a body-neural task_state without changing its old obs shape."""
        if not isinstance(state, dict):
            raise ValueError("Environment state must be an object.")
        if isinstance(state.get("neural"), dict) and "neurons" in state["neural"]:
            return self.encode_step(state)
        result = state.get("neural")
        if isinstance(result, dict) and "neural" in result:
            features = self.encode_step(result)
            if state.get("group_state") is not None and state["group_state"] != result["group_state"]:
                raise ValueError("Environment and neural summary refer to different group measurements.")
            return features
        # Embodied environments expose measured zero state immediately after reset.
        return self.encode_group_state(state.get("group_state"), duration_ms=0,
            learning=state.get("brain_learning", state.get("learning", False)))

    def currents_from_logits(self, logits):
        """Differentiable tanh bound: (..., 47) -> (..., 47) signed mV currents."""
        torch = _torch_tensor(logits)
        if torch is not None:
            if logits.ndim < 1 or logits.shape[-1] != CURRENT_DIM or not logits.is_floating_point() or not bool(torch.isfinite(logits).all()):
                raise ValueError("Coupler logits must be finite floating tensors ending in 47 values.")
            return torch.tanh(logits) * self.current_limit_mv
        array = np.asarray(logits)
        if array.dtype.kind != "f" or array.ndim < 1 or array.shape[-1] != CURRENT_DIM or not np.isfinite(array).all():
            raise ValueError("Coupler logits must be finite floating arrays ending in 47 values.")
        return np.tanh(array) * self.current_limit_mv

    def validate_currents(self, currents_mv):
        """Materialize exactly one current vector only at the external IO boundary."""
        torch = _torch_tensor(currents_mv)
        if torch is not None:
            if currents_mv.dtype == torch.bool:
                raise ValueError("Macro currents must be numerical, not boolean.")
            currents_mv = currents_mv.detach().float().cpu().numpy()
        values = _vector(currents_mv, CURRENT_DIM, "macro currents")
        if np.any(np.abs(values) > self.current_limit_mv):
            raise ValueError("Macro currents must be within [-30,30] mV-equivalent.")
        return values

    def stimulation(self, currents_mv):
        return [{"macro_group_currents_mv": self.validate_currents(currents_mv).tolist()}]

    def verify_brain(self, brain):
        """Check current destination group identity before the first numerical write."""
        if brain is None or not callable(getattr(brain, "group_state", None)):
            raise ValueError("Initialize/reset the actual brain before sending shared numerical IO.")
        if brain not in self._verified_brains:
            self._validated_groups(brain.group_state())
            info = getattr(brain, "info", {})
            if info.get("neurons") != NEURON_COVERAGE or info.get("edges") != GRAPH_EDGES:
                raise ValueError("Shared numerical output requires the complete verified brain.")
            self._verified_brains[brain] = True
        return {"macro_partition_sha256": MACRO_PARTITION_SHA256, "neuron_coverage": NEURON_COVERAGE}

    def step_brain(self, brain, rgb, currents_mv, *, duration_ms=28.6, reward_event=None):
        """Send the coupler's numerical output directly to the real complete graph."""
        self.verify_brain(brain)
        return brain.step(rgb, duration_ms=duration_ms, general_stimulation=self.stimulation(currents_mv),
            include_group_stats=True, reward_event=reward_event)

    def step_env(self, env, currents_mv):
        """Execute numerical currents through a real neural scene and its optional body.

        This replaces prior additional group/ID stimuli to make this output the
        sole general-current command. Actual RGB sensory input and an explicitly
        queued aversive event still follow the environment's normal physics/brain
        path. No body action is supplied directly by the coupler.
        """
        values = self.validate_currents(currents_mv)
        spec = getattr(env, "scenario_spec", {})
        if not spec.get("neural_input") or not callable(getattr(env, "set_general_stimulation", None)):
            raise ValueError("Numerical brain currents require a real neural-input environment.")
        self.verify_brain(getattr(env, "brain", None))
        if spec.get("action_interface") == "group_currents":
            if env.action_space.shape != (CURRENT_DIM,):
                raise ValueError("Full-brain environment action identity differs from the codec.")
            env.set_general_stimulation([])
            return env.step((values / self.current_limit_mv).astype(np.float32))
        if env.action_space.shape != (3,):
            raise ValueError("Unsupported neural scene action interface.")
        env.set_general_stimulation(self.stimulation(values))
        return env.step(-np.ones(3, dtype=np.float32))
