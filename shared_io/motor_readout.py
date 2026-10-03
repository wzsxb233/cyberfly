"""Optional, learned engineering readout from measured output cells to a body.

This is a separate motor adapter, not a recovered biological muscle circuit.
The full brain still runs upstream. Only actual output-cell rates and voltages
enter this module; pixels, language, goals and rewards are not inference inputs.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "artifacts/full_brain/catalog.json"
GRAPH_PATH = ROOT / "vendor/doomfly/outputs/doom/malecns_v1/graph.npz"
OUTPUT_SUPERCLASSES = ("cb_motor", "vnc_motor", "descending_neuron")
OUTPUT_COUNTS = {"cb_motor": 107, "vnc_motor": 708, "descending_neuron": 1314}
OUTPUT_NEURONS = 2129
OUTPUT_IDS_SHA256 = "8f49c6883c8f169575187f4e38e6512a96a5652e8902e3cd88fc2b7e57b902d0"
FEATURE_DIM = OUTPUT_NEURONS * 2
GRAPH_NEURONS = 166700
GRAPH_EDGES = 25582938


def array_sha256(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def output_selection():
    """Read the actual release IDs and exact superclass annotation partition."""
    catalog = json.loads(CATALOG_PATH.read_text())
    with np.load(GRAPH_PATH, allow_pickle=False) as graph:
        ids, classes = graph["ids"], graph["superclass"]
    if ids.dtype != np.int64 or ids.shape != (GRAPH_NEURONS,) or array_sha256(ids) != catalog["ids_sha256"]:
        raise ValueError("Motor readout requires the verified full MaleCNS ID order")
    if classes.shape != ids.shape:
        raise ValueError("Superclass annotations do not align with graph IDs")
    counts = {name: int(np.count_nonzero(classes == name)) for name in OUTPUT_SUPERCLASSES}
    if counts != OUTPUT_COUNTS:
        raise ValueError("Motor output annotation counts differ from the verified release")
    indices = np.flatnonzero(np.isin(classes, OUTPUT_SUPERCLASSES))
    selected = ids[indices]
    if array_sha256(selected) != OUTPUT_IDS_SHA256:
        raise ValueError("Output-cell identities differ from the verified annotation selection")
    return {"schema": 1, "graph_ids_sha256": catalog["ids_sha256"],
            "output_ids_sha256": array_sha256(selected), "neuron_ids": selected.tolist(),
            "indices": indices.tolist(), "superclass_counts": counts,
            "selection": "All retained cells annotated cb_motor, vnc_motor, or descending_neuron; tentative classes excluded",
            "source": str(GRAPH_PATH), "annotation_source": catalog["annotation_source"]}


class MotorReadout(nn.Module):
    """4,258 measured values -> trainable MLP -> two bounded CPG commands."""

    def __init__(self, *, hidden_width=32, seed=42):
        super().__init__()
        if type(hidden_width) is not int or not 4 <= hidden_width <= 256:
            raise ValueError("hidden_width must be an integer within 4..256")
        self.selection = output_selection()
        self.indices = np.asarray(self.selection["indices"], dtype=np.int64)
        self.neuron_ids = np.asarray(self.selection["neuron_ids"], dtype=np.int64)
        self.hidden_width = hidden_width
        # Constructing a readout must not alter another learner's RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.network = nn.Sequential(nn.Linear(FEATURE_DIM, hidden_width), nn.Tanh(),
                                         nn.Linear(hidden_width, 2), nn.Tanh())
        self.optimizer_updates = 0
        self.label_source = None
        self.checkpoint_path = None

    def describe(self):
        return {"schema": 1, "mode": "learned_readout", "feature_dim": FEATURE_DIM,
                "output_cells": OUTPUT_NEURONS, "output_ids_sha256": self.selection["output_ids_sha256"],
                "graph_ids_sha256": self.selection["graph_ids_sha256"],
                "superclass_counts": deepcopy(OUTPUT_COUNTS), "hidden_width": self.hidden_width,
                "optimizer_updates": self.optimizer_updates, "label_source": self.label_source,
                "checkpoint": self.checkpoint_path,
                "features": "Per-cell [log1p(actual interval Hz)/log1p(200), (actual membrane mV+65)/30]; graph-index order",
                "output": "Two engineering CPG commands within [-1,1], mapped by the body to descending amplitudes 0.8+0.4*command",
                "interpretation": "Learned engineering interface over actual annotated output cells; no claim of natural muscle mapping, thought, or vocal behavior"}

    def encode(self, arrays, descriptor, summary):
        """Bind full measured arrays to the exact just-completed neural interval."""
        neural = summary.get("neural", {})
        for value in (descriptor, neural):
            if value.get("neurons") != GRAPH_NEURONS or value.get("edges") != GRAPH_EDGES:
                raise ValueError("Motor observation requires the complete retained graph")
        ids, voltage, counts = (np.asarray(arrays.get(key)) for key in ("ids", "v", "counts"))
        if ids.dtype != np.int64 or ids.shape != (GRAPH_NEURONS,) or array_sha256(ids) != self.selection["graph_ids_sha256"]:
            raise ValueError("Motor observation neuron identity/order differs from its checkpoint")
        if not np.array_equal(ids[self.indices], self.neuron_ids):
            raise ValueError("Selected motor cell identity/order changed")
        if voltage.shape != ids.shape or voltage.dtype.kind != "f" or not np.isfinite(voltage).all():
            raise ValueError("Motor observation needs finite measured membrane voltages")
        if counts.shape != ids.shape or counts.dtype.kind not in "iu" or np.any(counts < 0):
            raise ValueError("Motor observation needs measured nonnegative interval spike counts")
        duration = neural.get("duration_ms")
        actual_time = neural.get("simulation_ms")
        if type(duration) not in (int, float) or not np.isfinite(duration) or not 0 < duration <= 200:
            raise ValueError("A measured positive neural interval is required for rate decoding")
        if type(actual_time) not in (int, float) or not np.isfinite(actual_time) or not np.isclose(descriptor.get("simulation_ms", -1), actual_time, rtol=0, atol=1e-6):
            raise ValueError("Snapshot and motor observation come from different neural intervals")
        if int(counts.sum()) != neural.get("spikes_this_step") or array_sha256(counts) != neural.get("counts_sha256"):
            raise ValueError("Snapshot spikes differ from the just-completed full-brain step")
        rates = counts[self.indices].astype(np.float64) / (duration / 1000)
        features = np.column_stack((np.log1p(rates) / np.log1p(200),
                                    (voltage[self.indices].astype(np.float64) + 65) / 30)).reshape(-1).astype(np.float32)
        if features.shape != (FEATURE_DIM,) or not np.isfinite(features).all():
            raise ValueError("Nonfinite motor readout features")
        return features

    def forward(self, features):
        if not isinstance(features, torch.Tensor) or not features.is_floating_point() or features.ndim not in (1, 2) or features.shape[-1] != FEATURE_DIM or not bool(torch.isfinite(features).all()):
            raise ValueError("Motor features must be a finite floating tensor ending in 4258 values")
        return self.network(features)

    def predict_measurement(self, arrays, descriptor, summary):
        features = self.encode(arrays, descriptor, summary)
        device = next(self.parameters()).device
        with torch.no_grad():
            action = self(torch.as_tensor(features, device=device)).detach().cpu().numpy().astype(np.float32)
        trace = {**self.describe(), "simulation_ms": summary["neural"]["simulation_ms"],
                 "input_counts_sha256": summary["neural"]["counts_sha256"],
                 "features_sha256": array_sha256(features),
                 "output_spikes": int(np.asarray(arrays["counts"])[self.indices].sum()),
                 "output_voltage_mean_mv": float(np.asarray(arrays["v"])[self.indices].mean()),
                 "body_action": action.tolist(), "raw_input_shape": [OUTPUT_NEURONS, 2]}
        return action, trace

    def fit_supervised(self, features, target_actions, *, label_source, steps=2, learning_rate=1e-3):
        """Optimize against explicit teaching actions; no invented reward labels."""
        x, y = np.asarray(features, dtype=np.float32), np.asarray(target_actions, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != FEATURE_DIM or y.shape != (len(x), 2) or not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(np.abs(y) > 1):
            raise ValueError("Supply finite measured feature rows and explicit two-value teaching actions")
        if not isinstance(label_source, str) or not label_source.strip() or type(steps) is not int or not 1 <= steps <= 100000:
            raise ValueError("A teaching-source description and 1..100000 optimization steps are required")
        if not np.isfinite(learning_rate) or not 0 < learning_rate <= .1:
            raise ValueError("learning_rate must be finite within (0,.1]")
        device = next(self.parameters()).device
        xt, yt = torch.as_tensor(x, device=device), torch.as_tensor(y, device=device)
        optimizer = torch.optim.Adam(self.parameters(), lr=float(learning_rate))
        initial = torch.cat([p.detach().cpu().reshape(-1) for p in self.parameters()]).clone()
        with torch.no_grad():
            before = float(torch.mean((self(xt) - yt) ** 2))
        losses = []
        for _ in range(steps):
            optimizer.zero_grad()
            loss = torch.mean((self(xt) - yt) ** 2)
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in self.parameters()):
                raise RuntimeError("Nonfinite motor readout gradient")
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
            self.optimizer_updates += 1
            losses.append(float(loss.detach()))
        final = torch.cat([p.detach().cpu().reshape(-1) for p in self.parameters()])
        self.label_source = label_source.strip()
        with torch.no_grad():
            after = float(torch.mean((self(xt) - yt) ** 2))
        return {"method": "supervised_motor_readout", "examples": len(x), "optimizer_steps": steps,
                "optimizer_state": "Adam initialized for this fit call; weights retained across calls",
                "loss_before": before, "loss_after": after, "losses": losses,
                "changed_parameters": int(torch.count_nonzero(final != initial)),
                "parameter_l2_change": float(torch.linalg.vector_norm(final - initial)),
                "initial_weights_sha256": array_sha256(initial.numpy()),
                "final_weights_sha256": array_sha256(final.numpy()), "label_source": self.label_source}

    def save(self, path):
        path = Path(path).resolve()
        path.mkdir(parents=True, exist_ok=True)
        if any(path.iterdir()):
            raise FileExistsError("Use an empty motor checkpoint directory")
        weights = path / "weights.npz"
        np.savez(weights, **{key: value.detach().cpu().numpy() for key, value in self.state_dict().items()})
        meta = {**self.describe(), "selection": self.selection,
                "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest()}
        (path / "metadata.json").write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n")
        self.checkpoint_path = str(path)
        return {"path": str(path), "weights_sha256": meta["weights_sha256"],
                "output_ids_sha256": self.selection["output_ids_sha256"]}

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        meta = json.loads((path / "metadata.json").read_text())
        if meta.get("schema") != 1 or meta.get("mode") != "learned_readout" or meta.get("feature_dim") != FEATURE_DIM:
            raise ValueError("Unsupported motor readout checkpoint")
        model = cls(hidden_width=meta["hidden_width"])
        if meta.get("selection") != model.selection:
            raise ValueError("Saved output-cell selection differs from the verified graph")
        weights = path / "weights.npz"
        if hashlib.sha256(weights.read_bytes()).hexdigest() != meta.get("weights_sha256"):
            raise ValueError("Motor checkpoint weight integrity mismatch")
        with np.load(weights, allow_pickle=False) as data:
            expected = model.state_dict()
            if set(data.files) != set(expected):
                raise ValueError("Motor checkpoint parameter names differ")
            state = {}
            for key, value in expected.items():
                if data[key].dtype != np.float32 or data[key].shape != tuple(value.shape) or not np.isfinite(data[key]).all():
                    raise ValueError("Motor checkpoint has invalid parameters")
                state[key] = torch.from_numpy(data[key].copy())
        model.load_state_dict(state, strict=True)
        updates = meta.get("optimizer_updates")
        if type(updates) is not int or updates < 0:
            raise ValueError("Invalid motor optimizer update count")
        model.optimizer_updates, model.label_source = updates, meta.get("label_source")
        model.checkpoint_path = str(path)
        return model.eval()
