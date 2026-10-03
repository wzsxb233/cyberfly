"""True MaleCNS v6 worker. Stdout is exclusively the JSON-lines protocol."""
from __future__ import annotations

import base64
from contextlib import redirect_stdout
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "vendor/doomfly"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO))

from connectome_adapter.ports import DRIVE_LIMITS_MV, validate_neural_drive
from connectome_adapter.full_io import FullBrainIO


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class FrozenStimulus:
    """Bookkeeping for external pulses without exposing plasticity controls."""

    enabled = False

    def __init__(self):
        self.events = 0
        self.until = 0
        self.last_steps = 0
        self.delivered_steps = 0
        self.cancelled_steps = 0
        self.terminal_events = 0

    def telemetry(self):
        return {
            "enabled": False,
            "plasticity": False,
            "external_aversive_events": self.events,
            "delivered_steps": self.delivered_steps,
            "last_steps": self.last_steps,
            "cancelled_steps": self.cancelled_steps,
            "pending_steps": 0,
            "stimulus_source": "engineered_external_event",
        }

    def state(self):
        return {
            "enabled": False,
            "events": self.events,
            "until": self.until,
            "last_steps": self.last_steps,
            "delivered_steps": self.delivered_steps,
            "cancelled_steps": self.cancelled_steps,
            "terminal_events": self.terminal_events,
        }

    def restore(self, value):
        if not isinstance(value, dict) or value.get("enabled", False):
            raise ValueError("Only frozen runtime checkpoints are supported")
        for key in ("events", "until", "last_steps", "delivered_steps", "cancelled_steps", "terminal_events"):
            number = value.get(key, 0)
            if type(number) is not int or number < 0:
                raise ValueError("Invalid frozen stimulus checkpoint")
            setattr(self, key, number)


class Worker(FullBrainIO):
    def __init__(self, learning, event_log):
        import numpy as np
        from doom_learning_v6.calibration import calibrated_brain
        from doom.engine import NeuralControls

        # The public runtime is inference-only.  Training, plasticity and
        # optimizer state stay in the private research workspace; callers may
        # not enable them through this worker protocol.
        if learning is not False:
            raise ValueError("The public worker is frozen; set learning=false")
        self.np = np
        self.brain = calibrated_brain()
        self.manifest = json.loads((REPO / "outputs/doom/malecns_v1/manifest.json").read_text())
        readouts = list(self.manifest["readouts"])
        for record in self.brain.circuit["report"]["DAN"] + self.brain.circuit["report"]["MBON"]:
            readouts.append({key: record[key] for key in ("index", "id", "type")} | {"side": record["soma_side"]})
        self.controls = NeuralControls(readouts, mode="bci")
        # Reuse its exact pulse splitting and telemetry. We NEVER call observe()
        # with invented game health: explicit external events schedule the pulse.
        self.training = FrozenStimulus()
        self._identify_input_ports()
        self.drive_state = self._empty_drive_state()
        self.input_steps = 0
        self.resets = 0
        self.restored_source = None
        self.events = Path(event_log)
        self.events.parent.mkdir(parents=True, exist_ok=True)
        self.record("initialized", source="engineered_external_event", learning=learning)

    def _identify_input_ports(self):
        from doom_learning.common import annotations
        a, b, np = annotations(self.brain.ids), self.brain, self.np
        retina_annotations = a.iloc[b.retina]
        if not retina_annotations.type.eq("R1-R6").all() or not retina_annotations.rootSide.isin(["L", "R"]).all():
            raise ValueError("Mapped retinal neurons need verified R1-R6 type and L/R rootSide")
        if not a.iloc[b.sugar].type.eq("LB3c").all():
            raise ValueError("Sugar input must match annotated LB3c cells")
        self.port_indices = {
            "retina_left": b.retina[retina_annotations.rootSide.eq("L").to_numpy()].copy(),
            "retina_right": b.retina[retina_annotations.rootSide.eq("R").to_numpy()].copy(),
            "sugar": b.sugar.copy(),
        }
        all_indices = np.concatenate(list(self.port_indices.values()))
        if any(not len(ix) for ix in self.port_indices.values()) or len(np.unique(all_indices)) != len(all_indices):
            raise ValueError("Sensory input ports must be nonempty and disjoint")
        self.input_ports = {}
        for name, indices in self.port_indices.items():
            self.input_ports[name] = {
                "normalized_range": [0.0, 1.0], "additional_current_range_mv_equivalent": [0.0, DRIVE_LIMITS_MV[name]],
                "mapping": "additional_current_mv_equivalent = normalized_amplitude * maximum",
                "count": int(len(indices)), "neuron_ids": [str(value) for value in b.ids[indices]],
                "cell_type": "LB3c" if name == "sugar" else "R1-R6",
                "side_annotation": None if name == "sugar" else {"field": "rootSide", "value": "L" if name == "retina_left" else "R"},
                "annotation_source": "MaleCNS v1 annotations.feather, matched by bodyId to retained graph IDs",
                "injection": "MemoryBrain stimulation list -> additive drive[index] -> full-graph native LIF integration",
                "duration": "Only the requested neural interval; omitted ports are zero, without persistent hold",
                "limits": "Chosen engineering current, not calibrated physiological stimulation. LB3c sugar identity is homology-inferred; sugar input is not established positive reinforcement." if name == "sugar" else "Chosen engineering current on the upstream LIF photoreceptor proxy, not calibrated optics or natural graded receptor physiology.",
            }
        self.input_ports_signature = hashlib.sha256(json.dumps(self.input_ports, sort_keys=True).encode()).hexdigest()
        self._init_full_io(a)

    def _empty_drive_state(self):
        return {"last_normalized": validate_neural_drive(), "last_duration_ms": 0.0,
                "cumulative_current_mv_ms_per_neuron": {name: 0.0 for name in DRIVE_LIMITS_MV}}

    def _validate_drive_state(self, value):
        if not isinstance(value, dict) or set(value) != set(self._empty_drive_state()):
            raise ValueError("Invalid direct neural input checkpoint state")
        last = validate_neural_drive(value["last_normalized"])
        duration = value["last_duration_ms"]
        totals = value["cumulative_current_mv_ms_per_neuron"]
        if type(duration) not in (int, float) or not math.isfinite(duration) or not 0 <= duration <= 200:
            raise ValueError("Invalid saved neural drive duration")
        if not isinstance(totals, dict) or set(totals) != set(DRIVE_LIMITS_MV) or any(
            type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in totals.values()
        ):
            raise ValueError("Invalid saved neural drive cumulative exposure")
        return {"last_normalized": last, "last_duration_ms": float(duration),
                "cumulative_current_mv_ms_per_neuron": {name: float(x) for name, x in totals.items()}}

    def _advance(self, rgb, steps, normalized, general_current=None, visual_input_enabled=True):
        """Preserve upstream pulse boundaries while adding all currents per segment."""
        b, training, np = self.brain, self.training, self.np
        active = min(steps, max(0, training.until - b.cursor))
        direct = [(self.port_indices[name], amplitude * DRIVE_LIMITS_MV[name])
                  for name, amplitude in normalized.items() if amplitude]
        if general_current is not None:
            driven = np.flatnonzero(general_current).astype(np.int32)
            if len(driven):
                direct.append((driven, general_current[driven]))
        counts, wall = np.zeros(b.n, dtype=np.int32), 0.0
        if not visual_input_enabled:
            # Do not leave prior optical adaptation as a hidden image drive.
            # Native tonic and the unchanged fixed lamina bias remain present.
            b.luminance.fill(0)
            b.r8_light.fill(0)
        for segment_steps, ppl_active in ((active, True), (steps - active, False)):
            if segment_steps:
                pulses = list(direct)
                if ppl_active:
                    pulses.append((b.circuit["dan"], 4.0))
                if visual_input_enabled:
                    c, t = b.rgb_step(rgb, segment_steps * b.dt, learning=training.enabled,
                                      stimulation=pulses or None)
                else:
                    c, t = b.step(np.zeros(len(b.retina), dtype=np.float32), segment_steps * b.dt,
                                  learning=training.enabled, stimulation=pulses or None)
                counts += c
                wall += t
        b.counts[:] = counts
        training.last_steps = active
        training.delivered_steps += active
        return counts, wall

    def record(self, event, **data):
        with self.events.open("a") as stream:
            stream.write(json.dumps({"event": event, "recorded_at_ms": int(time.time() * 1000),
                                     "neural_cursor": self.brain.cursor, **data}, allow_nan=False) + "\n")

    def learning(self):
        data = self.training.telemetry()
        # Keep the event name explicit: it is an engineered external pulse,
        # never a reward signal or a learned update.
        data["stimulus_source"] = "engineered_external_event"
        return data

    def describe(self):
        return {"protocol": 2, "backend": "malecns-v6-real", "neurons": self.brain.n,
                "edges": int(self.brain.weight.size), "synaptic_contacts": self.manifest["synaptic_contacts"],
                "simulation_ms": self.brain.sim_ms, "learning": self.learning(),
                "input_steps": self.input_steps, "resets": self.resets,
                "event_log": str(self.events), "restored_source": self.restored_source,
                "input_ports": self.input_ports, "input_ports_sha256": self.input_ports_signature,
                "neural_drive_state": self.drive_state,
                "full_brain_io": {"neurons": self.brain.n, "fine_groups": self.catalog["group_count"],
                                  "macro_groups": self.catalog["macro_group_count"], "catalog_path": str(self.catalog_path),
                                  "ids_sha256": self.catalog["ids_sha256"],
                                  "independent_neuron_currents": {"schema": 1, "neurons": self.brain.n,
                                      "stimulus_key": "neuron_currents_file", "file_arrays": {"currents_mv": "float32[n]", "ids": "int64[n]"},
                                      "allowed_directory": str(ROOT / "artifacts/neuron_currents"),
                                      "ordering": "Exact retained graph.ids order, verified from NPZ values and SHA256",
                                      "duration": "Current interval only; omitted input is zero"},
                                  "partition_sha256": self.catalog["partition_sha256"],
                                  "macro_partition_sha256": self.catalog["macro_partition_sha256"],
                                  "general_current_range_mv_equivalent": [-30.0, 30.0],
                                  "ppl_current_is_additional": True, "topology_modified": False},
                "visual_input_enabled_last_step": self.last_visual_input_enabled,
                "decoder": "Fixed upstream experimental BCI readout; scenario must define its own action mapping.",
                "limits": ["Retinal projection and dynamics are approximate engineering models.",
                           "External aversion schedules +4 mV-equivalent PPL101 input for 200 ms; overlapping events extend it.",
                           "Parameter changes do not establish learning efficacy or biological equivalence."]}

    def step(self, request):
        from PIL import Image
        duration = request.get("duration_ms", 28.6)
        if type(duration) not in (int, float) or not math.isfinite(duration) or not 0.1 <= duration <= 200:
            raise ValueError("duration_ms must be finite and within 0.1..200")
        event = request.get("reward_event", {"aversive": False})
        if not isinstance(event, dict) or set(event) != {"aversive"} or type(event["aversive"]) is not bool:
            raise ValueError("Only an explicit {'aversive': bool} reward_event is supported")
        normalized = validate_neural_drive(request.get("neural_drive"))
        visual = request.get("visual_input_enabled", True)
        if type(visual) is not bool:
            raise ValueError("visual_input_enabled must be bool")
        general_current, general_spec = self._prepare_general_stimulation(request.get("general_stimulation"), normalized)
        include_groups = request.get("include_group_stats", False)
        if type(include_groups) is not bool:
            raise ValueError("include_group_stats must be bool")
        group_level = request.get("group_stats_level", "macro")
        if group_level not in ("fine", "macro"):
            raise ValueError("group_stats_level must be fine or macro")
        encoded = request["rgb_png_base64"]
        if not isinstance(encoded, str) or len(encoded) > 12_000_000:
            raise ValueError("PNG base64 payload too large or invalid")
        png = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(png)) as image:
            if image.format != "PNG" or image.mode != "RGB" or not all(1 <= side <= 2048 for side in image.size):
                raise ValueError("Use an RGB PNG with both dimensions within 1..2048")
            rgb = self.np.asarray(image).copy()
        input_rgb_sha256 = hashlib.sha256(rgb.tobytes()).hexdigest()
        if event["aversive"]:
            self.training.events += 1
            self.training.until = self.brain.cursor + 2000
            self.record("external_aversion_scheduled", pulse_steps=2000, duration_ms=200,
                        current_mv_equivalent=4.0, source="engineered_external_event")
        neural_steps = round(duration / self.brain.dt)
        previous_delivered = self.training.delivered_steps
        counts, wall = self._advance(rgb, neural_steps, normalized, general_current, visual)
        if not self.np.isfinite(self.brain.weight).all() or not self.np.isfinite(self.brain.v).all():
            raise RuntimeError("Nonfinite neural state; refusing fabricated output")
        self.input_steps += 1
        self.cumulative_counts += counts.astype(self.np.int64)
        self.last_general_stimulation = general_spec
        self.last_visual_input_enabled = visual
        actual_ms = neural_steps * self.brain.dt
        currents = {name: value * DRIVE_LIMITS_MV[name] for name, value in normalized.items()}
        self.drive_state["last_normalized"] = normalized
        self.drive_state["last_duration_ms"] = actual_ms
        for name, current in currents.items():
            self.drive_state["cumulative_current_mv_ms_per_neuron"][name] += current * actual_ms
        action = self.controls.decode(counts, actual_ms / 1000)
        delivered_steps = self.training.delivered_steps - previous_delivered
        self.record("neural_interval", neural_steps=neural_steps, delivered_stimulus_steps=delivered_steps,
                    input_sha256=input_rgb_sha256, learning=self.training.enabled, visual_input_enabled=visual,
                    neural_drive=normalized, direct_current_mv_equivalent=currents,
                    direct_current_source="engineered_direct_neural_input", general_stimulation=general_spec)
        return {"backend": "malecns-v6-real", "input_step": self.input_steps,
                "input_rgb_sha256": input_rgb_sha256,
                "visual_input_enabled": visual,
                "native_action": {key: action[key] for key in ("turn", "forward", "attack")},
                "readouts": action["readouts"],
                "neural": {"neurons": self.brain.n, "edges": int(self.brain.weight.size),
                           "simulation_ms": self.brain.sim_ms, "duration_ms": actual_ms,
                           "integration_steps": neural_steps, "kernel_wall_seconds": wall,
                           "spikes_this_step": int(counts.sum()), "total_spikes": self.brain.total_spikes,
                           "counts_sha256": hashlib.sha256(counts.tobytes()).hexdigest()},
                "learning": self.learning(),
                "general_stimulation": {"source": "engineered_existing_neuron_current", "specifications": general_spec,
                                        "coverage": self.brain.n,
                                        "ids_sha256": self.catalog["ids_sha256"],
                                        "independent_neuron_vector": any("neuron_currents_file" in item for item in general_spec),
                                        "neuron_current_components": [
                                            {"specification_index": index,
                                             "file_sha256": item["neuron_currents_file"]["sha256"],
                                             "currents_sha256": item["neuron_currents_file"]["currents_sha256"],
                                             "ids_sha256": item["neuron_currents_file"]["ids_sha256"],
                                             "coverage": self.brain.n,
                                             "validation": "Actual loaded float32 values and all original IDs verified before additive integration"}
                                            for index, item in enumerate(general_spec) if "neuron_currents_file" in item],
                                        "driven_neurons": int(self.np.count_nonzero(general_current)),
                                        "minimum_current_mv": float(general_current.min()),
                                        "maximum_current_mv": float(general_current.max()),
                                        "current_vector_sha256": hashlib.sha256(general_current.tobytes()).hexdigest(),
                                        "kernel_final_total_drive_sha256": hashlib.sha256(self.brain.drive.tobytes()).hexdigest(),
                                        "topology_modified": False},
                **({"group_state": self.full_group_state(group_level)} if include_groups else {}),
                "neural_drive": {"source": "engineered_direct_neural_input", "normalized": normalized,
                                 "additional_current_mv_equivalent": currents, "duration_ms": actual_ms,
                                 "port_spikes": {name: int(counts[ix].sum()) for name, ix in self.port_indices.items()},
                                 "final_total_drive_mv_equivalent": {name: {"minimum": float(self.brain.drive[ix].min()),
                                                                            "maximum": float(self.brain.drive[ix].max())}
                                                                     for name, ix in self.port_indices.items()},
                                 "cumulative_current_mv_ms_per_neuron": dict(self.drive_state["cumulative_current_mv_ms_per_neuron"]),
                                 "input_ports_sha256": self.input_ports_signature},
                "stimulus": {"source": "engineered_external_event", "external_aversive_events": self.training.events,
                             "delivered_ms": self.training.delivered_steps * self.brain.dt,
                             "delivered_ms_this_step": delivered_steps * self.brain.dt,
                             "pending_ms": max(0, self.training.until - self.brain.cursor) * self.brain.dt},
                "action_mapping": "Scenario must interpret native_action; these are engineered BCI signals."}

    def reset(self, preserve_weights=True):
        if type(preserve_weights) is not bool:
            raise ValueError("preserve_weights must be bool")
        pending = max(0, self.training.until - self.brain.cursor)
        before = self.brain.memory()["sha256"]
        self.brain.reset(keep_memory=preserve_weights)
        self.controls.rates.fill(0)
        self.training.cancelled_steps += pending
        self.training.until = 0
        self.training.last_steps = 0
        self.drive_state["last_normalized"] = validate_neural_drive()
        self.drive_state["last_duration_ms"] = 0.0
        self.cumulative_counts.fill(0)
        self.cumulative_counts_complete = True
        self.last_general_stimulation = []
        self.last_visual_input_enabled = True
        self.resets += 1
        after = self.brain.memory()["sha256"]
        if preserve_weights and before != after:
            raise RuntimeError("Reset unexpectedly changed learned weights")
        self.record("reset", preserve_weights=preserve_weights, cancelled_stimulus_steps=pending,
                    memory_sha256_before=before, memory_sha256_after=after)
        return self.describe()

    def save(self, target):
        directory = Path(target).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        generation = uuid.uuid4().hex
        stage = directory / (generation + ".partial")
        stage.mkdir()
        self.brain.checkpoint(stage / "brain.npz")
        self.np.save(stage / "decoder.npy", self.controls.rates, allow_pickle=False)
        metadata = {"schema": 3, "backend": "malecns-v6-real", "generation": generation,
                    "training": self.training.state(), "learning_enabled": self.training.enabled,
                    "input_steps": self.input_steps, "resets": self.resets,
                    "input_ports_sha256": self.input_ports_signature, "neural_drive_state": self.drive_state,
                    "full_neural_io": self._full_io_checkpoint(),
                    "memory_sha256": self.brain.memory()["sha256"],
                    "sha256": {name: digest(stage / name) for name in ("brain.npz", "decoder.npy")}}
        (stage / "adapter.json").write_text(json.dumps(metadata, indent=2) + "\n")
        stage.rename(directory / generation)
        pointer = directory / "latest.partial"
        pointer.write_text(json.dumps({"generation": generation}) + "\n")
        pointer.replace(directory / "latest.json")
        self.record("checkpoint_saved", path=str(directory / generation), memory_sha256=metadata["memory_sha256"])
        return {"path": str(directory), "generation": generation, "memory_sha256": metadata["memory_sha256"],
                "simulation_ms": self.brain.sim_ms, "sha256": metadata["sha256"]}

    def restore(self, target):
        directory = Path(target).resolve()
        if (directory / "latest.json").is_file():
            generation = json.loads((directory / "latest.json").read_text())["generation"]
            if not re.fullmatch(r"[a-f0-9]{32}", generation):
                raise ValueError("Invalid checkpoint generation")
            directory = directory / generation
        native = (directory / "adapter.json").is_file()
        metadata = json.loads((directory / ("adapter.json" if native else "state.json")).read_text())
        if native and metadata.get("schema") not in (1, 2, 3):
            raise ValueError("Unsupported adapter checkpoint schema")
        restored_drive_state = self._empty_drive_state()
        if native and metadata.get("schema") in (2, 3):
            if metadata.get("input_ports_sha256") != self.input_ports_signature:
                raise ValueError("Checkpoint neural input port identity or current scale mismatch")
            restored_drive_state = self._validate_drive_state(metadata.get("neural_drive_state"))
        hashes = metadata.get("sha256", {})
        if set(hashes) != {"brain.npz", "decoder.npy"}:
            raise ValueError("Checkpoint must contain hashed brain.npz and decoder.npy")
        for name, expected in hashes.items():
            if digest(directory / name) != expected:
                raise ValueError(f"Checkpoint checksum mismatch: {name}")
        rates = self.np.load(directory / "decoder.npy", allow_pickle=False)
        if rates.shape != self.controls.rates.shape or rates.dtype != self.controls.rates.dtype or not self.np.isfinite(rates).all():
            raise ValueError("Incompatible decoder checkpoint")
        with self.np.load(directory / "brain.npz", allow_pickle=False) as saved:
            for name in ["weight", *self.brain.fields]:
                if not self.np.isfinite(saved[name]).all():
                    raise ValueError("Nonfinite saved neural state")
        self.brain.restore(directory / "brain.npz")  # Enforces upstream graph/model/config hashes.
        self.controls.rates[:] = rates
        if native:
            self.training.restore(metadata["training"])
            self.input_steps = metadata["input_steps"]
            self.resets = metadata["resets"]
            self.restored_source = "connectome_adapter"
        else:
            # Transfer learned brain state across scenarios. Genuine Doom damage
            # counts are not relabelled as external events; pending pulses cancel.
            self.training.until = self.brain.cursor
            self.training.events = self.training.delivered_steps = self.training.last_steps = 0
            self.training.terminal_events = self.training.cancelled_steps = 0
            self.restored_source = "upstream_doom_checkpoint; pending scenario stimulus cancelled"
        self.brain.weights_frozen = not self.training.enabled
        self.drive_state = restored_drive_state
        self._restore_full_io(metadata.get("full_neural_io") if native else None)
        self.record("checkpoint_restored", source=self.restored_source, path=str(directory))
        return self.describe()

    def dispatch(self, request):
        operation = request["op"]
        if operation == "step":
            return self.step(request)
        if operation == "describe":
            return self.describe()
        if operation == "group_catalog":
            return self.catalog
        if operation == "group_state":
            return self.full_group_state(request.get("level", "macro"))
        if operation == "read_state":
            return self.read_full_state(request.get("path"), request.get("include_graph", False))
        if operation == "neuron_details":
            return self.neuron_details(request["neuron_ids"])
        if operation == "reset":
            return self.reset(request.get("preserve_weights", True))
        if operation == "set_learning":
            enabled = request["enabled"]
            if enabled is not False:
                raise ValueError("The public worker is frozen; learning cannot be enabled")
            return self.describe()
        if operation == "save":
            return self.save(request["path"])
        if operation == "restore":
            return self.restore(request["path"])
        raise ValueError(f"Unsupported operation: {operation}")


def main():
    worker = None
    for line in sys.stdin:
        request = {}
        try:
            if len(line) > 13_000_000:
                raise ValueError("Request exceeds the maximum PNG payload size")
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("Request must be an object")
            if request.get("op") == "close":
                print(json.dumps({"id": request.get("id"), "ok": True, "result": {"closed": True}}), flush=True)
                return
            with redirect_stdout(sys.stderr):
                if request.get("op") == "init":
                    if worker is not None:
                        raise ValueError("Worker already initialized")
                    worker = Worker(request["learning"], request["event_log"])
                    result = worker.describe()
                elif worker is None:
                    raise ValueError("Initialize the worker first")
                else:
                    result = worker.dispatch(request)
            response = {"id": request.get("id"), "ok": True, "result": result}
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            response = {"id": request.get("id") if isinstance(request, dict) else None,
                        "ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
