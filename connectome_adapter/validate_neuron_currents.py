"""Real full-graph independent-neuron current, integrity and replay checks."""
import hashlib
import json
from pathlib import Path
import time
import uuid

import numpy as np

from .client import ConnectomeClient, ConnectomeError
from .neuron_currents import CURRENT_ROOT, array_sha256, write_neuron_currents


ROOT = Path(__file__).resolve().parents[1]


def main():
    tag = uuid.uuid4().hex[:12]
    output = ROOT / "artifacts" / ("independent-neuron-check-" + tag)
    output.mkdir(parents=True)
    current_dir = CURRENT_ROOT / ("validation-" + tag)
    current_dir.mkdir(parents=True)
    started, brain = time.monotonic(), None
    report = {"status": "running", "scope": "Actual individual neuron currents and full graph propagation; engineering interface, not validated natural semantics or task competence."}
    try:
        brain = ConnectomeClient(learning=False, timeout=180, log_dir=output / "worker")
        arrays, initial_descriptor = brain.read_arrays()
        ids = arrays["ids"]
        assert ids.shape == (166700,) and ids.dtype == np.int64
        assert initial_descriptor["duration_ms"] == 0 and initial_descriptor["simulation_ms"] == 0
        assert initial_descriptor["ids_sha256"] == array_sha256(ids)
        catalog = brain.groups()
        classes = {}
        for group in catalog["macro_groups"]:
            classes[group["superclass"]] = classes.get(group["superclass"], 0) + group["count"]
        brain.save(output / "initial")
        dark, bright = np.zeros((48, 64, 3), np.uint8), np.full((48, 64, 3), 255, np.uint8)
        baseline = brain.step(dark, duration_ms=100, visual_input_enabled=False)
        baseline_arrays, _ = brain.read_arrays()
        brain.restore(output / "initial")
        bright_off = brain.step(bright, duration_ms=100, visual_input_enabled=False)
        assert bright_off["neural"]["counts_sha256"] == baseline["neural"]["counts_sha256"]
        assert bright_off["input_rgb_sha256"] != baseline["input_rgb_sha256"]
        brain.restore(output / "initial")
        bright_on = brain.step(bright, duration_ms=100)
        assert bright_on["neural"]["counts_sha256"] != baseline["neural"]["counts_sha256"]
        # Toggling off removes both prior light-adaptation channels immediately.
        brain.step(bright, duration_ms=0.1, visual_input_enabled=False)
        dark_after_light, _ = brain.read_arrays()
        assert np.array_equal(dark_after_light["drive"], baseline_arrays["drive"])

        current = np.linspace(-18.0, 18.0, 166700, dtype=np.float32)
        assert np.unique(current).size == 166700 and np.count_nonzero(current) == 166700
        descriptor = brain.write_neuron_currents(current, ids=ids, directory=current_dir)
        assert len(json.dumps(descriptor)) < 1000
        brain.restore(output / "initial")
        driven = brain.step(dark, duration_ms=100, visual_input_enabled=False,
                            general_stimulation=[{"neuron_currents_file": descriptor}])
        driven_arrays, state_descriptor = brain.read_arrays()
        assert driven["general_stimulation"]["current_vector_sha256"] == descriptor["currents_sha256"]
        assert driven["general_stimulation"]["driven_neurons"] == 166700
        assert driven["general_stimulation"]["coverage"] == 166700
        assert driven["general_stimulation"]["independent_neuron_vector"]
        assert driven["neural"]["counts_sha256"] != baseline["neural"]["counts_sha256"]
        residual = driven_arrays["drive"] - baseline_arrays["drive"] - current
        assert np.max(np.abs(residual)) < 4e-6
        assert driven["learning"]["sha256"] == baseline["learning"]["sha256"]
        assert state_descriptor["duration_ms"] == 100 and not state_descriptor["visual_input_enabled"]
        print(json.dumps({"stage": "individual_vector", "independent_values": np.unique(current).size,
                          "baseline_spikes": baseline["neural"]["spikes_this_step"],
                          "driven_spikes": driven["neural"]["spikes_this_step"], "current_sha256": descriptor["currents_sha256"]}), flush=True)

        rejects = {}
        invalid = {"file_sha": descriptor | {"sha256": "0" * 64},
                   "declared_ids_sha": descriptor | {"ids_sha256": "0" * 64},
                   "current_array_sha": descriptor | {"currents_sha256": "0" * 64}}
        for name, bad_currents, bad_ids in [
            ("actual_id_order", current, ids[::-1].copy()),
            ("actual_unknown_id", current, np.concatenate((np.array([0], dtype=np.int64), ids[1:]))),
            ("shape", current[:-1], ids),
            ("dtype", current.astype(np.float64), ids),
            ("nonfinite", np.where(np.arange(166700) == 1, np.float32(np.nan), current).astype(np.float32), ids),
            ("current_range", np.full(166700, 31, np.float32), ids),
        ]:
            path = current_dir / ("invalid-" + name + ".npz")
            np.savez(path, currents_mv=bad_currents, ids=bad_ids)
            invalid[name] = descriptor | {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                         "currents_sha256": array_sha256(bad_currents)}
        outside = output / "out-of-allowlist.npz"
        outside.write_bytes(Path(descriptor["path"]).read_bytes())
        invalid["path_outside_allowlist"] = descriptor | {"path": str(outside)}
        before_invalid = brain.describe()["simulation_ms"]
        for name, candidate in invalid.items():
            try:
                brain.step(dark, visual_input_enabled=False, general_stimulation=[{"neuron_currents_file": candidate}])
            except (ValueError, ConnectomeError) as exc:
                rejects[name] = str(exc)
            else:
                raise AssertionError(f"Invalid descriptor accepted: {name}")
        assert brain.describe()["simulation_ms"] == before_invalid

        signed_zero = np.zeros(166700, dtype=np.float32)
        signed_zero[::2] = -0.0
        zero_descriptor = brain.write_neuron_currents(signed_zero, ids=ids, directory=current_dir)
        zero_step = brain.step(dark, duration_ms=0.1, visual_input_enabled=False,
                               general_stimulation=[{"neuron_currents_file": zero_descriptor}])
        assert zero_step["general_stimulation"]["current_vector_sha256"] == zero_descriptor["currents_sha256"]
        assert zero_step["general_stimulation"]["driven_neurons"] == 0

        brain.set_learning(True)
        first = brain.step(dark, duration_ms=100, visual_input_enabled=False,
                           general_stimulation=[{"neuron_currents_file": descriptor}], reward_event={"aversive": True})
        second = brain.step(dark, duration_ms=100, visual_input_enabled=False,
                            general_stimulation=[{"neuron_currents_file": descriptor}])
        assert first["stimulus"]["pending_ms"] == 100 and second["stimulus"]["pending_ms"] == 0
        assert second["stimulus"]["delivered_ms"] == 200
        saved = brain.save(output / "after_individual_currents")
        expected, _ = brain.read_arrays()
        Path(descriptor["path"]).unlink()
        brain.step(dark, duration_ms=28.6, visual_input_enabled=False)
        restored = brain.restore(output / "after_individual_currents")
        restored_arrays, restored_descriptor = brain.read_arrays()
        for key in ("ids", "v", "counts", "cumulative_counts", "drive"):
            assert np.array_equal(expected[key], restored_arrays[key]), key
        assert restored["learning"]["sha256"] == saved["memory_sha256"]
        assert not restored_descriptor["visual_input_enabled"]
        next_step = brain.step(dark, duration_ms=1, visual_input_enabled=False)
        assert next_step["general_stimulation"]["driven_neurons"] == 0
        snapshot = brain.read_state(output / "whole-graph", include_graph=True)
        with np.load(snapshot["path"], allow_pickle=False) as actual, np.load(ROOT / "vendor/doomfly/outputs/doom/malecns_v1/graph.npz", allow_pickle=False) as source:
            assert actual["weight"].shape == actual["post"].shape == (25582938,)
            assert np.array_equal(actual["ids"], source["ids"])
            assert np.array_equal(actual["ptr"], source["ptr"])
            assert np.array_equal(actual["post"], source["post"])
        report.update(status="passed", neurons=166700, edges=25582938, independent_current_values=166700,
                      actual_current_sha256=descriptor["currents_sha256"], maximum_drive_residual=float(np.max(np.abs(residual))),
                      baseline_spikes=baseline["neural"]["spikes_this_step"], driven_spikes=driven["neural"]["spikes_this_step"],
                      rejects=rejects, rejected_requests_did_not_advance_time=True,
                      visual_off_makes_same_neural_result_for_different_rgb=True,
                      visual_on_changes_neural_response=True, prior_visual_adaptation_cleared=True,
                      input_file_deleted_then_exact_checkpoint_restore=True, no_stimulus_hold=True,
                      signed_zero_current_bytes_preserved=True,
                      full_graph_matches_all_released_ids_ptr_post=True, ppl_delivered_ms=200,
                      full_graph_snapshot=snapshot, superclass_counts=classes,
                      sensory_candidate_count=sum(classes[key] for key in ("cb_sensory", "ol_sensory", "vnc_sensory")),
                      motor_candidate_count=sum(classes[key] for key in ("cb_motor", "vnc_motor", "descending_neuron")))
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        if brain is not None:
            brain.close()
            report["owned_worker_exit_code"] = brain.process.returncode
        report["elapsed_seconds"] = time.monotonic() - started
        (output / "evidence.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        print(json.dumps({"status": report["status"], "output": str(output), "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
