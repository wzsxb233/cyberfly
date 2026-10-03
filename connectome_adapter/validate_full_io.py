"""Independent full-node coverage, intervention and checkpoint validation."""
import json
from pathlib import Path
import time
import uuid

import numpy as np

from .client import ConnectomeClient, ConnectomeError


ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "artifacts" / ("full-brain-validation-" + uuid.uuid4().hex[:12])
    output.mkdir(parents=True)
    report = {"status": "running", "scope": "Real full retained graph, complete reads, bounded currents, and exact state persistence; not task mastery."}
    started = time.monotonic()
    brain = None
    try:
        brain = ConnectomeClient(learning=False, timeout=180, log_dir=output / "worker")
        catalog = brain.groups()
        assert catalog["coverage"] == 166700
        for key in ("groups", "macro_groups"):
            assert sum(group["count"] for group in catalog[key]) == 166700
        initial = brain.save(output / "initial")
        rgb = np.zeros((84, 84, 3), dtype=np.uint8)
        baseline = brain.step(rgb, duration_ms=100, include_group_stats=True)
        baseline_arrays, _ = brain.read_arrays()
        for key in ("ids", "v", "counts", "cumulative_counts", "group_index", "macro_group_index"):
            assert baseline_arrays[key].shape == (166700,)
        assert len(np.unique(baseline_arrays["ids"])) == 166700
        assert np.bincount(baseline_arrays["group_index"]).tolist() == [g["count"] for g in catalog["groups"]]
        assert np.bincount(baseline_arrays["macro_group_index"]).tolist() == [g["count"] for g in catalog["macro_groups"]]
        assert baseline_arrays["soma_valid"].sum() == 139662
        assert np.isfinite(baseline_arrays["soma_xyz"][baseline_arrays["soma_valid"]]).all()
        fine = max((group for group in catalog["groups"] if group["cell_type"] == "R1-R6" and group["root_side"] == "L"), key=lambda group: group["count"])
        selected = baseline_arrays["group_index"] == fine["index"]
        baseline_target_spikes = int(baseline_arrays["counts"][selected].sum())
        brain.restore(output / "initial")
        driven = brain.step(rgb, duration_ms=100, general_stimulation=[{"group_id": fine["group_id"], "current_mv": 10.0}], include_group_stats=True)
        driven_arrays, _ = brain.read_arrays()
        driven_target_spikes = int(driven_arrays["counts"][selected].sum())
        assert driven["neural"]["counts_sha256"] != baseline["neural"]["counts_sha256"]
        assert driven_target_spikes > baseline_target_spikes
        assert np.all(driven_arrays["drive"][selected] == 10.0)
        assert driven["learning"]["sha256"] == baseline["learning"]["sha256"]
        print(json.dumps({"stage": "fine_group_intervention", "group": fine["group_id"], "neurons": fine["count"],
                          "before_spikes": baseline_target_spikes, "after_spikes": driven_target_spikes}), flush=True)

        dense = [{"macro_group_currents_mv": [0.5] * catalog["macro_group_count"]}]
        all_driven = brain.step(rgb, duration_ms=28.6, general_stimulation=dense, include_group_stats=True)
        assert all_driven["general_stimulation"]["driven_neurons"] == 166700
        assert len(all_driven["group_state"]["group_ids"]) == 47
        assert sum(all_driven["group_state"]["spikes"]) == all_driven["neural"]["spikes_this_step"]
        first_id = str(baseline_arrays["ids"][np.flatnonzero(selected)[0]])
        single = brain.step(rgb, duration_ms=28.6, general_stimulation=[{"neuron_ids": [first_id], "current_mv": -5.0}])
        assert single["general_stimulation"]["driven_neurons"] == 1
        detail = brain.neuron_details([first_id])["neurons"][0]
        assert detail["id"] == first_id and detail["group"]["group_id"] == fine["group_id"]
        for invalid in ([{"neuron_ids": ["0"], "current_mv": 1.0}], [{"macro_group_currents_mv": [0.0]}]):
            try:
                brain.step(rgb, general_stimulation=invalid)
            except (ValueError, ConnectomeError):
                pass
            else:
                raise AssertionError("Unknown neuron or wrong dense length accepted")

        brain.set_learning(True)
        brain.step(rgb, duration_ms=100, general_stimulation=dense, reward_event={"aversive": True})
        combined = brain.step(rgb, duration_ms=100, general_stimulation=dense, include_group_stats=True)
        assert combined["stimulus"]["delivered_ms"] == 200.0 and combined["stimulus"]["pending_ms"] == 0.0
        saved = brain.save(output / "learned_full_state")
        expected_arrays, _ = brain.read_arrays()
        brain.step(rgb, duration_ms=28.6)
        restored = brain.restore(output / "learned_full_state")
        restored_arrays, _ = brain.read_arrays()
        for key in ("ids", "v", "counts", "cumulative_counts", "drive", "group_index", "macro_group_index"):
            assert np.array_equal(expected_arrays[key], restored_arrays[key]), f"Full-state restore mismatch: {key}"
        assert restored["learning"]["sha256"] == saved["memory_sha256"]
        assert int(restored_arrays["cumulative_counts"].sum()) == combined["neural"]["total_spikes"]
        groups = brain.group_state()
        (ROOT / "artifacts/full_brain/group-state.json").write_text(json.dumps(groups, indent=2, allow_nan=False) + "\n")
        fine_groups = brain.group_state("fine")
        assert len(fine_groups["group_ids"]) == 23163
        assert sum(fine_groups["spikes"]) == sum(groups["spikes"])

        snapshot = brain.read_state(output / "export", include_graph=True)
        with np.load(snapshot["path"], allow_pickle=False) as exported, np.load(ROOT / "vendor/doomfly/outputs/doom/malecns_v1/graph.npz", allow_pickle=False) as released:
            assert exported["ptr"].shape == (166701,)
            assert exported["post"].shape == exported["weight"].shape == (25582938,)
            assert np.array_equal(exported["ids"], released["ids"])
            assert np.array_equal(exported["ptr"], released["ptr"])
            assert np.array_equal(exported["post"], released["post"])
        report.update(status="passed", coverage=166700, fine_groups=23163, macro_groups=47,
                      valid_soma_coordinates=139662, missing_soma_coordinates=27038,
                      fine_intervention={"group": fine, "baseline_spikes": baseline_target_spikes, "driven_spikes": driven_target_spikes},
                      dense_macro_drives_every_neuron=True, single_existing_id_write=True,
                      exact_full_state_restore=True, cumulative_counts_exact=True, ppl_delivered_ms=200.0,
                      topology_matches_all_released_csr_entries=True, graph_snapshot=snapshot,
                      group_state_path=str(ROOT / "artifacts/full_brain/group-state.json"))
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        if brain is not None:
            brain.close()
            report["owned_worker_exit_code"] = brain.process.returncode
        report["elapsed_seconds"] = time.monotonic() - started
        (output / "evidence.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"status": report["status"], "output": str(output), "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
