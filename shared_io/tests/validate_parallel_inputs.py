"""Real preserved vision/ports/old currents plus budgeted model input."""
import json
from pathlib import Path
import time
import uuid

import numpy as np

from connectome_adapter.client import ConnectomeClient
from connectome_adapter.neuron_currents import load_neuron_currents
from shared_io.parallel_inputs import array_sha256, prepare_parallel_stimulation, expand_general_stimulation

ROOT = Path(__file__).resolve().parents[2]


def main():
    out = ROOT / "artifacts" / ("parallel-input-check-" + uuid.uuid4().hex[:12])
    out.mkdir(parents=True)
    report = {"status": "running", "scope": "Actual additive neural inputs, clipping only new model current; no behavioral efficacy conclusion"}
    brain = None
    started = time.monotonic()
    try:
        brain = ConnectomeClient(learning=False, timeout=180, log_dir=out / "worker")
        catalog, description = brain.groups(), brain.describe()
        arrays, _ = brain.read_arrays()
        ids = arrays["ids"]
        snapshot = brain.read_state(out / "before")
        original = brain.save(out / "initial")
        rgb = np.full((48, 64, 3), 128, np.uint8)
        # Already executed float32 action -> float64 normalization, matching env.
        action = np.array([-.4, 0, .2], dtype=np.float32)
        normalized = (action.astype(np.float64) + 1) / 2
        ports = dict(zip(("retina_left", "retina_right", "sugar"), normalized.tolist()))
        old_file_values = np.full(166700, .125, np.float32)
        old_file = brain.write_neuron_currents(old_file_values, ids=ids)
        sugar_id = description["input_ports"]["sugar"]["neuron_ids"][0]
        existing = [{"neuron_currents_file": old_file},
                    {"macro_group_currents_mv": [.25] * 47},
                    {"neuron_ids": [sugar_id], "current_mv": 1.25}]
        baseline = brain.step(rgb, duration_ms=100, neural_drive=ports, general_stimulation=existing)
        baseline_arrays, _ = brain.read_arrays()
        assert baseline["visual_input_enabled"] is True
        requested = np.linspace(21, 29, 166700, dtype=np.float32)
        requested_file = brain.write_neuron_currents(requested, ids=ids)
        plan = prepare_parallel_stimulation(brain, snapshot, requested_file, existing, ports)
        (out / "parallel-plan.json").write_text(json.dumps(plan, indent=2))
        assert plan["preserved_general"] == existing and plan["preserved_neural_drive"] == ports
        assert plan["requested_vs_applied"]["changed_neurons"] > 0
        applied = load_neuron_currents(plan["applied_currents_file"], ids, catalog["ids_sha256"])
        assert np.all(applied <= requested) and np.all(applied >= 0)
        exact_general = expand_general_stimulation(existing, arrays, catalog) + applied
        assert array_sha256(exact_general) == plan["expected_general_currents_sha256"]
        assert not plan["existing_inputs_modified"]
        brain.restore(out / "initial")
        combined = brain.step(rgb, duration_ms=100, neural_drive=ports,
                              general_stimulation=existing + [{"neuron_currents_file": plan["applied_currents_file"]}])
        combined_arrays, _ = brain.read_arrays()
        assert combined["visual_input_enabled"] is True
        assert combined["neural_drive"]["normalized"] == ports
        assert combined["input_rgb_sha256"] == baseline["input_rgb_sha256"]
        assert combined["general_stimulation"]["current_vector_sha256"] == plan["expected_general_currents_sha256"]
        components = combined["general_stimulation"]["neuron_current_components"]
        assert len(components) == 2
        assert components[0]["currents_sha256"] == old_file["currents_sha256"]
        assert components[1]["currents_sha256"] == plan["applied_currents_file"]["currents_sha256"]
        # Same initial state and pixels: any total-drive delta is the new input.
        residual = combined_arrays["drive"] - baseline_arrays["drive"] - applied
        assert np.max(np.abs(residual)) < 8e-6
        assert combined["neural"]["counts_sha256"] != baseline["neural"]["counts_sha256"]
        assert combined["learning"]["sha256"] == baseline["learning"]["sha256"]
        # Existing zero-input values and signed model zeros remain exactly intact.
        zeros = np.zeros(166700, np.float32); zeros[::2] = -0.0
        zero_file = brain.write_neuron_currents(zeros, ids=ids)
        zero_plan = prepare_parallel_stimulation(brain, snapshot, zero_file, existing, ports)
        assert zero_plan["requested_vs_applied"]["changed_neurons"] == 0
        rejected = False
        try:
            prepare_parallel_stimulation(brain, snapshot, requested_file,
                [{"macro_group_currents_mv": [30.0] * 47}], ports)
        except ValueError:
            rejected = True
        assert rejected
        brain.set_learning(True)
        aversive = brain.step(rgb, duration_ms=100, neural_drive=ports,
            general_stimulation=existing + [{"neuron_currents_file": plan["applied_currents_file"]}], reward_event={"aversive": True})
        end = brain.step(rgb, duration_ms=100, neural_drive=ports, general_stimulation=existing)
        assert aversive["stimulus"]["pending_ms"] == 100 and end["stimulus"]["delivered_ms"] == 200
        assert end["stimulus"]["pending_ms"] == 0 and end["visual_input_enabled"] is True
        assert end["general_stimulation"]["current_vector_sha256"] == plan["existing_general_currents_sha256"]
        # Port identity and voltage/spike arrays remain the complete retained graph.
        assert combined_arrays["v"].shape == combined_arrays["counts"].shape == (166700,)
        assert combined["neural"]["edges"] == 25582938
        report.update(status="passed", neurons=166700, edges=25582938,
            old_visual_preserved=True, old_ports_preserved=ports, old_general_preserved=True,
            old_file_and_new_file_both_integrated=True,
            requested_vs_applied=plan["requested_vs_applied"],
            total_general_sha256=plan["expected_general_currents_sha256"],
            maximum_drive_delta_error=float(np.max(np.abs(residual))),
            baseline_spikes=baseline["neural"]["spikes_this_step"],
            combined_spikes=combined["neural"]["spikes_this_step"],
            new_input_omission_keeps_existing_input=True, signed_zero_unchanged=True,
            invalid_existing_budget_rejected=True, ppl_delivered_ms=200,
            components=components)
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        if brain is not None:
            brain.close()
            report["worker_exit_code"] = brain.process.returncode
        report["seconds"] = time.monotonic() - started
        (out / "evidence.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({"status": report["status"], "output": str(out), "seconds": report["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
