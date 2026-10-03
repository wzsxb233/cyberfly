"""Full-graph causal input validation, not a language or task mastery benchmark."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import uuid

import numpy as np

from .client import ConnectomeClient


ROOT = Path(__file__).resolve().parents[1]


def main():
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    output = ROOT / "artifacts" / f"connectome-direct-drive-{tag}"
    output.mkdir(parents=True)
    rgb = np.zeros((84, 84, 3), dtype=np.uint8)
    started = time.monotonic()
    evidence = {"status": "running", "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                "conditions": {}, "limitations": "Same-state same-RGB intervention establishes implemented neural causality, not semantic understanding, physiological validity, trained behavior, or task improvement."}
    print(json.dumps({"stage": "initializing_full_graph", "output": str(output)}), flush=True)
    brain = None
    try:
        brain = ConnectomeClient(learning=False, timeout=180, log_dir=output / "worker")
        info = brain.describe()
        assert info["neurons"] == 166700 and info["edges"] == 25582938
        assert set(info["input_ports"]) == {"retina_left", "retina_right", "sugar"}
        evidence["brain"] = {key: info[key] for key in ("neurons", "edges", "input_ports", "input_ports_sha256")}
        baseline_checkpoint = brain.save(output / "identical_initial_state")
        evidence["baseline_checkpoint"] = baseline_checkpoint
        baseline = brain.step(rgb, duration_ms=100)
        brain.restore(output / "identical_initial_state")
        repeated = brain.step(rgb, duration_ms=100)
        assert baseline["neural"]["counts_sha256"] == repeated["neural"]["counts_sha256"], "Identical-state control was not deterministic"
        evidence["control_repeat_counts_exact"] = True
        evidence["conditions"]["rgb_only"] = baseline
        for port in info["input_ports"]:
            brain.restore(output / "identical_initial_state")
            result = brain.step(rgb, duration_ms=100, neural_drive={port: 1.0})
            assert result["neural"]["counts_sha256"] != baseline["neural"]["counts_sha256"], f"{port} failed to affect actual neural counts"
            assert result["neural_drive"]["port_spikes"][port] > baseline["neural_drive"]["port_spikes"][port], f"{port} did not excite its declared real neurons"
            maximum = info["input_ports"][port]["additional_current_range_mv_equivalent"][1]
            measured = result["neural_drive"]["final_total_drive_mv_equivalent"][port]
            assert measured == {"minimum": maximum, "maximum": maximum}, "Native input currents did not match the documented mapping"
            assert result["learning"]["sha256"] == baseline["learning"]["sha256"], "Frozen A/B unexpectedly changed weights"
            evidence["conditions"][port] = result
            print(json.dumps({"stage": "direct_input_verified", "port": port,
                              "target_count": info["input_ports"][port]["count"],
                              "baseline_port_spikes": baseline["neural_drive"]["port_spikes"][port],
                              "driven_port_spikes": result["neural_drive"]["port_spikes"][port],
                              "total_spikes": result["neural"]["spikes_this_step"]}), flush=True)

        before_invalid = brain.describe()["simulation_ms"]
        rejected = []
        for drive in ({"arbitrary_motor": 1}, {"sugar": -0.1}, {"sugar": 1.01}, {"sugar": True}, {"sugar": float("nan")}):
            try:
                brain.step(rgb, neural_drive=drive)
            except ValueError:
                rejected.append(str(drive))
            else:
                raise AssertionError("Invalid neural current request accepted")
        assert before_invalid == brain.describe()["simulation_ms"]
        evidence["invalid_requests_rejected_without_advancing"] = rejected

        brain.restore(output / "identical_initial_state")
        brain.set_learning(True)
        combined = {"retina_left": 0.2, "retina_right": 0.3, "sugar": 0.4}
        first = brain.step(rgb, duration_ms=28.6, neural_drive=combined, reward_event={"aversive": True})
        assert abs(first["stimulus"]["delivered_ms"] - 28.6) < 1e-8
        assert abs(first["stimulus"]["pending_ms"] - 171.4) < 1e-8
        checkpoint = brain.save(output / "combined_pending_ppl_checkpoint")
        saved = brain.describe()
        second = brain.step(rgb, duration_ms=171.4, neural_drive=combined)
        restored = brain.restore(output / "combined_pending_ppl_checkpoint")
        assert restored["neural_drive_state"] == saved["neural_drive_state"]
        assert restored["learning"]["sha256"] == saved["learning"]["sha256"]
        replay = brain.step(rgb, duration_ms=171.4, neural_drive=combined)
        assert second["neural"]["counts_sha256"] == replay["neural"]["counts_sha256"]
        assert second["learning"]["sha256"] == replay["learning"]["sha256"]
        assert second["neural_drive"] == replay["neural_drive"]
        assert replay["stimulus"]["delivered_ms"] == 200.0 and replay["stimulus"]["pending_ms"] == 0.0
        zeroed = brain.step(rgb, duration_ms=28.6)
        assert all(value == 0.0 for value in zeroed["neural_drive"]["normalized"].values())
        before_reset = brain.describe()
        reset = brain.reset(preserve_weights=True)
        assert all(value == 0.0 for value in reset["neural_drive_state"]["last_normalized"].values())
        assert reset["learning"]["sha256"] == before_reset["learning"]["sha256"]
        evidence["combined_inputs_and_ppl"] = {"first": first, "completion": replay,
                                              "checkpoint": checkpoint,
                                              "input_state_restore_exact": True,
                                              "neural_and_weight_continuation_exact": True,
                                              "ppl_delivered_ms": 200.0,
                                              "omitted_ports_clear_current": True,
                                              "reset_preserves_weights_clears_inputs": True}
        evidence["status"] = "passed"
    except BaseException as exc:
        evidence["status"], evidence["error"] = "failed", repr(exc)
        raise
    finally:
        if brain is not None:
            brain.close()
            evidence["owned_worker_exit_code"] = brain.process.returncode
        evidence["elapsed_seconds"] = time.monotonic() - started
        (output / "evidence.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"status": evidence["status"], "output": str(output),
                          "elapsed_seconds": evidence["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
