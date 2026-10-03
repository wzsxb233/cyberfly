"""Send actual SIGINT while the full neural worker is processing an interval."""
import json
import io
import os
from pathlib import Path
import signal
import threading
import time
import uuid

import numpy as np
from PIL import Image

from .client import ConnectomeClient, ConnectomeStopped


ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "artifacts" / ("brain-interrupt-" + uuid.uuid4().hex[:12])
    output.mkdir(parents=True)
    report = {"status": "running", "scope": "Actual controller SIGINT, completed neural interval, checkpoint and identical continuation"}
    brain = None
    timer = None
    try:
        brain = ConnectomeClient(learning=True, timeout=180, log_dir=output)
        assert os.getpgid(brain.process.pid) != os.getpgrp(), "Worker must survive controller-group Ctrl+C"
        rgb = np.zeros((84, 84, 3), np.uint8)
        encoded = io.BytesIO()
        Image.fromarray(rgb).save(encoded, format="PNG")
        png = encoded.getvalue()
        drive = {"retina_left": 0.5, "retina_right": 0.7, "sugar": 0.2}
        # A 200 ms full-graph interval takes appreciably longer than this timer.
        before = brain.describe()["simulation_ms"]
        timer = threading.Timer(0.02, lambda: os.kill(os.getpid(), signal.SIGINT))
        timer.start()
        try:
            brain.step(png, duration_ms=200, neural_drive=drive, reward_event={"aversive": True})
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError("SIGINT did not interrupt the caller")
        timer.join()
        interrupted = brain.last_interrupted_request
        assert interrupted and interrupted["operation"] == "step" and interrupted["completed"]
        actual = interrupted["result"]
        assert actual["neural"]["simulation_ms"] - before == 200.0
        assert actual["neural"]["spikes_this_step"] > 0 and actual["stimulus"]["delivered_ms"] == 200.0
        assert brain.stop_requested and brain.process.poll() is None
        saved = brain.save(output / "checkpoint_after_sigint")
        assert saved["memory_sha256"] == actual["learning"]["sha256"]
        brain.clear_stop()
        expected = brain.step(rgb, duration_ms=28.6, neural_drive=drive)
        restored = brain.restore(output / "checkpoint_after_sigint")
        assert restored["learning"]["sha256"] == saved["memory_sha256"]
        replay = brain.step(rgb, duration_ms=28.6, neural_drive=drive)
        assert expected["neural"]["counts_sha256"] == replay["neural"]["counts_sha256"]
        assert expected["learning"]["sha256"] == replay["learning"]["sha256"]
        brain.request_stop()
        assert brain.wait_until_idle(timeout=5)
        try:
            brain.step(rgb)
        except ConnectomeStopped:
            pass
        else:
            raise AssertionError("Cooperative stop allowed an additional step")
        # Save remains valid after a cooperative stop.
        cooperative = brain.save(output / "checkpoint_after_cooperative_stop")
        report.update(status="passed", interrupted_interval=actual,
                      interrupt_checkpoint=saved, cooperative_checkpoint=cooperative,
                      exact_neural_and_weight_continuation=True, owned_worker_alive_after_sigint=True)
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        if timer is not None:
            timer.cancel()
        if brain is not None:
            brain.close()
            report["owned_worker_exit_code"] = brain.process.returncode
        (output / "evidence.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"status": report["status"], "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
