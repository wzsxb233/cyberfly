"""Start/stop/status/smoke the isolated local MiniCPM runtime."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
ROOT = Path(os.environ.get("CYBERFLY_MINICPM_ROOT", "/root/.cache/cyberfly/minicpm-runtime"))
PROCESSES = ROOT / "processes.json"
RUNTIME = HERE / "runtime.json"
MODEL = "MiniCPM-o-4.5-Q4_K_M"


def save_runtime(status, **extra):
    data = {"transport": "openai", "base_url": "http://127.0.0.1:18645/v1", "model": MODEL,
            "status": status, "gpu_index": 2, "weights_dir": str(ROOT / "models"), **extra}
    temp = RUNTIME.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    temp.replace(RUNTIME)


def running(item):
    try:
        cmdline = Path(f"/proc/{item['pid']}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        return item["identity"] in cmdline
    except (FileNotFoundError, PermissionError, KeyError):
        return False


def registry():
    return json.loads(PROCESSES.read_text()) if PROCESSES.exists() else {}


def status():
    items = registry()
    data = json.loads(RUNTIME.read_text()) if RUNTIME.exists() else {"status": "not_prepared"}
    data["processes"] = {name: {"pid": item["pid"], "running": running(item)} for name, item in items.items()}
    if not items or not all(item["running"] for item in data["processes"].values()):
        data["status"] = "stopped_or_incomplete"
    print(json.dumps(data, ensure_ascii=False, indent=2))


def stop():
    items = registry()
    for item in reversed(list(items.values())):
        if running(item):
            os.kill(item["pid"], signal.SIGTERM)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and any(running(item) for item in items.values()):
        time.sleep(0.1)
    for item in items.values():
        if running(item):
            os.kill(item["pid"], signal.SIGKILL)
    PROCESSES.unlink(missing_ok=True)
    save_runtime("stopped", verification="Only this runtime's recorded and identity-checked processes were stopped.")
    print("Stopped the isolated MiniCPM runtime.")


def start():
    items = registry()
    if items and all(running(item) for item in items.values()):
        print("This isolated runtime is already running.")
        return
    if any(running(item) for item in items.values()):
        raise RuntimeError("Partial runtime is running; run manage.py stop before restarting.")
    binary = ROOT / "build/bin/llama-omni-server"
    python = ROOT / "venv/bin/python"
    if not binary.exists() or not python.exists():
        raise RuntimeError("Runtime is not built; see model_runtime/README.md.")
    manifest = json.loads((ROOT / "models-manifest.json").read_text())
    for entry in manifest["files"]:
        path = ROOT / "models" / entry["path"]
        if not path.exists() or path.stat().st_size != entry["size"]:
            raise RuntimeError("Official model bundle incomplete; run download_models.py first.")
    for port in (18645, 18646):
        with socket.socket() as probe:
            probe.settimeout(0.3)
            # TIME_WAIT from our previous WS sessions isn't a listening
            # service. The server's own bind still detects any startup race.
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError(f"Port {port} is already occupied; no existing service was touched.")
    gpu_free = int(subprocess.check_output(["nvidia-smi", "-i", "2", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
    if gpu_free < 13000:
        raise RuntimeError("GPU 2 has less than 13GB free; no existing GPU job was stopped.")
    gpu_uuid = subprocess.check_output(["nvidia-smi", "-i", "2", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": gpu_uuid, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "OMP_NUM_THREADS": "6", "CYBERFLY_OMNI_BACKEND": "ws://127.0.0.1:18646/backend"})
    env["LD_LIBRARY_PATH"] = str(ROOT / "build/bin") + ":/usr/local/cuda-12.9/lib64:/usr/lib/wsl/lib:" + env.get("LD_LIBRARY_PATH", "")
    commands = {
        "backend": [str(binary), "-m", str(ROOT / "models/MiniCPM-o-4_5-Q4_K_M.gguf"),
                    "--host", "127.0.0.1", "--port", "18646", "-c", "8192", "-ngl", "99", "-t", "6"],
        "gateway": [str(python), "-m", "uvicorn", "gateway:app", "--app-dir", str(HERE),
                    "--host", "127.0.0.1", "--port", "18645", "--log-level", "warning"],
    }
    saved = {}
    try:
        for name, command in commands.items():
            with (ROOT / f"{name}.log").open("ab") as log:
                proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                        stdout=log, stderr=log, start_new_session=True)
            identity = str(binary) if name == "backend" else "--app-dir " + str(HERE)
            saved[name] = {"pid": proc.pid, "identity": identity}
            PROCESSES.write_text(json.dumps(saved, indent=2))
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if not all(running(item) for item in saved.values()):
                raise RuntimeError("A runtime process exited; inspect its log.")
            try:
                for port in (18645, 18646):
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                        response.read()
                break
            except Exception:
                time.sleep(0.3)
        else:
            raise RuntimeError("The local runtime did not become healthy within 30 seconds.")
        save_runtime("running_unverified", verification="Processes healthy; run manage.py smoke to verify actual inference.")
        print("Running at http://127.0.0.1:18645/v1 on GPU 2. Model loads on first request.")
    except Exception:
        stop()
        raise


def smoke():
    sys.path.insert(0, str(PROJECT))
    from bridge import BridgeConfig, MiniCPMBridge
    bridge = MiniCPMBridge(BridgeConfig(transport="openai", base_url="http://127.0.0.1:18645/v1",
                                     model=MODEL, timeout_s=180))
    started = time.monotonic()
    result = bridge.plan("去世界坐标 [3,0] 毫米处，先设置目标，不要声称已经走到。",
                         state={"position_mm": [0, 0], "food_goal_mm": [3, 0], "policy_status": "not_executed"})
    elapsed = time.monotonic() - started
    if result.task != "navigate" or result.goal_mm != (3.0, 0.0):
        raise RuntimeError("Real model returned a valid but unexpected plan: " + json.dumps(result.to_dict(), ensure_ascii=False))
    evidence = {"verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "elapsed_s": round(elapsed, 3), "result": result.to_dict(),
                "source_revision": "64d092c60db4b4ee45768476bd752f03fdcc98ea",
                "weights_revision": json.loads((ROOT / "models-manifest.json").read_text())["revision"],
                "test_kind": "actual_local_model_text_inference", "training_performed": False}
    (HERE / "inference_verification.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2))
    save_runtime("verified", verification=evidence)
    print(json.dumps(evidence, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["start", "stop", "status", "smoke"])
    args = parser.parse_args()
    try:
        globals()[args.command]()
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
