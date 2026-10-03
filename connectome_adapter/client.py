"""Dependency-light synchronous JSON-lines client for the Python 3.11 brain."""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time
import uuid

from .ports import validate_neural_drive, validate_general_stimulation

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = Path("/root/.cache/cyberfly/doom-env/bin/python")


class ConnectomeError(RuntimeError):
    pass


class ConnectomeStopped(ConnectomeError):
    """A cooperative stop prevents another step, while saving remains available."""


class ConnectomeClient:
    """One owned worker, serialized requests and real upstream neural state.

    `native_action` is the existing experimental BCI readout, not a universal
    action policy. Each scenario must explicitly map its turn/forward/attack.
    `reward_event` accepts only {'aversive': bool}; numerical rewards are never
    silently translated into neural punishment or dopamine stimulation.
    """

    def __init__(self, *, learning=False, timeout=120.0, checkpoint=None,
                 python=DEFAULT_PYTHON, log_dir=None):
        if learning is not False or not 0 < timeout <= 600:
            raise ValueError("The public connectome client is frozen; learning must be false")
        python = Path(python).absolute()
        if not python.is_file():
            raise FileNotFoundError(f"Brain worker interpreter is missing: {python}")
        self.timeout = float(timeout)
        self._lock = threading.RLock()
        self._responses = queue.Queue()
        self._counter = 0
        self._closed = False
        self._stop_requested = threading.Event()
        self.last_interrupted_request = None
        logs = Path(log_dir) if log_dir else ROOT / "artifacts/connectome"
        logs.mkdir(parents=True, exist_ok=True)
        tag = uuid.uuid4().hex[:12]
        self.stderr_path = logs / f"worker-{tag}.log"
        self.event_log_path = logs / f"stimuli-{tag}.jsonl"
        self._stderr = self.stderr_path.open("w")
        try:
            self.process = subprocess.Popen(
                [str(python), "-u", str(Path(__file__).with_name("worker.py"))],
                cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self._stderr, text=True, encoding="utf-8", bufsize=1,
                # Terminal Ctrl+C must reach the controller, not abort the
                # neural worker before its pending state can be checkpointed.
                start_new_session=True,
            )
        except BaseException:
            self._stderr.close()
            raise
        self._reader = threading.Thread(target=self._read_responses, daemon=True)
        self._reader.start()
        try:
            self.info = self._request("init", learning=learning,
                                      event_log=str(self.event_log_path.resolve()))
            if checkpoint is not None:
                self.restore(checkpoint)
        except BaseException:
            self.close(force=True)
            raise

    def _read_responses(self):
        try:
            for line in self.process.stdout:
                try:
                    self._responses.put(json.loads(line))
                except ValueError:
                    self._responses.put(ConnectomeError("Worker emitted malformed JSON"))
        finally:
            self._responses.put(ConnectomeError(f"Brain worker closed its output; inspect {self.stderr_path}"))

    def _request(self, operation, **data):
        with self._lock:
            if self._closed or self.process.poll() is not None:
                raise ConnectomeError("Brain worker is closed")
            self._counter += 1
            request_id = self._counter
            payload = json.dumps({"id": request_id, "op": operation, **data}, allow_nan=False) + "\n"
            interrupted = 0
            sent = False
            response = None
            previous_handler = None
            deadline = time.monotonic() + self.timeout

            def defer_interrupt(signum, frame):
                nonlocal interrupted
                interrupted += 1
                self._stop_requested.set()
                if interrupted > 1:
                    raise KeyboardInterrupt("Second interrupt forces the owned brain worker to stop")

            if threading.current_thread() is threading.main_thread():
                current_handler = signal.getsignal(signal.SIGINT)
                if current_handler is signal.default_int_handler:
                    previous_handler = current_handler
                    signal.signal(signal.SIGINT, defer_interrupt)
            try:
                self.process.stdin.write(payload)
                self.process.stdin.flush()
                sent = True
                response = self._responses.get(timeout=max(0, deadline - time.monotonic()))
            except KeyboardInterrupt:
                # Also handle custom handlers raising KeyboardInterrupt. Once
                # a request was sent, consume its one response before allowing
                # the caller's checkpoint request onto this serialized stream.
                self._stop_requested.set()
                if sent and interrupted < 2:
                    try:
                        response = self._responses.get(timeout=max(0, deadline - time.monotonic()))
                        self._validate_response(response, request_id)
                        self.last_interrupted_request = {"id": request_id, "operation": operation,
                                                         "completed": True, "result": response["result"]}
                    except BaseException:
                        self.close(force=True)
                        raise
                else:
                    self.close(force=True)
                raise
            except queue.Empty as exc:
                self.close(force=True)
                raise ConnectomeError(f"Brain request timed out; owned worker stopped. Log: {self.stderr_path}") from exc
            except (BrokenPipeError, OSError) as exc:
                raise ConnectomeError(f"Brain worker pipe failed: {self.stderr_path}") from exc
            finally:
                if previous_handler is not None:
                    signal.signal(signal.SIGINT, previous_handler)
            self._validate_response(response, request_id)
            if interrupted:
                self.last_interrupted_request = {"id": request_id, "operation": operation,
                                                 "completed": True, "result": response["result"]}
                raise KeyboardInterrupt("Brain interval finished safely; checkpoint before resuming")
            return response["result"]

    @staticmethod
    def _validate_response(response, request_id):
        if isinstance(response, Exception):
            raise response
        if not isinstance(response, dict) or response.get("id") != request_id:
            raise ConnectomeError("Brain protocol response ID mismatch")
        if response.get("ok") is not True:
            raise ConnectomeError(response.get("error", "Unknown brain worker error"))

    @property
    def stop_requested(self):
        return self._stop_requested.is_set()

    def request_stop(self):
        """Finish any in-flight request, then refuse new steps; save is allowed."""
        self._stop_requested.set()
        return {"stop_requested": True, "mode": "after_current_neural_interval"}

    def clear_stop(self):
        self._stop_requested.clear()
        return {"stop_requested": False}

    def wait_until_idle(self, timeout=None):
        timeout = self.timeout if timeout is None else float(timeout)
        acquired = self._lock.acquire(timeout=max(0.0, timeout))
        if acquired:
            self._lock.release()
        return acquired

    def step(self, rgb, *, duration_ms=28.6, reward_event=None, neural_drive=None,
             general_stimulation=None, include_group_stats=False, group_stats_level="macro",
             visual_input_enabled=True):
        """Advance RGB plus optional bounded currents to existing graph neurons.

        Legacy sensory ports use normalized 0..1 amplitudes. General stimulation
        uses signed model mV and explicit existing IDs or catalog groups.
        Omitted currents are zero for this interval.
        """
        if self.stop_requested:
            raise ConnectomeStopped("Brain stop was requested; save state and clear_stop() before another step")
        neural_drive = validate_neural_drive(neural_drive)
        general_stimulation = validate_general_stimulation(general_stimulation)
        if type(visual_input_enabled) is not bool:
            raise ValueError("visual_input_enabled must be bool")
        if reward_event is None:
            reward_event = {"aversive": False}
        if not isinstance(reward_event, dict) or set(reward_event) != {"aversive"} or type(reward_event["aversive"]) is not bool:
            raise ValueError("reward_event must be exactly {'aversive': bool}")
        if isinstance(rgb, (bytes, bytearray)):
            png = bytes(rgb)
        else:
            from PIL import Image
            image = rgb if isinstance(rgb, Image.Image) else Image.fromarray(rgb)
            if image.mode != "RGB":
                raise ValueError("Image must be RGB uint8; convert explicitly in the scenario")
            stream = io.BytesIO()
            image.save(stream, format="PNG")
            png = stream.getvalue()
        return self._request("step", rgb_png_base64=base64.b64encode(png).decode("ascii"),
                             duration_ms=duration_ms, reward_event=reward_event, neural_drive=neural_drive,
                             general_stimulation=general_stimulation, include_group_stats=include_group_stats,
                             group_stats_level=group_stats_level, visual_input_enabled=visual_input_enabled)

    def group_catalog(self):
        return self._request("group_catalog")

    def groups(self):
        return self.group_catalog()

    def write_neuron_currents(self, currents_mv, *, ids, directory=None):
        """Create an immutable independent-neuron stimulus descriptor, no step."""
        from .neuron_currents import array_sha256, write_neuron_currents
        import numpy as np
        if array_sha256(np.asarray(ids)) != self.info["full_brain_io"]["ids_sha256"]:
            raise ValueError("Neuron IDs/order do not match this brain")
        return write_neuron_currents(currents_mv, ids, directory=directory)

    def group_state(self, level="macro"):
        return self._request("group_state", level=level)

    def read_state(self, path=None, *, include_graph=False):
        """Write an immutable NPZ; return its path/hashes/shapes, never giant JSON arrays.

        Default state-UUID.npz files under artifacts/full_brain are temporary and
        may be deleted after reading. An explicit output directory is persistent.
        """
        return self._request("read_state", path=str(Path(path).resolve()) if path else None,
                             include_graph=include_graph)

    def snapshot(self, path=None, *, include_graph=False):
        return self.read_state(path, include_graph=include_graph)

    def read_arrays(self, *, include_graph=False):
        import numpy as np
        descriptor = self.read_state(include_graph=include_graph)
        path = Path(descriptor["path"])
        try:
            with np.load(path, allow_pickle=False) as archive:
                return {key: archive[key].copy() for key in archive.files}, descriptor
        finally:
            if descriptor["temporary"]:
                path.unlink(missing_ok=True)

    def neuron_details(self, neuron_ids):
        return self._request("neuron_details", neuron_ids=neuron_ids)

    def reset(self, *, preserve_weights=True):
        return self._request("reset", preserve_weights=preserve_weights)

    def set_learning(self, enabled):
        if enabled is not False:
            raise ValueError("The public connectome client is frozen; learning cannot be enabled")
        return self.describe()

    def save(self, path):
        return self._request("save", path=str(Path(path).resolve()))

    def restore(self, path):
        """Restore our checkpoint or a compatible upstream DOOMFLY generation."""
        return self._request("restore", path=str(Path(path).resolve()))

    def describe(self):
        return self._request("describe")

    def close(self, *, force=False):
        if self._closed:
            return
        self._closed = True
        try:
            if self.process.poll() is None:
                if not force:
                    try:
                        self.process.stdin.write(json.dumps({"id": -1, "op": "close"}) + "\n")
                        self.process.stdin.flush()
                        self.process.wait(timeout=5)
                    except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                        pass
                if self.process.poll() is None:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait(timeout=5)
        finally:
            if self.process.stdin:
                self.process.stdin.close()
            if self.process.stdout:
                self.process.stdout.close()
            self._stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
