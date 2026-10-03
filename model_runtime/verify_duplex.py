"""Actual native audio streaming, cancellation and post-cancel input probe."""
import base64
import io
import json
from pathlib import Path
import time
import wave

import numpy as np
from websockets.sync.client import connect


def main():
    project = Path(__file__).resolve().parents[1]
    output = project / "artifacts/lab-integration"
    source = project / "model_runtime/audio_verification/duplex_question.wav"
    with wave.open(str(source)) as wav:
        rate = wav.getframerate()
        original = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
    speech = np.interp(np.arange(round(len(original) * 16000 / rate)) * rate / 16000, np.arange(len(original)), original).astype("<f4")
    speech = np.concatenate([speech, np.zeros(16000 * 24, dtype="<f4")])
    speech = np.pad(speech, (0, (-len(speech)) % 16000))
    frames = [base64.b64encode(frame.tobytes()).decode() for frame in np.split(speech, len(speech) // 16000)]
    evidence = {"status": "running", "source_wav": str(source), "input_format": "16000Hz mono float32 PCM",
                "native_mode": "full_duplex", "events": [], "epochs": {}}
    sequence, offset, phase, credits = 0, 0, 1, 0
    audio = {}
    started = time.monotonic()
    cancelled = False
    close_sent = False
    try:
        with connect("ws://127.0.0.1:18645/v1/realtime?mode=audio", max_size=4 * 1024 * 1024, open_timeout=10, close_timeout=5) as ws:
            def append_next():
                nonlocal sequence, offset, credits
                if offset >= len(frames):
                    raise RuntimeError("No model speech after complete audio plus silence")
                ws.send(json.dumps({"type": "input.append", "sequence": sequence, "input": {"audio": frames[offset]}}))
                sequence += 1
                offset += 1
                credits += 1

            while time.monotonic() - started < 180:
                event = json.loads(ws.recv(timeout=60))
                record = {key: value for key, value in event.items() if key != "audio"}
                record["elapsed_s"] = round(time.monotonic() - started, 3)
                evidence["events"].append(record)
                kind = event.get("type")
                if kind == "session.queue_done":
                    ws.send(json.dumps({"type": "session.init", "payload": {}}))
                elif kind == "session.created":
                    print("CREATED", event["epoch"], event["mode"], flush=True)
                    evidence["epochs"][str(event["epoch"])] = {"session_id": event["session_id"], "native_mode": event["mode"]}
                    if phase == 1:
                        append_next()
                elif kind == "input.processed" and not cancelled and not close_sent:
                    credits -= 1
                    append_next()
                elif kind == "response.output.delta" and event.get("kind") == "audio":
                    samples = np.frombuffer(base64.b64decode(event["audio"], validate=True), dtype="<f4")
                    if not len(samples) or not np.isfinite(samples).all():
                        raise RuntimeError("Invalid actual streamed model PCM")
                    epoch = event["epoch"]
                    audio.setdefault(epoch, []).append(samples.copy())
                    record["audio_samples"] = len(samples)
                    print("AUDIO", epoch, len(samples), flush=True)
                    if phase == 1 and not cancelled:
                        ws.send(json.dumps({"type": "input.cancel"}))
                        cancelled = True
                    elif phase == 2 and not close_sent:
                        ws.send(json.dumps({"type": "session.close"}))
                        close_sent = True
                elif kind == "response.cancelled" and phase == 1:
                    print("CANCELLED", event, flush=True)
                    phase, offset, credits, cancelled = 2, 0, 0, False
                    append_next()
                elif kind == "session.closed":
                    break
                elif kind == "error":
                    raise RuntimeError(event)
            if not close_sent or set(audio) != {1, 2}:
                raise RuntimeError("Did not prove audio output before and after cancellation")
        for epoch, parts in audio.items():
            pcm = np.concatenate(parts)
            path = output / f"minicpm-duplex-epoch-{epoch}.wav"
            with wave.open(str(path), "wb") as wav:
                wav.setparams((1, 2, 24000, len(pcm), "NONE", "not compressed"))
                wav.writeframes((np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())
            evidence["epochs"][str(epoch)].update({"wav": str(path), "audio_seconds": len(pcm) / 24000,
                                                    "rms": float(np.sqrt(np.mean(pcm**2)))})
        evidence["status"] = "verified"
        evidence["limitation"] = "Native full-duplex frames and actual audio output verified. Cancel resets the native conversational context; this is not seamless in-context barge-in or task-mastery evidence. Input is replayed real synthesized speech, not a browser microphone benchmark."
    except Exception as exc:
        evidence["status"], evidence["error"] = "failed", repr(exc)
        raise
    finally:
        evidence["elapsed_s"] = round(time.monotonic() - started, 3)
        (output / "minicpm-duplex-verification.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
        print(evidence["status"], evidence.get("error", ""), flush=True)


if __name__ == "__main__":
    main()
