"""Controlled real-model visual/audio A/B with a fixed brain vector and coupler.

Run only after the model owner confirms its supervised training has finished.
This script does not train, restart services, or step the brain. It uses five
encode calls: real view, same-size solid view, native speech A, native speech B,
and a repeated real-view baseline. Speech is linearly resampled and cropped to
the same three-second interval; no synthetic speech is substituted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
import uuid
import wave

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
IDENTITY_FIELDS = ("model_revision", "weights_manifest_sha256", "codec_sha256",
                   "coupler_sha256", "coupler_generation", "coupler_training_steps")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def call(endpoint, operation, body=None):
    request = urllib.request.Request(endpoint.rstrip("/") + "/" + operation,
        data=None if body is None else json.dumps(body, ensure_ascii=False, allow_nan=False).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(error.read(8000).decode("utf-8", errors="replace")) from None
    if not isinstance(result, dict) or result.get("error"):
        raise RuntimeError("Model response was not a successful object")
    return result


def identity(result):
    values = {key: result.get(key) for key in IDENTITY_FIELDS}
    if any(value is None for value in values.values()):
        raise RuntimeError("Model omitted a required weight, codec or checkpoint identity")
    return values


def native_audio(source, destination):
    raw = source.read_bytes()
    with wave.open(str(source), "rb") as wav:
        rate, channels, width, frames = wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes()
        if channels != 1 or width != 2 or rate <= 0 or frames < 3 * rate:
            raise ValueError("Native audio fixture must contain at least three seconds of mono PCM16")
        samples = np.frombuffer(wav.readframes(frames), dtype="<i2").astype(np.float32)
    # Identical duration and sample count remove audio-token-length differences.
    positions = np.arange(3 * 16000, dtype=np.float64) * rate / 16000
    converted = np.interp(positions, np.arange(len(samples)), samples).round().clip(-32768, 32767).astype("<i2")
    if np.sqrt(np.mean((converted.astype(np.float64) / 32768) ** 2)) < 1e-4:
        raise RuntimeError("Native speech fixture is effectively silent")
    with wave.open(str(destination), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(converted.tobytes())
    return {"source_path": str(source), "source_sha256": digest(raw), "source_sample_rate": rate,
            "source_frames": frames, "output_path": str(destination),
            "output_sha256": digest(destination.read_bytes()), "output_sample_rate": 16000,
            "output_frames": len(converted), "seconds": 3.0,
            "operation": "First three seconds of actual native MiniCPM speech; linear resampling only"}


def compare(left, right, prefix):
    delta = left[prefix].astype(np.float64) - right[prefix].astype(np.float64)
    return {"l2": float(np.linalg.norm(delta)), "max_abs": float(np.max(np.abs(delta))),
            "changed_elements": int(np.count_nonzero(delta)), "total_elements": int(delta.size)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18647")
    parser.add_argument("--request", type=Path, default=ROOT / "artifacts/lab-integration/shared-hidden-01/request.json")
    parser.add_argument("--image", type=Path, default=ROOT / "artifacts/shared_io_codec_checks/5ad56f7328fc/fly_neural_link_shared_input.png")
    parser.add_argument("--audio-a", type=Path, default=ROOT / "model_runtime/audio_verification/native_tts.wav")
    parser.add_argument("--audio-b", type=Path, default=ROOT / "model_runtime/audio_verification/duplex_question.wav")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve() if args.output else ROOT / "artifacts/shared_io_modal_ab" / uuid.uuid4().hex[:12]
    if not output.is_relative_to(ROOT / "artifacts"):
        raise ValueError("Use a new artifacts directory for auditable local model input")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    evidence = {"status": "running", "output": str(output), "started_at": time.time(),
                "scope": "Fixed-identity original multimodal MiniCPM input sensitivity; not understanding, task skill or live neural learning"}
    try:
        original_request = json.loads(args.request.read_text())
        features = np.asarray(original_request["brain_features"], dtype=np.float32)
        if features.shape != (240,) or not np.isfinite(features).all():
            raise ValueError("Original shared verification request lacks its finite 240-feature vector")
        text = "请结合当前画面与可能存在的声音，简短描述你接收到的内容。"
        with Image.open(args.image) as source:
            actual = source.convert("RGB")
            actual.save(output / "actual_view.png")
            Image.new("RGB", actual.size, color=(5, 13, 31)).save(output / "solid_view.png")
        audio_a = native_audio(args.audio_a.resolve(), output / "native_a_16k_3s.wav")
        audio_b = native_audio(args.audio_b.resolve(), output / "native_b_16k_3s.wav")
        if audio_a["output_sha256"] == audio_b["output_sha256"]:
            raise RuntimeError("Two audio fixtures contain identical supplied waveforms")
        expected = identity(call(args.endpoint, "health"))
        evidence.update(identity=expected, text=text, text_sha256=digest(text.encode()),
            brain_feature_source_request=str(args.request.resolve()),
            brain_features_sha256=digest(features.astype("<f4").tobytes()),
            image_source=str(args.image.resolve()), image_source_sha256=digest(args.image.read_bytes()),
            image_shape=[actual.height, actual.width, 3], audio_sources=[audio_a, audio_b],
            fixed_controls=["text", "brain_features", *IDENTITY_FIELDS],
            audio_control="Both actual native speech recordings cropped to the same first three seconds and 48000 samples")
        np.save(output / "fixed_brain_features.npy", features, allow_pickle=False)
        write_json(output / "evidence.json", evidence)
        conditions = [
            ("actual_view", "actual_view.png", None),
            ("solid_view", "solid_view.png", None),
            ("native_audio_a", "actual_view.png", "native_a_16k_3s.wav"),
            ("native_audio_b", "actual_view.png", "native_b_16k_3s.wav"),
            ("actual_view_repeat", "actual_view.png", None),
        ]
        results = {}
        for name, image_name, audio_name in conditions:
            if identity(call(args.endpoint, "health")) != expected:
                raise RuntimeError("Model/coupler changed during controlled experiment")
            request = {"event_id": "modal-ab-" + uuid.uuid4().hex, "text": text,
                       "brain_features": features.tolist(), "image_path": str(output / image_name)}
            if audio_name:
                request["audio_path"] = str(output / audio_name)
            write_json(output / (name + "-request.json"), request)
            response = call(args.endpoint, "encode", request)
            write_json(output / (name + "-response.json"), response)
            if identity(response) != expected or response.get("event_id") != request["event_id"]:
                raise RuntimeError("Response event or model/coupler identity changed")
            hidden = np.asarray(response["hidden"], dtype=np.float32)
            currents = np.asarray(response["currents_mv"], dtype=np.float32)
            if hidden.shape != (4096,) or currents.shape != (47,) or not np.isfinite(hidden).all() or not np.isfinite(currents).all():
                raise RuntimeError("Actual hidden/current arrays are malformed")
            if response["image_token_count"] <= 0 or (audio_name and response["audio_token_count"] <= 0):
                raise RuntimeError("A requested raw modality was not actually encoded")
            modalities = ["text", "image"] + (["audio"] if audio_name else [])
            if response.get("modalities_encoded") != modalities:
                raise RuntimeError("Model did not report the exact supplied raw modalities")
            results[name] = {"hidden": hidden, "currents": currents, "response": response}
            np.savez_compressed(output / (name + ".npz"), hidden=hidden, currents_mv=currents)
            print(json.dumps({"stage": name, "hidden_sha256": digest(hidden.astype("<f4").tobytes()),
                              "image_tokens": response["image_token_count"], "audio_tokens": response["audio_token_count"]}), flush=True)
        if identity(call(args.endpoint, "health")) != expected:
            raise RuntimeError("Model/coupler changed at the end of controlled experiment")
        visual = compare(results["actual_view"], results["solid_view"], "hidden")
        audio = compare(results["native_audio_a"], results["native_audio_b"], "hidden")
        repeat = compare(results["actual_view"], results["actual_view_repeat"], "hidden")
        same_soft = {item["response"]["brain_soft_sha256"] for item in results.values()}
        if len(same_soft) != 1:
            raise RuntimeError("Brain-derived soft tokens changed while their input/checkpoint was held fixed")
        if results["actual_view"]["response"]["input_token_count"] != results["solid_view"]["response"]["input_token_count"]:
            raise RuntimeError("Visual conditions changed token count")
        if results["native_audio_a"]["response"]["input_token_count"] != results["native_audio_b"]["response"]["input_token_count"]:
            raise RuntimeError("Audio conditions changed token count")
        minimum_effect = max(1e-6, 10 * repeat["l2"])
        if visual["l2"] <= minimum_effect or audio["l2"] <= minimum_effect:
            raise RuntimeError("Modality effect did not exceed the observed repeat baseline")
        evidence.update(status="passed", visual_hidden_difference=visual, audio_hidden_difference=audio,
            repeat_hidden_difference=repeat, minimum_effect_l2=minimum_effect,
            visual_current_difference=compare(results["actual_view"], results["solid_view"], "currents"),
            audio_current_difference=compare(results["native_audio_a"], results["native_audio_b"], "currents"),
            fixed_brain_soft_sha256=next(iter(same_soft)), actual_encode_calls=len(conditions),
            equal_token_counts_within_each_comparison=True, brain_step_executed=False,
            training_performed=False, service_restarted=False,
            limitations="Original vision and audio encoders measurably affect real language hidden states under controlled input. This does not establish perception accuracy, subjective thought decoding, brain learning or task competence.")
    except BaseException as error:
        evidence.update(status="failed", error=str(error))
        raise
    finally:
        evidence["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "evidence.json", evidence)
    print(json.dumps({"status": evidence["status"], "evidence": str(output / "evidence.json"),
                      "visual_l2": visual["l2"], "audio_l2": audio["l2"], "repeat_l2": repeat["l2"]}), flush=True)


if __name__ == "__main__":
    main()
