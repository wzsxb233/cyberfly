"""Fail-closed contract for a future *native* MiniCPM duplex latent hook.

The pinned upstream currently drops custom append fields. These helpers are
CPU-ready; they do not register that missing serving/worker extension.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path

EXTENSION = "cyberfly_duplex_latent"
SCHEMA = 1
SAMPLES_PER_UNIT = 16000
SOFT_TOKEN_ID = 151662
IDENTITY_KEYS = (
    "model_revision", "weights_manifest_sha256", "vllm_omni_commit",
    "connection_mode", "io_scope", "codec_sha256", "ids_sha256",
    "coupler_sha256", "coupler_generation", "read_ids_sha256", "write_ids_sha256",
)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def require_native_capabilities(session: dict, *, require_latent: bool = True) -> dict:
    """Reject a chat-stream fallback, even if its output contains audio."""
    caps = session.get("capabilities")
    if not isinstance(caps, dict):
        raise ValueError("Native duplex capabilities were not advertised")
    if caps.get("implementation_level") != "model_native_duplex":
        raise ValueError("A serving chat adapter is not native duplex")
    for key in ("supports_model_native_turn_policy", "supports_input_append",
                "supports_model_internal_state", "supports_core_resumable_request",
                "supports_independent_io_streams", "requires_model_runner_kv"):
        if caps.get(key) is not True:
            raise ValueError("Missing actual native capability: " + key)
    if caps.get("stage_handoff_transport") != "scheduler_data_plane":
        raise ValueError("Unexpected native stage transport")
    if require_latent:
        extension = caps.get(EXTENSION)
        expected = {"schema": SCHEMA, "placement": "before_sensory",
                    "append_only": True, "facts_in_model_input": False,
                    "generation_hidden_export": True, "unit_samples": SAMPLES_PER_UNIT}
        if not isinstance(extension, dict) or any(extension.get(k) != v for k, v in expected.items()):
            raise ValueError("Native numerical duplex extension is not installed/advertised")
    return copy.deepcopy(caps)


def condition_identity(condition: dict) -> dict:
    result = {}
    for key in IDENTITY_KEYS:
        value = condition.get(key)
        if value is None:
            raise ValueError("Missing pinned numerical identity: " + key)
        result[key] = value
    for name in ("body_input", "body_feedback"):
        body = condition.get(name)
        if isinstance(body, dict) and body:
            result[name] = {key: body[key] for key in
                            ("layout_sha256", "body_embedding_sha256")}
    # Adapter changes must also fence an epoch when support is enabled later.
    result["language_adapter"] = copy.deepcopy(condition.get("language_adapter", {"enabled": False}))
    return result


@dataclass
class UnitLedger:
    """Client input ordinal is zero-based; engine seq is never guessed from it."""
    session_id: str
    epoch: int
    identity: dict | None = None
    next_unit: int = 0
    last_brain_ms: float = -1.0
    recent_events: list[str] = field(default_factory=list)

    def prepare(self, *, event_id: str, pcm16: bytes, condition: dict,
                image_path: str | None = None) -> tuple[dict, dict]:
        from vllm_integration.latent_contract import load_condition

        if not isinstance(event_id, str) or not event_id or len(event_id) > 128:
            raise ValueError("Invalid event identity")
        if event_id in self.recent_events:
            raise ValueError("An already submitted input event cannot be replayed")
        if not isinstance(pcm16, bytes) or len(pcm16) != SAMPLES_PER_UNIT * 2:
            raise ValueError("Send exactly 1 s of real 16 kHz mono PCM16; chunk/pad tails explicitly outside this client")
        if condition.get("event_id") != event_id or condition.get("facts_in_model_input") is not False:
            raise ValueError("Condition must belong to this exact event and contain no facts text")
        values = load_condition(condition)
        identity = condition_identity(condition)
        if self.identity is not None and identity != self.identity:
            raise ValueError("Model/converter identity changed during a native epoch; open a new session")
        brain_ms = condition.get("brain_source", {}).get("simulation_ms")
        if not isinstance(brain_ms, (int, float)) or isinstance(brain_ms, bool) or not math.isfinite(brain_ms):
            raise ValueError("Missing actual brain observation time")
        if brain_ms < self.last_brain_ms:
            raise ValueError("Brain time moved backwards; reset requires a new session/epoch")
        image_bytes = None
        if image_path is not None:
            image_bytes = Path(image_path).read_bytes()
            if len(image_bytes) > 3 * 1024 * 1024 or not (
                    image_bytes.startswith(b"\x89PNG\r\n\x1a\n") or image_bytes.startswith(b"\xff\xd8\xff")):
                raise ValueError("Expected bounded original PNG/JPEG bytes")
        extension = {
            "schema": SCHEMA, "session_id": self.session_id, "epoch": self.epoch,
            "unit_index": self.next_unit, "event_id": event_id,
            "condition": copy.deepcopy(condition), "soft_token_count": len(values),
            "placement": "before_sensory", "facts_in_model_input": False,
            "audio_pcm16_sha256": _sha(pcm16),
            "image_file_sha256": _sha(image_bytes) if image_bytes is not None else None,
        }
        wire = {
            "type": "input_audio_buffer.append", "event_id": event_id,
            "audio": base64.b64encode(pcm16).decode("ascii"),
            "format": "pcm16", "sample_rate_hz": 16000, "duration_ms": 1000,
            "audio_end_ms": (self.next_unit + 1) * 1000, EXTENSION: extension,
        }
        if image_bytes is not None:
            wire["video_frames"] = [base64.b64encode(image_bytes).decode("ascii")]
        audit = {key: copy.deepcopy(value) for key, value in extension.items() if key != "condition"}
        audit.update(identity=identity, brain_simulation_ms=brain_ms,
                     condition_sha256=condition["sha256"], audio_samples=SAMPLES_PER_UNIT,
                     runtime_seq=None, status="prepared_not_applied")
        return wire, audit

    def submitted(self, audit: dict) -> None:
        if audit["unit_index"] != self.next_unit or audit["session_id"] != self.session_id or audit["epoch"] != self.epoch:
            raise ValueError("Out-of-order input submission")
        self.identity = copy.deepcopy(audit["identity"])
        self.last_brain_ms = float(audit["brain_simulation_ms"])
        self.recent_events.append(audit["event_id"])
        self.recent_events = self.recent_events[-32:]
        self.next_unit += 1


def reserve_append_slots(prompt: dict, *, soft_token_count: int) -> dict:
    """Unregistered runtime-plan hook; extends only this new append's budget.

    Call after the official plan builder and before it enters the scheduler.
    It cannot fix an already scheduled prompt or mutate existing KV positions.
    """
    if type(soft_token_count) is not int or not 4 <= soft_token_count <= 6:
        raise ValueError("Expected four brain plus up to two body soft tokens")
    result = copy.deepcopy(prompt)
    duplex = result["model_intermediate_buffer"]["duplex"]
    if duplex.get("data_plane") is not True or EXTENSION in duplex:
        raise ValueError("Expected a fresh native append plan")
    ids = result["prompt_token_ids"]
    if duplex.get("scheduler_token_budget") != len(ids):
        raise ValueError("Original scheduler budget disagrees with append tokens")
    result["prompt_token_ids"] = [duplex["scheduler_token_id"]] * soft_token_count + ids
    duplex["scheduler_token_budget"] += soft_token_count
    duplex[EXTENSION] = {"schema": SCHEMA, "reserved_tokens": soft_token_count}
    return result


def fuse_current_unit(native_embeddings, native_token_ids, soft_tokens, *, unit_token_id: int):
    """Unregistered helper-result hook, BEFORE official padding/offset slicing.

    Original current-unit <unit>, VPM, APM and terminal sensory embedding stay
    in order. Original generated control tokens are never masked/overwritten.
    Input arrays here are CPU NumPy tensors; the GPU hook must use torch.cat
    with the same positions/dtype, and retain the original native TTS object.
    """
    import numpy as np

    original = np.asarray(native_embeddings)
    soft = np.asarray(soft_tokens)
    ids = list(native_token_ids)
    if original.ndim != 2 or original.shape != (len(ids), 4096) or not np.isfinite(original).all():
        raise ValueError("Invalid native unit embeddings")
    if soft.dtype != np.float32 or soft.shape[1:] != (4096,) or not 4 <= len(soft) <= 6 or not np.isfinite(soft).all():
        raise ValueError("Invalid numeric soft tokens")
    positions = [i for i, value in enumerate(ids) if value == unit_token_id]
    if len(positions) != 1:
        raise ValueError("First version requires exactly one emitted native unit; buffering/multiple-unit flush needs explicit mapping")
    insert_at = positions[0] + 1
    fused = np.concatenate((original[:insert_at], soft.astype(original.dtype), original[insert_at:]))
    fused_ids = ids[:insert_at] + [SOFT_TOKEN_ID] * len(soft) + ids[insert_at:]
    return fused, fused_ids, {"soft_start_in_actual_append": insert_at, "soft_count": len(soft),
                             "native_last_embedding_sha256": _sha(original[-1].tobytes()),
                             "no_past_kv_edit": True}


def scheduled_slice(full_embeddings, token_ids, *, prompt_len: int, token_offset: int, span: int):
    """Strict audit after native left-padding, covering chunked prefill only."""
    for value in (prompt_len, token_offset, span):
        if type(value) is not int or value < 0:
            raise ValueError("Invalid scheduler coordinates")
    if token_offset >= prompt_len:
        raise ValueError("Generation tokens must use original embedding lookup, never the prompt slice")
    if len(full_embeddings) != prompt_len or len(token_ids) != prompt_len or token_offset + span > prompt_len:
        raise ValueError("Scheduler budget mismatch; refusing truncation/pad-filled fake input")
    return full_embeddings[token_offset:token_offset + span], token_ids[token_offset:token_offset + span]
