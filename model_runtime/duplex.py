"""Native full-duplex ownership, bounded input flow and session cancellation."""
import asyncio
import base64
from collections import deque
import json
import os
from pathlib import Path
import wave

import numpy as np
from fastapi import WebSocketDisconnect
from websockets.asyncio.client import connect


def audio_input(event):
    if set(event) - {"type", "input", "sequence"}:
        raise ValueError("Unexpected input.append fields")
    data = event.get("input")
    if not isinstance(data, dict) or set(data) - {"audio", "force_listen"}:
        raise ValueError("Audio mode accepts audio and optional force_listen only")
    raw = base64.b64decode(data.get("audio", ""), validate=True)
    pcm = np.frombuffer(raw, dtype="<f4")
    if not 1600 <= len(pcm) <= 16000 or not np.isfinite(pcm).all() or np.abs(pcm).max() > 1.001:
        raise ValueError("Input must be 0.1-1.0s finite mono float32 PCM at 16kHz, amplitude [-1,1]")
    if type(data.get("force_listen", False)) is not bool:
        raise ValueError("force_listen must be boolean")
    sequence = event.get("sequence")
    if type(sequence) is not int or sequence < 0:
        raise ValueError("A nonnegative integer sequence is required")
    return sequence, {"type": "input.append", "input": {
        "audio": data["audio"], "force_listen": data.get("force_listen", False)}}


async def serve_duplex(client, worker_lock, backend_url, close_session):
    """One socket owns its native session. Cancellation resets only its epoch."""
    backend = None
    session_id = None
    client_read = None
    backend_read = None
    epoch = 0
    inflight = deque()
    processed = set()
    last_sequence = -1

    async def stop_native():
        nonlocal backend, session_id, backend_read
        if backend_read is not None:
            backend_read.cancel()
            await asyncio.gather(backend_read, return_exceptions=True)
            backend_read = None
        try:
            if session_id is not None:
                await close_session(session_id)
        finally:
            if backend is not None:
                await backend.close()
            backend, session_id = None, None
            inflight.clear()
            processed.clear()

    async def start_native():
        nonlocal backend, session_id, backend_read, epoch
        backend = await connect(backend_url, max_size=4 * 1024 * 1024, max_queue=4,
                                open_timeout=15, close_timeout=2)
        reference = Path(os.environ.get("CYBERFLY_MINICPM_ROOT", "/root/.cache/cyberfly/minicpm-runtime")) / "llama.cpp-omni/tools/omni/assets/default_ref_audio/default_ref_audio.wav"
        with wave.open(str(reference)) as wav:
            if wav.getframerate() != 16000 or wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                raise RuntimeError("Official default reference voice must be PCM16 mono 16kHz")
            voice = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype("<f4") / 32768
        await backend.send(json.dumps({"type": "session.init", "payload": {
            "mode": "full_duplex", "use_tts": True,
            "voice": {"ref_audio": base64.b64encode(voice.tobytes()).decode()},
            "config": {"temperature": 0.1, "force_listen_count": 0,
                       "max_new_speak_tokens_per_chunk": 20, "listen_prob_scale": 1.0}}}))
        created = json.loads(await backend.recv())
        if created.get("type") != "session.created":
            raise RuntimeError("Native full-duplex initialization failed")
        session_id = created["session_id"]
        epoch += 1
        await client.send_json(created | {"epoch": epoch, "brain_involved": False,
                                          "input_sample_rate": 16000, "output_sample_rate": 24000,
                                          "max_inflight": 3})
        backend_read = asyncio.create_task(backend.recv())

    try:
        if worker_lock.locked():
            await client.send_json({"type": "session.queued", "position": 1})
        # Long voice sessions intentionally own the single model worker. HTTP
        # planners queue until this session closes; ownership is never stolen.
        async with asyncio.timeout(300):
            async with worker_lock:
                try:
                    await client.send_json({"type": "session.queue_done"})
                    init = await asyncio.wait_for(client.receive_json(), 20)
                    if init.get("type") != "session.init" or set(init) - {"type", "payload"}:
                        raise ValueError("Expected session.init")
                    payload = init.get("payload", {})
                    if not isinstance(payload, dict) or payload:
                        raise ValueError("Native voice uses payload={} and does not execute brain actions")
                    await start_native()
                    client_read = asyncio.create_task(client.receive_json())
                    while True:
                        pending = {client_read}
                        if backend_read is not None:
                            pending.add(backend_read)
                        done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                        if client_read in done:
                            event = client_read.result()
                            kind = event.get("type")
                            if kind in ("input.cancel", "session.close"):
                                if set(event) != {"type"}:
                                    raise ValueError("Cancellation cannot name another session")
                                await stop_native()
                                await client.send_json({"type": "response.cancelled", "epoch": epoch,
                                                        "context_reset": True, "pending_input_discarded": True})
                                if kind == "session.close":
                                    await client.send_json({"type": "session.closed", "reason": "client_closed", "epoch": epoch})
                                    return
                            elif kind == "input.append":
                                sequence, forwarded = audio_input(event)
                                if sequence <= last_sequence:
                                    raise ValueError("sequence must strictly increase across cancellations")
                                if len(inflight) >= 3:
                                    raise ValueError("Input backpressure exceeded: await input.processed before sending more")
                                last_sequence = sequence
                                if backend is None:
                                    await start_native()
                                inflight.append(sequence)
                                await backend.send(json.dumps(forwarded))
                                await client.send_json({"type": "input.accepted", "sequence": sequence, "epoch": epoch})
                            else:
                                raise ValueError("Expected input.append, input.cancel or session.close")
                            client_read = asyncio.create_task(client.receive_json())
                        # A cancellation above invalidates the old read task;
                        # stale audio from a previous epoch is never forwarded.
                        if backend_read is not None and backend_read in done:
                            event = json.loads(backend_read.result())
                            if event.get("type") in ("session.closed", "error"):
                                raise RuntimeError("Native full-duplex session ended unexpectedly")
                            await client.send_json(event | {"epoch": epoch})
                            boundary = event.get("type") == "response.done" or (event.get("type") == "response.output.delta" and event.get("kind") == "listen")
                            response_id = event.get("response_id")
                            if boundary and response_id not in processed:
                                processed.add(response_id)
                                if inflight:
                                    sequence = inflight.popleft()
                                    await client.send_json({"type": "input.processed", "sequence": sequence, "epoch": epoch})
                            backend_read = asyncio.create_task(backend.recv())
                finally:
                    # The worker lock remains owned while native inference and
                    # transports stop, including disconnect/error paths.
                    if client_read is not None:
                        client_read.cancel()
                        await asyncio.gather(client_read, return_exceptions=True)
                    await stop_native()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await client.send_json({"type": "error", "error": {"message": str(exc) or "Native duplex failed"},
                                    "epoch": epoch, "fallback_used": False})
        except Exception:
            pass
    finally:
        try:
            await client.close()
        except Exception:
            pass
