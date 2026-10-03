"""Mock transport ownership checks, separate from actual audio evidence."""
import asyncio
import base64
import io
import json
import unittest
import wave
from unittest.mock import patch
import numpy as np
from model_runtime.duplex import audio_input, serve_duplex


def frame(sequence=0):
    return {"type": "input.append", "sequence": sequence,
            "input": {"audio": base64.b64encode(np.zeros(1600, dtype="<f4").tobytes()).decode()}}


class Client:
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.outgoing = asyncio.Queue()
        self.incoming.put_nowait({"type": "session.init", "payload": {}})
    async def receive_json(self): return await self.incoming.get()
    async def send_json(self, value): await self.outgoing.put(value)
    async def close(self): pass
    async def until(self, kind):
        while True:
            event = await asyncio.wait_for(self.outgoing.get(), 3)
            if event["type"] == kind: return event


class DuplexProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_owns_only_its_session_and_releases_after_cleanup(self):
        backends, closed = [], []
        class Backend:
            def __init__(self, name): self.name = name; self.incoming = asyncio.Queue()
            async def send(self, raw):
                event = json.loads(raw)
                if event["type"] == "session.init":
                    await self.incoming.put(json.dumps({"type": "session.created", "session_id": self.name, "mode": "full_duplex"}))
                else:
                    await self.incoming.put(json.dumps({"type": "response.output.delta", "kind": "listen", "response_id": self.name + "-1"}))
            async def recv(self): return await self.incoming.get()
            async def close(self): pass
        async def connect(*args, **kwargs):
            backend = Backend("owned-" + str(len(backends) + 1)); backends.append(backend); return backend
        async def close_session(session): closed.append(session)
        stream = io.BytesIO()
        with wave.open(stream, "wb") as wav:
            wav.setparams((1, 2, 16000, 1600, "NONE", "not compressed")); wav.writeframes(b"\0\0" * 1600)
        raw = stream.getvalue(); real_open = wave.open
        lock = asyncio.Lock(); first, second = Client(), Client()
        with patch("model_runtime.duplex.connect", connect), patch("model_runtime.duplex.wave.open", side_effect=lambda *args: real_open(io.BytesIO(raw), "rb")):
            one = asyncio.create_task(serve_duplex(first, lock, "ws://fixture/backend", close_session))
            await first.until("session.created")
            two = asyncio.create_task(serve_duplex(second, lock, "ws://fixture/backend", close_session))
            await second.until("session.queued")
            self.assertEqual(len(backends), 1)
            await first.incoming.put({"type": "input.cancel"})
            await first.until("response.cancelled")
            self.assertEqual(closed, ["owned-1"])
            await first.incoming.put(frame())
            created = await first.until("session.created")
            self.assertEqual(created["epoch"], 2)
            await first.until("input.processed")
            await first.incoming.put({"type": "session.close"})
            await one
            await second.until("session.created")
            self.assertEqual(closed, ["owned-1", "owned-2"])
            await second.incoming.put({"type": "session.close"})
            await two
        self.assertEqual(closed, ["owned-1", "owned-2", "owned-3"])
        self.assertFalse(lock.locked())

    def test_audio_framing_rejects_invalid_or_unbounded_data(self):
        self.assertEqual(audio_input(frame())[0], 0)
        for value in (float("nan"), float("inf"), 2.0):
            event = frame(); event["input"]["audio"] = base64.b64encode(np.full(1600, value, dtype="<f4").tobytes()).decode()
            with self.assertRaises(ValueError): audio_input(event)
        with self.assertRaises(ValueError): audio_input({**frame(), "session_id": "someone-else"})


if __name__ == "__main__": unittest.main()
