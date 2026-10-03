"""Native official WS client wrapper; latent extension remains deployment gated.

No GPU/model import at module import. This file does not install server hooks.
The official client owns native audio chunks, turn decisions, ACKs and cancel.
"""
from __future__ import annotations

import asyncio
from urllib.parse import urlsplit

from vllm_integration.duplex_contract import UnitLedger, require_native_capabilities


class NativeLatentDuplexClient:
    def __init__(self, url: str, *, model: str, ref_audio: str,
                 session_id: str, initial_user_text: str = "", connect=None):
        parts = urlsplit(url)
        if parts.scheme != "ws" or parts.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("The artifact-based numerical duplex bridge is loopback-only")
        if not ref_audio:
            raise ValueError("Official MiniCPM native speech requires real reference audio")
        self.url = url
        self.model = model
        self.ref_audio = ref_audio
        self.session_id = session_id
        self.initial_user_text = initial_user_text
        self.connect = connect
        self.client = None
        self.ledger = None
        self._append_lock = asyncio.Lock()
        self.failed = False

    async def __aenter__(self):
        from vllm_omni.clients.duplex import DuplexClient
        from vllm_omni.clients.minicpmo_4_5 import create_duplex_session_config

        config = create_duplex_session_config(
            ref_audio=self.ref_audio, temperature=0.0,
            extra_body={"duplex_initial_user_text": self.initial_user_text})
        # Fail closed on a transport drop: automatic replay needs server-side
        # numeric operation acknowledgments before it can safely be enabled.
        self.client = DuplexClient(self.url, model=self.model, config=config,
                                   session_id=self.session_id, reconnect=None,
                                   connect=self.connect)
        await self.client.__aenter__()
        try:
            require_native_capabilities(self.client.session_info)
            epoch = self.client.session_info.get("epoch")
            if type(epoch) is not int or epoch < 0:
                raise ValueError("Native session did not advertise an explicit epoch")
            self.ledger = UnitLedger(self.client.session_id, epoch)
        except BaseException as exc:
            await self.client.__aexit__(type(exc), exc, exc.__traceback__)
            raise
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        if self.client is not None:
            await self.client.__aexit__(exc_type, exc, traceback)

    async def append_unit(self, *, event_id: str, pcm16: bytes, condition: dict,
                          image_path: str | None = None) -> dict:
        if self.client is None or self.ledger is None or self.failed:
            raise RuntimeError("No usable native numerical session")
        async with self._append_lock:
            wire, audit = self.ledger.prepare(event_id=event_id, pcm16=pcm16,
                                             condition=condition, image_path=image_path)
            try:
                await self.client.send(wire)
            except BaseException:
                # Send failure is ambiguous; never automatically repeat a
                # potentially already-consumed real brain observation.
                self.failed = True
                raise
            self.ledger.submitted(audit)
            return {**audit, "status": "sent_not_yet_worker_verified"}

    def events(self):
        """Raw native events; empty listen beats remain empty."""
        return self.client.events()

    def responses(self):
        """Original incremental audio handles; no re-TTS/string round trip."""
        return self.client.responses()

    async def commit(self):
        return await self.client.commit()

    async def ack_playback(self, played_ms: int):
        return await self.client.ack_playback(played_ms)

    async def cancel_response(self, response_id: str | None = None):
        # Cancellation may advance the runtime epoch. Require a fresh client
        # before sending more numeric units until the extension reports its
        # new epoch explicitly; never bind an old condition to a new epoch.
        self.failed = True
        return await self.client.cancel_response(response_id)
