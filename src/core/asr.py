"""Deepgram streaming ASR, one socket per participant.

Core, not a provider: the agent hears the meeting through Recall, never
through the avatar, so this is identical under every video vendor.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode, quote

import websockets
from websockets.exceptions import ConnectionClosed

from src.agent_policy import FinalUtteranceBuffer

logger = logging.getLogger("SpikedMeetingAgent")

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API") or os.getenv("DEEPGRAM_API_KEY", "")
AGENT_ENDPOINTING_MS = int(os.getenv("AGENT_ENDPOINTING_MS", "300"))

# Below 1000 Deepgram rejects the connection with a 400, which looks exactly
# like an agent that hears nobody. Clamped so tuning cannot deafen the bot.
DEEPGRAM_MIN_UTTERANCE_END_MS = 1000
_requested_utterance_end_ms = int(os.getenv("AGENT_UTTERANCE_END_MS", "1000"))
AGENT_UTTERANCE_END_MS = max(_requested_utterance_end_ms, DEEPGRAM_MIN_UTTERANCE_END_MS)
if _requested_utterance_end_ms < DEEPGRAM_MIN_UTTERANCE_END_MS:
    logger.warning(
        "[Deepgram] AGENT_UTTERANCE_END_MS=%d is below the API minimum; using %d",
        _requested_utterance_end_ms,
        AGENT_UTTERANCE_END_MS,
    )


class ParticipantTranscriber:
    def __init__(
        self,
        participant_id: str,
        participant_name: str,
        keywords: List[str],
        on_utterance: Any,
    ):
        self.participant_id = participant_id
        self.participant_name = participant_name
        self.keywords = keywords
        self.on_utterance = on_utterance
        self.ws: Any = None
        self.receiver_task: Optional[asyncio.Task] = None
        self.buffer = FinalUtteranceBuffer()
        self.start_lock = asyncio.Lock()
        # Proxy for "when this person stopped talking"; no timestamp exists.
        self._last_words_at: Optional[float] = None

    async def ensure_started(self) -> None:
        if self.ws:
            return
        async with self.start_lock:
            if self.ws:
                return
            params = {
                "model": "nova-3",
                "encoding": "linear16",
                "sample_rate": "16000",
                "channels": "1",
                "smart_format": "true",
                "interim_results": "true",
                "endpointing": str(AGENT_ENDPOINTING_MS),
                "utterance_end_ms": str(AGENT_UTTERANCE_END_MS),
                "vad_events": "true",
                "punctuate": "true",
            }
            url = f"wss://api.deepgram.com/v1/listen?{urlencode(params)}"
            keyterms: List[str] = []
            keyterm_tokens = 0
            for item in self.keywords[:100]:
                if not item or not item.strip():
                    continue
                estimated_tokens = max(1, len(item.split()))
                if keyterm_tokens + estimated_tokens > 450:
                    break
                keyterms.append(item.strip())
                keyterm_tokens += estimated_tokens
            if keyterms:
                url += "&" + "&".join(f"keyterm={quote(item.strip())}" for item in keyterms)
            headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}
            try:
                try:
                    self.ws = await websockets.connect(url, additional_headers=headers)
                except TypeError:
                    self.ws = await websockets.connect(url, extra_headers=headers)
            except websockets.exceptions.InvalidStatus as exc:
                # The reason is only in the body; without it a rejected param
                # looks like an agent that never hears anybody.
                detail = ""
                try:
                    detail = exc.response.body.decode()[:300]
                except Exception:
                    pass
                logger.error(
                    "[Deepgram] Handshake rejected status=%s detail=%s params=%s",
                    exc.response.status_code, detail, params,
                )
                raise
            self.receiver_task = asyncio.create_task(self._receive())
            logger.info(
                "[Deepgram] Connected participant_id=%s participant_name=%s",
                self.participant_id,
                self.participant_name,
            )

    async def send(self, pcm: bytes) -> None:
        await self.ensure_started()
        await self.ws.send(pcm)

    async def _receive(self) -> None:
        try:
            while True:
                now = time.monotonic()
                data = json.loads(await self.ws.recv())
                msg_type = data.get("type")

                has_words = False
                if msg_type == "Results":
                    alternatives = data.get("channel", {}).get("alternatives", [])
                    has_words = bool(alternatives and (alternatives[0].get("transcript") or "").strip())

                utterance = self.buffer.add_result(data)
                if utterance:
                    # ~0 means speech_final fired (fast path); a large gap means
                    # it fell back to utterance_end_ms.
                    if self._last_words_at is not None:
                        logger.info(
                            "[TIMING] deepgram_finalize_wait=%.2fs trigger=%s participant_id=%s "
                            "(last words heard -> Deepgram signaled utterance done)",
                            now - self._last_words_at, msg_type, self.participant_id,
                        )
                    self._last_words_at = None
                    self.on_utterance(self.participant_id, self.participant_name, utterance)
                elif has_words:
                    self._last_words_at = now

                if msg_type == "Error":
                    logger.error("[Deepgram] participant_id=%s error=%s", self.participant_id, data)
        except (ConnectionClosed, asyncio.CancelledError):
            pass
        except Exception:
            logger.error("[Deepgram] Receiver failed participant_id=%s", self.participant_id, exc_info=True)

    async def close(self) -> None:
        if self.ws:
            try:
                await self.ws.send(json.dumps({"type": "CloseStream"}))
                await self.ws.close()
            except Exception:
                pass
        if self.receiver_task:
            self.receiver_task.cancel()
