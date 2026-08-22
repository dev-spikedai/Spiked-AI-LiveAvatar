"""Cartesia TTS. Pinned to raw pcm_s16le @ 16kHz so nothing resamples."""

import logging
import os
from typing import AsyncIterator

import httpx

from src.core.protocol import AudioFormat
from src.providers.base import TtsProvider

logger = logging.getLogger("SpikedMeetingAgent")

CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", "")
CARTESIA_BASE_URL = os.getenv("CARTESIA_BASE_URL", "https://api.cartesia.ai")
CARTESIA_VOICE_ID = os.getenv("CARTESIA_VOICE_ID", "f786b574-daa5-4673-aa0c-cbe3e8534c02")
CARTESIA_MODEL = os.getenv("CARTESIA_MODEL", "sonic-3.5")
CARTESIA_SAMPLE_RATE = int(os.getenv("CARTESIA_SAMPLE_RATE", "16000"))
CARTESIA_VERSION = os.getenv("CARTESIA_VERSION", "2026-03-01")


class CartesiaTtsProvider(TtsProvider):
    name = "cartesia"
    audio_format = AudioFormat(sample_rate=CARTESIA_SAMPLE_RATE, channels=1, encoding="pcm_s16le")

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        """Yield raw PCM as it arrives; one POST per utterance."""
        if not CARTESIA_API_KEY:
            raise RuntimeError("CARTESIA_API_KEY is not configured")

        payload = {
            "model_id": CARTESIA_MODEL,
            "transcript": text,
            "voice": {"mode": "id", "id": CARTESIA_VOICE_ID},
            "output_format": {
                "container": "raw",
                "encoding": self.audio_format.encoding,
                "sample_rate": self.audio_format.sample_rate,
            },
            "language": "en",
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {CARTESIA_API_KEY}",
            "Cartesia-Version": CARTESIA_VERSION,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST", f"{CARTESIA_BASE_URL.rstrip('/')}/tts/bytes", headers=headers, json=payload
            ) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode("utf-8", "replace")
                    logger.error("[Cartesia] TTS failed: %s %s", resp.status_code, body[:300])
                    raise RuntimeError(f"Cartesia TTS failed ({resp.status_code})")
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        yield chunk
