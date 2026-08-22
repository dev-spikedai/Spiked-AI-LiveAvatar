"""Simli. Face only, so always paired with a TtsProvider. Reference adapter."""

import logging
import os

import httpx
from fastapi import HTTPException

from src.core.protocol import PCM16_16K_MONO
from src.providers.base import RunContext, VideoProvider, VideoSession

logger = logging.getLogger("SpikedMeetingAgent")

SIMLI_API_KEY = os.getenv("SIMLI_API_KEY", "")
SIMLI_FACE_ID = os.getenv("SIMLI_FACE_ID", "")
SIMLI_BASE_URL = os.getenv("SIMLI_API_URL", "https://api.simli.ai")
SIMLI_MAX_SESSION_SECONDS = int(os.getenv("SIMLI_MAX_SESSION_SECONDS", "1800"))
SIMLI_MAX_IDLE_SECONDS = int(os.getenv("SIMLI_MAX_IDLE_SECONDS", "300"))
# ~187ms of PCM16@16k per frame.
SIMLI_AUDIO_CHUNK_BYTES = int(os.getenv("SIMLI_AUDIO_CHUNK_BYTES", "6000"))


class SimliVideoProvider(VideoProvider):
    name = "simli"
    accepts = "audio"
    browser_module = "/providers/simli.js"
    audio_format = PCM16_16K_MONO
    # Bounded by maxIdleTime on the token; nothing to ping.
    keepalive_interval_s = 0.0

    chunk_bytes = SIMLI_AUDIO_CHUNK_BYTES

    async def create_session(self, ctx: RunContext) -> VideoSession:
        if not SIMLI_API_KEY:
            raise HTTPException(status_code=500, detail="SIMLI_API_KEY is not configured")
        if not SIMLI_FACE_ID:
            raise HTTPException(status_code=500, detail="SIMLI_FACE_ID is not configured")

        payload = {
            "faceId": SIMLI_FACE_ID,
            "maxSessionLength": SIMLI_MAX_SESSION_SECONDS,
            "maxIdleTime": SIMLI_MAX_IDLE_SECONDS,
            "handleSilence": True,
            "audioInputFormat": "pcm16",
        }

        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                f"{SIMLI_BASE_URL.rstrip('/')}/compose/token",
                headers={"Content-Type": "application/json", "x-simli-api-key": SIMLI_API_KEY},
                json=payload,
            )

        if resp.status_code not in (200, 201):
            logger.error("[Simli] token creation failed: %s", resp.text)
            raise HTTPException(status_code=resp.status_code, detail=resp.text[:300])

        data = resp.json()
        token = data.get("session_token") if isinstance(data, dict) else None
        # Simli can return "FAIL"/"FAIL TOKEN" with a 200.
        if not token or not isinstance(token, str) or token in ("FAIL TOKEN", "FAIL"):
            logger.error("[Simli] token response invalid: %s", data)
            raise HTTPException(status_code=500, detail="Simli session token creation failed")

        logger.info("[Simli] Session token created face_id=%s", SIMLI_FACE_ID)
        return VideoSession(
            provider=self.name,
            session_id=None,  # Simli's session identity is the token itself
            credentials={
                "simli_session_token": token,
                "simli_base_url": SIMLI_BASE_URL,
            },
        )
