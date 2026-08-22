"""LiveAvatar (HeyGen). Text in, face and voice out. FULL mode only."""

import logging
import os
from typing import Optional

import httpx
from fastapi import HTTPException

from src.providers.base import RunContext, VideoProvider, VideoSession

logger = logging.getLogger("SpikedMeetingAgent")

LIVEAVATAR_API_KEY = os.getenv("LIVEAVATAR_API_KEY", "")
LIVEAVATAR_BASE_URL = os.getenv("LIVEAVATAR_API_URL", "https://api.liveavatar.com")
LIVEAVATAR_AVATAR_ID = "9650a758-1085-4d49-8bf3-f347565ec229"
LIVEAVATAR_SANDBOX = os.getenv("LIVEAVATAR_SANDBOX", "false").lower() == "true"
# LiveAvatar auto-closes a session that sees no join/interaction for a while
# (docs.liveavatar.com/reference/keep_session_alive_v1_sessions_keep_alive_post).
# Tom is silent unless addressed, so a long idle stretch in a meeting would
# otherwise trip that timeout — ping it well under the observed ~3 minute cliff.
LIVEAVATAR_KEEPALIVE_INTERVAL_S = float(os.getenv("LIVEAVATAR_KEEPALIVE_INTERVAL_S", "60"))


class LiveAvatarVideoProvider(VideoProvider):
    name = "liveavatar"
    accepts = "text"
    browser_module = "/providers/liveavatar.js"
    keepalive_interval_s = LIVEAVATAR_KEEPALIVE_INTERVAL_S

    def __init__(self, mode: str = "FULL", quality: str = "medium", is_sandbox: Optional[bool] = None):
        self.mode = mode
        self.quality = quality
        self.is_sandbox = LIVEAVATAR_SANDBOX if is_sandbox is None else is_sandbox

    @property
    def _headers(self) -> dict:
        return {"Content-Type": "application/json", "X-API-KEY": LIVEAVATAR_API_KEY}

    async def create_session(self, ctx: RunContext) -> VideoSession:
        if not LIVEAVATAR_API_KEY:
            raise HTTPException(status_code=500, detail="LIVEAVATAR_API_KEY is not configured")

        avatar_id = ctx.avatar_id or LIVEAVATAR_AVATAR_ID or None

        token_payload = {
            "mode": self.mode,
            "avatar_id": avatar_id,
            "video_settings": {"quality": self.quality, "encoding": "H264"},
            "is_sandbox": self.is_sandbox,
        }
        if self.mode == "FULL":
            # PUSH_TO_TALK + empty persona: stops HeyGen's own LLM answering
            # outside the turn gate.
            token_payload["interactivity_type"] = "PUSH_TO_TALK"
            token_payload["avatar_persona"] = {}

        async with httpx.AsyncClient(timeout=15.0) as client:
            token_res = await client.post(
                f"{LIVEAVATAR_BASE_URL}/v1/sessions/token",
                headers=self._headers,
                json=token_payload,
            )
            if token_res.status_code not in (200, 201):
                logger.error("LiveAvatar token creation failed: %s", token_res.text)
                raise HTTPException(status_code=token_res.status_code, detail=token_res.text)

            token_json = token_res.json()
            token_data = token_json.get("data", {}) if isinstance(token_json.get("data"), dict) else token_json
            session_token = token_data.get("session_token")

            start_res = await client.post(
                f"{LIVEAVATAR_BASE_URL}/v1/sessions/start",
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {session_token}"},
                json={},
            )
            if start_res.status_code not in (200, 201):
                logger.error("LiveAvatar start session failed: %s", start_res.text)
                raise HTTPException(status_code=start_res.status_code, detail=start_res.text)

            start_json = start_res.json()
            start_data = start_json.get("data", {}) if isinstance(start_json.get("data"), dict) else start_json

        session_id = start_data.get("session_id")
        logger.info("[LiveAvatar] Session started in %s mode: session_id=%s", self.mode, session_id)

        return VideoSession(
            provider=self.name,
            session_id=session_id,
            credentials={
                "mode": self.mode,
                "rate": "$0.20/minute" if self.mode == "FULL" else "$0.10/minute",
                "session_id": session_id,
                "livekit_url": start_data.get("livekit_url"),
                "livekit_token": start_data.get("livekit_client_token"),
                "session_token": session_token,
            },
        )

    async def keepalive(self, session: VideoSession) -> None:
        if not session.session_id:
            return
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{LIVEAVATAR_BASE_URL}/v1/sessions/keep-alive",
                headers=self._headers,
                json={"session_id": session.session_id},
            )
            if resp.status_code not in (200, 201):
                logger.warning(
                    "[LiveAvatar] keep-alive for session %s -> %s: %s",
                    session.session_id, resp.status_code, resp.text,
                )

    async def close(self, session: VideoSession) -> None:
        """One concurrent session per key: a leaked session blocks every later
        start with `4032 Session concurrency limit reached`."""
        if not session.session_id or not LIVEAVATAR_API_KEY:
            return
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(
                    f"{LIVEAVATAR_BASE_URL}/v1/sessions/stop",
                    headers=self._headers,
                    json={"session_id": session.session_id},
                )
        except Exception:
            logger.warning("[LiveAvatar] session stop failed for %s", session.session_id, exc_info=True)
