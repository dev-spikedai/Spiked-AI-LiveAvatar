"""Anam. Driven or native, switched by llmId. See docs/PROVIDER_REFACTOR_PLAN.md §5."""

import logging
import os
from typing import Optional

import httpx
from fastapi import HTTPException

from src.providers.base import RunContext, VideoProvider, VideoSession

logger = logging.getLogger("SpikedMeetingAgent")

ANAM_API_KEY = os.getenv("ANAM_API_KEY", "")
ANAM_BASE_URL = os.getenv("ANAM_API_URL", "https://api.anam.ai")
ANAM_AVATAR_ID = os.getenv("ANAM_AVATAR_ID", "")
ANAM_VOICE_ID = os.getenv("ANAM_VOICE_ID", "")
ANAM_AVATAR_MODEL = os.getenv("ANAM_AVATAR_MODEL", "cara-4-latest")
ANAM_REGION = os.getenv("ANAM_REGION", "us")
ANAM_MAX_SESSION_SECONDS = int(os.getenv("ANAM_MAX_SESSION_SECONDS", "1800"))
# Sentinel that turns Anam's brain off; a real llmId makes it answer on its own.
ANAM_CLIENT_DRIVEN_LLM_ID = "CUSTOMER_CLIENT_V1"
# Only consulted in native mode.
ANAM_NATIVE_LLM_ID = os.getenv("ANAM_NATIVE_LLM_ID", "")


class AnamVideoProvider(VideoProvider):
    name = "anam"
    accepts = "text"
    browser_module = "/providers/anam.js"
    # Session length is bounded at token creation; there is nothing to ping.
    keepalive_interval_s = 0.0

    def __init__(self, native: bool = False, llm_id: Optional[str] = None):
        self.native = native
        self.llm_id = llm_id or (ANAM_NATIVE_LLM_ID if native else ANAM_CLIENT_DRIVEN_LLM_ID)

    async def create_session(self, ctx: RunContext) -> VideoSession:
        if not ANAM_API_KEY:
            raise HTTPException(status_code=500, detail="ANAM_API_KEY is not configured")
        avatar_id = ctx.avatar_id or ANAM_AVATAR_ID
        if not avatar_id:
            raise HTTPException(status_code=500, detail="ANAM_AVATAR_ID is not configured")
        if self.native and not self.llm_id:
            raise HTTPException(
                status_code=500,
                detail="ANAM_NATIVE_LLM_ID is required when running Anam in native mode",
            )

        persona_config = {
            "name": ctx.bot_name,
            "avatarId": avatar_id,
            "avatarModel": ANAM_AVATAR_MODEL,
            "llmId": self.llm_id,
            # A greeting on join is speech that never passed the turn gate.
            "skipGreeting": True,
            "maxSessionLengthSeconds": ANAM_MAX_SESSION_SECONDS,
        }
        if ANAM_VOICE_ID:
            persona_config["voiceId"] = ANAM_VOICE_ID
        if self.native:
            # Only lever on native replies; AGENT_MAX_REPLY_WORDS does not apply.
            persona_config["systemPrompt"] = ctx.persona_prompt or ""

        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                f"{ANAM_BASE_URL.rstrip('/')}/v1/auth/session-token",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {ANAM_API_KEY}",
                },
                json={
                    "personaConfig": persona_config,
                    "sessionOptions": {"region": ANAM_REGION, "videoQuality": "auto"},
                },
            )

        if resp.status_code not in (200, 201):
            logger.error("[Anam] session token creation failed: %s %s", resp.status_code, resp.text[:300])
            raise HTTPException(status_code=resp.status_code, detail=resp.text[:300])

        data = resp.json()
        token = data.get("sessionToken") if isinstance(data, dict) else None
        if not token:
            logger.error("[Anam] session token response invalid: %s", data)
            raise HTTPException(status_code=500, detail="Anam session token creation failed")

        logger.info(
            "[Anam] Session token created avatar_id=%s mode=%s",
            avatar_id, "native" if self.native else "driven",
        )
        return VideoSession(
            provider=self.name,
            session_id=None,  # Anam's session identity is the token itself
            credentials={
                "anam_session_token": token,
                "anam_mode": "native" if self.native else "driven",
                # A page mic would be a second listener answering outside the gate.
                "anam_disable_input_audio": True,
            },
        )
