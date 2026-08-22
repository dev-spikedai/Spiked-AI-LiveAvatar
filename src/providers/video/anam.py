"""Anam -- face, optionally voice, optionally brain.

Two configurations, one adapter, distinguished by a single field:

  driven  llmId = "CUSTOMER_CLIENT_V1"  -> Anam's own LLM is off. We stream our
          answer in with talk commands. This is the drop-in replacement for
          LiveAvatar FULL and the launch configuration.

  native  llmId = <a real model UUID>   -> Anam composes the reply itself from
          systemPrompt. We still gate the turn and hand over the transcript via
          sendUserMessage(); see src/providers/answer/anam_native.py.

Both set disableInputAudio: the avatar page never hears the meeting in any
configuration. Recall -> backend -> Deepgram owns listening, which is what keeps
the turn gate intact even when the vendor owns the answer.

Anam also supports audio passthrough (PCM16/16k, same as Simli) via
enableAudioPassthrough. Not used here: passthrough imposes an ~800ms buffer
before render that text mode does not pay, so text is the better default while
Anam is doing its own TTS. `accepts` is the only line that would change.
"""

import logging
import os
from typing import Optional

import httpx
from fastapi import HTTPException

from src.providers.base import RunContext, VideoProvider, VideoSession

logger = logging.getLogger("LiveAvatar-Spiked")

ANAM_API_KEY = os.getenv("ANAM_API_KEY", "")
ANAM_BASE_URL = os.getenv("ANAM_API_URL", "https://api.anam.ai")
ANAM_AVATAR_ID = os.getenv("ANAM_AVATAR_ID", "")
ANAM_VOICE_ID = os.getenv("ANAM_VOICE_ID", "")
ANAM_AVATAR_MODEL = os.getenv("ANAM_AVATAR_MODEL", "cara-4-latest")
ANAM_REGION = os.getenv("ANAM_REGION", "us")
ANAM_MAX_SESSION_SECONDS = int(os.getenv("ANAM_MAX_SESSION_SECONDS", "1800"))
# The documented sentinel that turns Anam's built-in brain off and hands all
# conversation logic to the client. Not a UUID, and not optional in driven
# mode -- with a real llmId Anam will answer on its own and talk over the gate.
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
        #: native=True hands the brain to Anam. The paired AnswerEngine must be
        #: the delegated one; the registry refuses the mismatched combination.
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
            # skipGreeting matters in both modes: an avatar that introduces
            # itself on join has spoken in the meeting without passing the turn
            # gate, which is exactly the thing the gate exists to prevent.
            "skipGreeting": True,
            "maxSessionLengthSeconds": ANAM_MAX_SESSION_SECONDS,
        }
        if ANAM_VOICE_ID:
            persona_config["voiceId"] = ANAM_VOICE_ID
        if self.native:
            # The only lever on what a native-mode reply says or how long it
            # runs -- compose_reply and AGENT_MAX_REPLY_WORDS do not apply.
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
                # The browser half needs to know which mode it is in: driven
                # renders talk streams, native forwards transcripts instead.
                "anam_mode": "native" if self.native else "driven",
                # Never let the page open a mic. Meeting audio reaches the agent
                # through Recall, and a second listener would double-hear the
                # room and answer outside the gate.
                "anam_disable_input_audio": True,
            },
        )
