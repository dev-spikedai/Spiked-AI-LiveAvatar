"""Anam's persona LLM as the answer engine. Gate stays ours, content does not."""

import asyncio
import logging
from typing import Any, Dict, Optional

from src.core import protocol
from src.providers.base import AnswerEngine, DelegatedResult, TurnContext, VideoSession

logger = logging.getLogger("SpikedMeetingAgent")

# Covers Anam's LLM latency plus the whole spoken reply, not just a round trip.
ANAM_NATIVE_TURN_TIMEOUT_S = 45.0


class AnamNativeAnswerEngine(AnswerEngine):
    name = "anam_native"
    mode = "delegated"

    def __init__(self, send_control: Any, wait_for_reply: Any):
        """`send_control(message)` puts a message on the run's control socket;
        `wait_for_reply(turn_id, timeout)` resolves when the page reports the
        vendor finished speaking. Both are injected by the executor so this
        engine never touches run state directly.
        """
        self._send_control = send_control
        self._wait_for_reply = wait_for_reply

    async def delegate_turn(self, ctx: TurnContext, session: VideoSession) -> DelegatedResult:
        await self._send_control(protocol.avatar_user_message(ctx.question, ctx.turn_id))
        try:
            reply: Dict[str, Any] = await self._wait_for_reply(
                ctx.turn_id, ANAM_NATIVE_TURN_TIMEOUT_S
            )
        except asyncio.TimeoutError:
            # The floor must be released either way, or the agent is stuck in
            # SPEAKING forever on a vendor that never answered.
            logger.warning(
                "[Anam] native turn timed out run_id=%s turn_id=%s", ctx.run_id, ctx.turn_id
            )
            return DelegatedResult(spoken_text="", interrupted=True)

        spoken: Optional[str] = (reply or {}).get("text")
        if not spoken:
            # Speech happened but its text never arrived, so the echo suppressor
            # has no record of it. Worth a warning: this is the failure mode
            # where the agent starts answering itself.
            logger.warning(
                "[Anam] native turn produced no reply text run_id=%s turn_id=%s "
                "-- echo suppression is blind for this turn",
                ctx.run_id, ctx.turn_id,
            )
        return DelegatedResult(
            spoken_text=spoken or "",
            interrupted=bool((reply or {}).get("interrupted")),
        )
