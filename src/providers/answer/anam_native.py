"""Anam's own persona LLM as the answer engine.

The turn gate still runs: agent_policy decides the turn happens, and only then
is the transcript handed over. What is given up is control of the *content* --
compose_reply, normalize_reply and AGENT_MAX_REPLY_WORDS never run, so the only
levers on what comes back are the persona's systemPrompt and directorNotes.
That is a real trade, not an implementation gap; see
docs/PROVIDER_REFACTOR_PLAN.md §8.
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from src.core import protocol
from src.providers.base import AnswerEngine, DelegatedResult, TurnContext, VideoSession

logger = logging.getLogger("LiveAvatar-Spiked")

# How long to wait for the vendor to finish speaking before giving up on the
# turn. Generous: this covers Anam's own LLM latency plus the whole spoken
# reply, not just a network round trip.
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
