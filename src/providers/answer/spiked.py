"""Spiked /ask as an answer engine. Wraps live_avatar.query_spiked_rag for now."""

import logging
from typing import AsyncIterator

from src.providers.base import AnswerEngine, TurnContext

logger = logging.getLogger("SpikedMeetingAgent")


class SpikedAnswerEngine(AnswerEngine):
    name = "spiked"
    mode = "stream"

    async def stream_answer(self, ctx: TurnContext) -> AsyncIterator[str]:
        """Turn query_spiked_rag's sentence callbacks into a pull interface."""
        import asyncio

        # Lazy: live_avatar -> registry -> this module would cycle.
        from src import live_avatar

        queue: "asyncio.Queue[object]" = asyncio.Queue()
        _DONE = object()

        async def on_sentence(sentence: str) -> None:
            await queue.put(sentence)

        async def pump() -> None:
            try:
                await live_avatar.query_spiked_rag(
                    ctx.question,
                    auth_token=ctx.auth_token,
                    client_id=ctx.client_id,
                    on_sentence=on_sentence,
                )
            except Exception:
                logger.error("[Spiked] retrieval failed run_id=%s turn_id=%s",
                             ctx.run_id, ctx.turn_id, exc_info=True)
            finally:
                await queue.put(_DONE)

        task = asyncio.create_task(pump())
        try:
            while True:
                item = await queue.get()
                if item is _DONE:
                    return
                yield item  # type: ignore[misc]
        finally:
            # Early exit (barge-in, backstop) must not leave retrieval running.
            if not task.done():
                task.cancel()
