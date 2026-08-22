"""The Spiked backend as an answer engine -- /ask streaming retrieval.

This is a thin adapter, not a move. The retrieval path in live_avatar.py
(source_id resolution, the per-run cache, the cognitive fallback, the degraded-
answer markers) is still entangled with the turn pipeline, and pulling it out
wholesale is step 2 work that has not happened yet. Wrapping it now costs one
lazy import and gets the interface real, which is what unblocks every other
provider.
"""

import logging
from typing import AsyncIterator

from src.providers.base import AnswerEngine, TurnContext

logger = logging.getLogger("LiveAvatar-Spiked")


class SpikedAnswerEngine(AnswerEngine):
    name = "spiked"
    mode = "stream"

    async def stream_answer(self, ctx: TurnContext) -> AsyncIterator[str]:
        """Yield the backend's answer one complete sentence at a time.

        query_spiked_rag already emits sentence callbacks; this turns that
        push interface into the pull interface the executor wants, via a
        queue, so the caller controls pacing rather than the retrieval code.
        """
        import asyncio

        # Imported here, not at module scope: live_avatar imports the registry,
        # and the registry imports this module. Delete once the retrieval code
        # moves to src/core (plan step 2) and the cycle goes away with it.
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
            # A consumer that stops early (barge-in, word backstop) must not
            # leave retrieval running and writing into a queue nobody reads.
            if not task.done():
                task.cancel()
