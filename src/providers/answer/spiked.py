"""Spiked /ask as an answer engine."""

from typing import Any, Optional

from src.providers.base import AnswerEngine, TurnContext


class SpikedAnswerEngine(AnswerEngine):
    name = "spiked"

    async def answer(self, ctx: TurnContext, on_sentence: Optional[Any] = None) -> str:
        # Lazy: live_avatar -> registry -> this module would cycle. Goes away
        # when retrieval moves into src/core.
        from src import live_avatar

        return await live_avatar.query_spiked_rag(
            ctx.question,
            ctx.auth_token or "",
            ctx.client_id,
            source_ids=ctx.source_ids,
            timeout_s=ctx.timeout_s,
            kyc_id=ctx.kyc_id,
            source_ids_task=ctx.source_ids_task,
            on_sentence=on_sentence,
        )
