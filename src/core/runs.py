"""The run registry and teardown.

_ACTIVE_RUNS is process-local: a run exists only in the process that created
it, which is why teardown has no cross-version fallback path.
"""

import asyncio
import logging
import os
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger("SpikedMeetingAgent")

RECALL_API_KEY = os.getenv("RECALL_API_KEY", "")
RECALL_BASE_URL = os.getenv("RECALL_BASE_URL", "https://ap-northeast-1.recall.ai")

_ACTIVE_RUNS: Dict[str, Dict[str, Any]] = {}


async def _leave_recall_call(bot_id: Optional[str]) -> Optional[int]:
    if not bot_id or not RECALL_API_KEY:
        return None
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{RECALL_BASE_URL.rstrip('/')}/api/v1/bot/{bot_id}/leave_call/",
                headers={"Authorization": f"Token {RECALL_API_KEY}"},
            )
        logger.info("[Teardown] Recall bot %s leave_call -> %s", bot_id, resp.status_code)
        return resp.status_code
    except Exception:
        logger.error("[Teardown] Failed to evict Recall bot %s", bot_id, exc_info=True)
        return None


async def _no_session() -> None:
    return None


async def _teardown_run(run_id: str) -> Dict[str, Any]:
    """Release every resource a run holds, then evict it from the registry.

    Written to be idempotent and to never raise: disconnect is the path a user
    reaches for when something has already gone wrong, so a failure in one leg
    must not strand the others. Each leg is attempted independently and the
    run is evicted regardless of what the remote services report.
    """
    run = _ACTIVE_RUNS.get(run_id)
    if not run:
        return {"ok": True, "already_gone": True}

    for key in ("active_response_task", "watchdog_task", "keepalive_task", "mute_expiry_task"):
        task = run.get(key)
        if task and not task.done():
            task.cancel()
    for entry in (run.get("pending_turns") or {}).values():
        timer = entry.get("timer")
        if timer and not timer.done():
            timer.cancel()

    intel = run.get("intel")
    if intel:
        intel.stop()

    control_ws = run.get("control_ws")
    if control_ws:
        try:
            await control_ws.close()
        except Exception:
            pass
        run["control_ws"] = None

    for socket in list(run.get("rep_sockets") or ()):
        try:
            await socket.close()
        except Exception:
            pass
    run["rep_sockets"] = set()

    # Tolerant of a half-built run: teardown is the path reached when
    # something has already gone wrong, so a missing session must not raise.
    providers, session = run.get("providers"), run.get("video_session")
    recall_status, avatar_status = await asyncio.gather(
        _leave_recall_call(run.get("bot_id")),
        providers.video.close(session) if providers and session else _no_session(),
        return_exceptions=True,
    )

    _ACTIVE_RUNS.pop(run_id, None)
    logger.info("[Teardown] Run %s released (%d still active)", run_id, len(_ACTIVE_RUNS))
    return {
        "ok": True,
        "run_id": run_id,
        "recall_status": recall_status,
        "avatar_session_status": avatar_status,
    }


def _find_run_id_by_bot(bot_id: str) -> Optional[str]:
    for run_id, run in _ACTIVE_RUNS.items():
        if run.get("bot_id") == bot_id:
            return run_id
    return None
