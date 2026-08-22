"""Floor state, speech dispatch, and the notification fan-out.

Provider-agnostic: every send goes through core.speech, which picks text or
PCM from the run's provider. See docs/PROVIDER_REFACTOR_PLAN.md.
"""

import asyncio
import logging
import os
import re
import time
from typing import Any, Awaitable, Callable, Dict, Optional

from src.agent_policy import (
    AgentState,
    FloorState,
    MAX_QUESTION_WORDS,
    SpeechGovernor,
    estimate_speech_seconds,
)
from src.core import speech

logger = logging.getLogger("SpikedMeetingAgent")

AGENT_BARGE_IN_MS = int(os.getenv("AGENT_BARGE_IN_MS", "700"))
# Scales the follow-up window by how long the reply took to say.
AGENT_FOLLOWUP_WINDOW_REPLY_SCALE = float(os.getenv("AGENT_FOLLOWUP_WINDOW_REPLY_SCALE", "0.5"))


def _push_heard(
    run: Dict[str, Any],
    speaker: str,
    text: str,
    reply: bool,
    reason: str,
) -> None:
    """Mirror the gate's verdict onto the avatar overlay and the rep console.

    In avatar mode this is the *only* path a transcript can reach the frontend:
    the recall backend never saw this bot created, so its transcript stream is
    empty by construction. These are the same finalized, speaker-attributed
    turns the agent itself reasons over.
    """
    _push_rep(run, {
        "type": "heard",
        "speaker": speaker,
        "text": text,
        "reply": reply,
        "reason": reason,
    })

    control_ws = run.get("control_ws")
    if not control_ws:
        return

    async def send() -> None:
        try:
            await control_ws.send_json({
                "type": "heard",
                "speaker": speaker,
                "text": text,
                "reply": reply,
                "reason": reason,
            })
        except Exception:
            pass

    asyncio.create_task(send())


def _push_control(run: Dict[str, Any], message: Dict[str, Any]) -> None:
    """Fire-and-forget a message to the avatar page itself (/ws/control).

    Distinct from _push_rep: that reaches the rep's console, this reaches the
    actual meeting video feed avatar.js renders — used for the mute countdown
    overlay, which every meeting participant sees, not just the rep.
    """
    control_ws = run.get("control_ws")
    if not control_ws:
        return

    async def send() -> None:
        try:
            await control_ws.send_json(message)
        except Exception:
            pass

    try:
        asyncio.create_task(send())
    except RuntimeError:
        send().close()


def _push_rep(run: Dict[str, Any], message: Dict[str, Any]) -> None:
    """Fire-and-forget a message to the rep's console, if one is attached.

    Never awaited by the turn pipeline: a rep with a wedged socket must not be
    able to slow down or block what the agent says in the room.
    """
    sockets = list(run.get("rep_sockets") or ())
    if not sockets:
        return

    async def send() -> None:
        for socket in sockets:
            try:
                await socket.send_json(message)
            except Exception:
                # A dead console is dropped on its own disconnect path; failing
                # to reach one must not stop the others from being told.
                pass

    try:
        asyncio.create_task(send())
    except RuntimeError:
        # No running loop. Notifying the console is never worth raising into a
        # caller — _release_floor is on the speak path, and an exception there
        # would strand the agent out of LISTENING permanently.
        send().close()


def _set_mute(run: Dict[str, Any], seconds: int) -> None:
    """Mute the agent for `seconds`. A voice command reissued while already
    muted simply replaces the previous timer, matching how the wake name
    always grants the floor regardless of current state."""
    now_mono = time.monotonic()
    until_mono = now_mono + seconds
    until_epoch_ms = int((time.time() + seconds) * 1000)
    run["muted_until"] = until_mono
    run["muted_until_epoch_ms"] = until_epoch_ms

    previous = run.get("mute_expiry_task")
    if previous and not previous.done():
        previous.cancel()

    logger.info("[Mute] Agent muted for %ds run_id=%s", seconds, run.get("run_id"))
    mute_message = {"type": "agent_muted", "muted_until_epoch_ms": until_epoch_ms, "seconds": seconds}
    _push_rep(run, mute_message)
    _push_control(run, mute_message)

    async def expire() -> None:
        try:
            await asyncio.sleep(seconds)
            # Only clear if nothing re-muted (or cleared) it in the meantime.
            if run.get("muted_until") == until_mono:
                run["muted_until"] = None
                run["muted_until_epoch_ms"] = None
                logger.info("[Mute] Expired run_id=%s", run.get("run_id"))
                _push_rep(run, {"type": "agent_unmuted", "reason": "expired"})
                _push_control(run, {"type": "agent_unmuted", "reason": "expired"})
        except asyncio.CancelledError:
            pass

    run["mute_expiry_task"] = asyncio.create_task(expire())


def _clear_mute(run: Dict[str, Any], reason: str = "invoked") -> None:
    """Ask Tom / explicit invocation always overrides mute — the button press
    is itself the invitation, so it bypasses mute the same way it bypasses
    the wake name and the addressee gate."""
    if run.get("muted_until") is None:
        return
    run["muted_until"] = None
    run["muted_until_epoch_ms"] = None
    task = run.get("mute_expiry_task")
    if task and not task.done():
        task.cancel()
    logger.info("[Mute] Cleared (%s) run_id=%s", reason, run.get("run_id"))
    _push_rep(run, {"type": "agent_unmuted", "reason": reason})
    _push_control(run, {"type": "agent_unmuted", "reason": reason})


def _push_insight(run: Dict[str, Any], speaker: str, topic: str) -> None:
    """Signal that the agent has something, without it taking the floor.

    This is the whole of Level 1 on the wire: the rep's console learns the agent
    could contribute, and nothing is said in the room unless somebody accepts.
    """
    _push_rep(run, {
        "type": "insight_available",
        "speaker": speaker,
        "topic": topic,
    })


async def _dispatch_reply(run: Dict[str, Any], answer: str, turn_id: int, source: str = "addressed") -> bool:
    """Send a finished reply to the avatar and arm the speak watchdog.

    Shared by the addressed-turn path, the invoke endpoint, and the Level 1.5
    autonomous path so every turn passes the same duplicate guard and echo
    suppression — invitation bypasses the wake name, not the safety rails.
    `source` ("addressed" | "invoke" | "autonomous") rides along on the rep
    socket's agent_spoke event only, purely for console display/audit — it
    has no effect on gating.
    """
    governor: SpeechGovernor = run["governor"]
    bot_name = run.get("bot_name") or "Tom"
    spoken_at = time.monotonic()

    if governor.is_duplicate(answer, spoken_at):
        logger.info("[Governor] Suppressed duplicate reply turn_id=%s", turn_id)
        _release_floor(run, spoken_at)
        return False

    control_ws = run.get("control_ws")
    if not control_ws:
        logger.warning("[Agent] Avatar control socket unavailable for turn_id=%s", turn_id)
        _release_floor(run, spoken_at)
        return False

    history = run.setdefault("history", [])
    history.append({"speaker": bot_name, "participant_id": "bot", "text": answer})
    del history[:-40]
    # Register before dispatch: the echo can return before the socket ack.
    run["echo"].note_bot_speech(answer, spoken_at)
    governor.note_reply(answer, spoken_at)
    _push_rep(run, {"type": "agent_spoke", "text": answer, "turn_id": turn_id, "source": source})

    timing = run.setdefault("turn_timing", {}).setdefault(turn_id, {})
    timing["dispatched_at"] = spoken_at
    finalized_at = timing.get("finalized_at")
    if finalized_at is not None:
        logger.info(
            "[TIMING] turn_id=%s finalize->dispatch=%.2fs (turn gate + classification + RAG)",
            turn_id, spoken_at - finalized_at,
        )

    await speech.emit_speech(run, turn_id, None, answer)
    run["watchdog_task"] = asyncio.create_task(_speak_watchdog(run, turn_id, answer))
    return True


async def _speak_chunk(run: Dict[str, Any], turn_id: int, chunk_id: str, text: str) -> bool:
    """Send one sentence and wait for it to actually finish playing before
    returning, so sequential chunks of one answer never overlap in the
    avatar's audio. Returns False once the turn is no longer live (barge-in,
    superseded turn, or the control socket dropped) — callers use that to
    stop dispatching further chunks rather than talking over a turn that's
    already over.
    """
    control_ws = run.get("control_ws")
    # active_turn_id alone isn't enough: an in-place barge-in during SPEAKING
    # releases the floor (state -> LISTENING) without changing active_turn_id
    # at all, so a state check is the only thing that actually catches "this
    # exact turn was just interrupted" between one chunk and the next.
    if (
        not control_ws
        or run.get("active_turn_id") != turn_id
        or run.get("state") not in (AgentState.THINKING, AgentState.SPEAKING)
    ):
        return False
    event = asyncio.Event()
    run.setdefault("chunk_events", {})[chunk_id] = event
    # Stamp once, on this turn's first chunk only: the same field name
    # _dispatch_reply writes for the legacy single-shot path, so the
    # avatar_speak_started handler's dispatch->speaking log works for both
    # without duplicating that logic. Streamed replies never went through
    # _dispatch_reply at all, so without this, that log line's gating field
    # was silently never set and the line never printed — see live_avatar.py
    # avatar_control_endpoint's avatar_speak_started handling.
    timing = run.setdefault("turn_timing", {}).setdefault(turn_id, {})
    timing.setdefault("dispatched_at", time.monotonic())
    try:
        await speech.emit_speech(run, turn_id, chunk_id, text)
        timeout = estimate_speech_seconds(text) + 5.0
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("[Stream] turn_id=%s chunk_id=%s timed out waiting for speak_ended; continuing", turn_id, chunk_id)
    finally:
        (run.get("chunk_events") or {}).pop(chunk_id, None)
    return run.get("active_turn_id") == turn_id and run.get("state") in (AgentState.THINKING, AgentState.SPEAKING)


def _make_streaming_sentence_handler(
    run: Dict[str, Any], turn_id: int, reply_word_limit: int,
    filler_task: "Optional[asyncio.Task[bool]]" = None,
) -> "tuple[Callable[[str], Awaitable[None]], Dict[str, Any]]":
    """Build the on_sentence callback query_spiked_rag calls per sentence.

    Speaks each complete sentence as soon as it's ready instead of waiting
    for the whole answer — the entire point of streaming. Two safety checks
    apply before anything is actually spoken:
    - The first sentence is checked against the same degraded/error markers
      _generate_grounded_reply already checks the full answer against. If it
      looks like a failure, nothing is spoken here at all — the caller falls
      through to the existing full-buffer degraded-check-and-retry path
      exactly as if streaming had never been attempted.
    - A running word count enforces the same backstop as the non-streaming
      path (reply_word_limit + MAX_QUESTION_WORDS), stopping at a sentence
      boundary instead of a raw word-count slice — so a long answer still
      ends on a complete thought, just decided incrementally instead of
      after the fact.

    filler_task, when given, is the in-flight _speak_chunk() call for the
    latency-masking filler line (see _generate_grounded_reply) — started
    concurrently with the RAG call, not before it. The first real sentence
    awaits it before dispatching, so the filler's own playback (~1-2s) never
    overlaps the real answer's audio even though the RAG round trip that
    produced this sentence ran the whole time the filler was talking.
    Awaiting an already-finished task is a no-op, so a slow filler vs. a
    fast RAG response (or the reverse) both resolve correctly without any
    extra bookkeeping here.

    Returns (callback, state) — state accumulates what was actually spoken
    (state["spoken_parts"]) and whether anything was dispatched
    (state["dispatched"]), which the caller uses to decide whether this
    attempt already committed to speaking or is still free to retry.
    """
    backstop_words = reply_word_limit + MAX_QUESTION_WORDS
    state: Dict[str, Any] = {
        "dispatched": False,
        "aborted": False,
        "seen_first": False,
        "cumulative_words": 0,
        "chunk_n": 0,
        "spoken_parts": [],
    }

    def _degraded_marker(text: str) -> bool:
        lowered = text.lower()
        return not text or lowered.startswith("error:") or any(marker in lowered for marker in (
            "could not retrieve", "no specific documentation", "an error occurred",
            "could not find relevant documents", "no relevant information found",
            "request timed out", "service unavailable", "failed to get response",
        ))

    async def on_sentence(raw_sentence: str) -> None:
        if state["aborted"]:
            return
        cleaned = re.sub(r"\[\d+\]", "", raw_sentence)
        cleaned = re.sub(r"[#*`_~]", "", cleaned)
        cleaned = re.sub(r"^\s*[-+•]\s+", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            return
        if not state["seen_first"]:
            state["seen_first"] = True
            if _degraded_marker(cleaned):
                # Don't speak a raw failure string — bail out entirely and
                # let the caller's existing full-buffer path (degraded
                # check, one retry) handle it exactly as before.
                state["aborted"] = True
                return
        if state["dispatched"] and (
            run.get("active_turn_id") != turn_id
            or run.get("state") not in (AgentState.THINKING, AgentState.SPEAKING)
        ):
            # Superseded, or barge-in released the floor without changing
            # active_turn_id (see _speak_chunk) — stop speaking further
            # chunks. The backend stream is still left to finish reading
            # quietly in the background so the connection isn't torn down
            # mid-response, but nothing more will be dispatched.
            state["aborted"] = True
            return
        sentence_words = len(cleaned.split())
        if state["cumulative_words"] and state["cumulative_words"] + sentence_words > backstop_words:
            state["aborted"] = True  # budget hit; stop, same as the non-streaming backstop
            return
        if filler_task is not None and not state["dispatched"]:
            # First real sentence: wait for the filler's own playback to
            # finish so the two never overlap. The RAG round trip that
            # produced this sentence already ran concurrently with that
            # wait -- this is not extra latency on top, just sequencing.
            filler_still_live = await filler_task
            if not filler_still_live:
                # Barge-in (or turn superseded) during the filler itself --
                # don't speak the real answer on top of an interruption
                # that already happened.
                state["aborted"] = True
                return
        state["chunk_n"] += 1
        chunk_id = f"{turn_id}-{state['chunk_n']}"
        state["dispatched"] = True
        state["cumulative_words"] += sentence_words
        state["spoken_parts"].append(cleaned)
        still_live = await _speak_chunk(run, turn_id, chunk_id, cleaned)
        if not still_live:
            state["aborted"] = True
        if state["cumulative_words"] >= backstop_words:
            state["aborted"] = True

    return on_sentence, state


def _finish_streamed_reply(run: Dict[str, Any], turn_id: int, spoken_text: str) -> None:
    """Bookkeeping for a reply already spoken chunk-by-chunk via _speak_chunk.

    Mirrors what _dispatch_reply does after a single-shot send — history,
    echo registration, governor, timing log — without re-sending audio.
    Unlike _dispatch_reply, this can't check governor.is_duplicate() before
    speaking (there's no full text to check until streaming is already
    underway), so within-window duplicate suppression doesn't apply to a
    streamed reply — a real, accepted gap, not an oversight.
    """
    bot_name = run.get("bot_name") or "Tom"
    spoken_at = time.monotonic()
    governor: SpeechGovernor = run["governor"]

    history = run.setdefault("history", [])
    history.append({"speaker": bot_name, "participant_id": "bot", "text": spoken_text})
    del history[:-40]
    run["echo"].note_bot_speech(spoken_text, spoken_at)
    governor.note_reply(spoken_text, spoken_at)
    _push_rep(run, {"type": "agent_spoke", "text": spoken_text, "turn_id": turn_id})

    timing = (run.get("turn_timing") or {}).pop(turn_id, None)
    if timing and timing.get("finalized_at") is not None:
        logger.info(
            "[TIMING] turn_id=%s streamed reply complete, total=%.2fs (turn_finalize->last_chunk_spoken)",
            turn_id, spoken_at - timing["finalized_at"],
        )
    _release_floor(run, spoken_at, reply_text=spoken_text)


def _release_floor(run: Dict[str, Any], now: Optional[float] = None, reply_text: str = "") -> None:
    """Return the agent to LISTENING and open the follow-up window.

    reply_text, when given, sizes the follow-up window to how long that reply
    actually took to say (see AGENT_FOLLOWUP_WINDOW_REPLY_SCALE) — omitted on
    release paths where nothing was actually spoken (failures, duplicates,
    watchdog timeouts), which correctly leaves the window at its flat floor.
    """
    run["state"] = AgentState.LISTENING
    floor: FloorState = run["floor"]
    floor.last_bot_finished_at = now if now is not None else time.monotonic()
    if reply_text:
        floor.last_reply_seconds = estimate_speech_seconds(reply_text)
    _push_rep(run, {"type": "agent_state", "state": AgentState.LISTENING.value})


async def _interrupt_avatar(run: Dict[str, Any], participant_id: str) -> None:
    """Stop an in-flight turn. Playback and pending inference are both cancelled."""
    state = run.get("state")
    if state not in (AgentState.SPEAKING, AgentState.THINKING):
        return

    if state == AgentState.THINKING:
        # The answer is not spoken yet and is already stale; drop it silently
        # rather than delivering a reply to a question the room moved past.
        pending = run.get("active_response_task")
        if pending and not pending.done():
            pending.cancel()
        logger.info("[Barge-In] Cancelled pending turn participant_id=%s", participant_id)
        _release_floor(run)
        return

    control_ws = run.get("control_ws")
    if not control_ws:
        return
    run["state"] = AgentState.INTERRUPTING
    logger.info("[Barge-In] participant_id=%s sustained_ms=%d", participant_id, AGENT_BARGE_IN_MS)
    await control_ws.send_json({
        "type": "avatar_interrupt",
        "turn_id": run.get("active_turn_id"),
    })


async def _speak_watchdog(run: Dict[str, Any], turn_id: int, text: str) -> None:
    """Force LISTENING if the avatar never reports speak_started/speak_ended.

    Without this a single dropped LiveKit data event wedges the run in SPEAKING
    forever, after which every reply is suppressed and only barge-in ever fires.
    """
    try:
        await asyncio.sleep(AGENT_SPEAK_START_TIMEOUT_S)
        if run.get("active_turn_id") != turn_id:
            return
        if run.get("state") == AgentState.THINKING:
            logger.warning("[Watchdog] No speak_started for turn_id=%s; releasing floor", turn_id)
            _release_floor(run)
            return

        remaining = estimate_speech_seconds(text) + AGENT_SPEAK_MAX_OVERRUN_S
        await asyncio.sleep(remaining)
        if run.get("active_turn_id") != turn_id:
            return
        if run.get("state") != AgentState.LISTENING:
            logger.warning(
                "[Watchdog] turn_id=%s stuck in %s after %.1fs; releasing floor",
                turn_id,
                run.get("state"),
                AGENT_SPEAK_START_TIMEOUT_S + remaining,
            )
            _release_floor(run)
    except asyncio.CancelledError:
        pass
