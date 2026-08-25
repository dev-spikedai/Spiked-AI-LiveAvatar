"""Replay a scripted multi-participant meeting through Tom's live pipeline.

The transcript, Gemini, RAG, turn gate, autospeak judge, and interruption state
remain real. Only the external speech/video transport is simulated. Scenario
steps can wait for Tom's filler or answer, assert text, and barge in while Tom
is speaking.

Usage:
    python scripts/replay_meeting.py examples/meeting_scenario.json
    python scripts/replay_meeting.py examples/meeting_scenario.json --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import live_avatar  # noqa: E402
from src.core import floor as floor_module  # noqa: E402
from src.agent_policy import AgentState, EchoSuppressor, FloorState, SpeechGovernor  # noqa: E402


class ReplayFailure(RuntimeError):
    pass


def make_run(config: Dict[str, Any]) -> Dict[str, Any]:
    settings = config.get("settings") or {}
    user_context = dict(config.get("user_context") or {})
    user_context.setdefault("company_name", "SpikedAI")
    user_context.setdefault("bot_name", settings.get("bot_name", "Tom"))
    return {
        "run_id": "replay-test", "bot_name": settings.get("bot_name", "Tom"),
        "state": AgentState.LISTENING, "history": [], "live_turns": {},
        "pending_turns": {}, "background_tasks": set(), "turn_timing": {},
        "turn_counter": 0, "active_turn_id": None, "active_response_task": None,
        "watchdog_task": None, "control_ws": None, "rep_sockets": set(),
        "pending_insight": None, "proactive_prefetches": {}, "last_insight_at": None,
        "last_autospeak_at": None, "autospeak_count": 0,
        "autospeak_enabled": bool(settings.get("autospeak_enabled", True)),
        "meeting_preferences": dict(settings.get("meeting_preferences") or {}),
        "role_profile": settings.get("role_profile", "solution architect"),
        "taught_facts": [], "persistent_memory": list(settings.get("persistent_memory") or []),
        "token": settings.get("token") or os.getenv("SUPABASE_ACCESS_TOKEN", ""),
        "client_id": settings.get("client_id"), "source_ids": settings.get("source_ids") or [],
        "user_context": user_context, "intel": None, "floor": FloorState(),
        "echo": EchoSuppressor(similarity_threshold=0.9, tail_seconds=8),
        "governor": SpeechGovernor(
            cooldown_seconds=max(0.1, 2.0 * float(settings.get("speech_scale", 1.0))),
            max_replies_per_window=4,
            window_seconds=max(1.0, 30.0 * float(settings.get("speech_scale", 1.0))),
        ),
    }


def _event(events: List[Dict[str, Any]], kind: str, **fields: Any) -> None:
    events.append({"kind": kind, "at": time.monotonic(), **fields})


class ReplayControlWS:
    def __init__(self, run: Dict[str, Any], events: List[Dict[str, Any]]) -> None:
        self.run, self.events = run, events

    async def send_json(self, message: Dict[str, Any]) -> None:
        _event(self.events, "avatar_control", message=message)
        if message.get("type") == "avatar_interrupt":
            self.run["state"] = AgentState.LISTENING
            _event(self.events, "tom_interrupted", turn_id=message.get("turn_id"))


from contextlib import asynccontextmanager

def install_recorder(run: Dict[str, Any], events: List[Dict[str, Any]], speech_scale: float) -> None:
    """Replace only output transport, while retaining reasoning and RAG."""
    run["control_ws"] = ReplayControlWS(run, events)

    async def record_chunk(target_run: Dict[str, Any], turn_id: int, chunk_id: str, text: str) -> bool:
        if target_run.get("active_turn_id") != turn_id or target_run.get("state") not in (AgentState.THINKING, AgentState.SPEAKING):
            return False
        target_run["state"] = AgentState.SPEAKING
        phase = "filler" if chunk_id.endswith("-filler") else "answer"
        _event(events, "tom_speech", turn_id=turn_id, chunk_id=chunk_id, phase=phase, text=text)
        if phase == "answer":
            _event(events, "tom_answer_chunk", turn_id=turn_id, source="addressed", text=text)
        words = max(1, len(text.split()))
        await asyncio.sleep(max(0.05, words / 3.0 * speech_scale))
        return target_run.get("active_turn_id") == turn_id and target_run.get("state") in (AgentState.THINKING, AgentState.SPEAKING)

    async def record_dispatch(target_run: Dict[str, Any], answer: str, turn_id: int, source: str = "addressed") -> bool:
        _event(events, "tom_answer", turn_id=turn_id, source=source, text=answer)
        target_run.setdefault("history", []).append({"speaker": target_run.get("bot_name", "Tom"), "participant_id": "bot", "text": answer})
        target_run["echo"].note_bot_speech(answer, time.monotonic())
        # Back-date the governor timestamp by the full cooldown so the next
        # turn sees the governor as ready immediately.  In a real call the
        # spoken audio takes `cooldown_seconds` of wall time; in replay the
        # speech_scale already compressed that wait, so we compensate here.
        governor: SpeechGovernor = target_run["governor"]
        backdated = time.monotonic() - governor.cooldown_seconds
        governor.note_reply(answer, backdated)
        live_avatar._release_floor(target_run, reply_text=answer)
        return True

    async def record_autospeak(target_run: Dict[str, Any], question: str, speaker: str, **kwargs: Any) -> Dict[str, Any]:
        target_run["turn_counter"] = int(target_run.get("turn_counter", 0)) + 1
        turn_id = target_run["turn_counter"]
        text = kwargs.get("warm_reply") or ""
        _event(events, "tom_interjection", turn_id=turn_id, source="autonomous", speaker_context=speaker, question=question, text=text)
        target_run.setdefault("history", []).append({"speaker": target_run.get("bot_name", "Tom"), "participant_id": "bot", "text": text})
        live_avatar._release_floor(target_run, reply_text=text)
        return {"accepted": True, "turn_id": turn_id, "warm": True}

    live_avatar._speak_chunk = record_chunk
    floor_module._speak_chunk = record_chunk
    live_avatar._dispatch_reply = record_dispatch
    live_avatar._take_floor_and_speak = record_autospeak


@asynccontextmanager
async def patch_recorder(run: Dict[str, Any], events: List[Dict[str, Any]], speech_scale: float):
    """Context manager that installs the replay recorder and restores original module functions on exit."""
    orig_live_speak = live_avatar._speak_chunk
    orig_floor_speak = getattr(floor_module, "_speak_chunk", None)
    orig_dispatch = live_avatar._dispatch_reply
    orig_take_floor = live_avatar._take_floor_and_speak
    try:
        install_recorder(run, events, speech_scale)
        yield
    finally:
        live_avatar._speak_chunk = orig_live_speak
        if orig_floor_speak is not None:
            floor_module._speak_chunk = orig_floor_speak
        live_avatar._dispatch_reply = orig_dispatch
        live_avatar._take_floor_and_speak = orig_take_floor


def install_offline_brain() -> None:
    """Use deterministic local speech chunks without calling Gemini or RAG."""

    async def fake_process(**kwargs: Any) -> str:
        run = kwargs["run"]
        turn_id = kwargs["turn_id"]
        answer = f"I’ll check the docs and answer this: {kwargs['transcript']}"
        await live_avatar._speak_chunk(run, turn_id, f"{turn_id}-filler", "I’ll check the docs first.")
        if run.get("state") not in (AgentState.THINKING, AgentState.SPEAKING):
            return ""
        await live_avatar._speak_chunk(run, turn_id, f"{turn_id}-1", answer)
        if run.get("state") in (AgentState.THINKING, AgentState.SPEAKING):
            live_avatar._finish_streamed_reply(run, turn_id, answer)
        return answer

    async def fake_grounded_reply(**kwargs: Any) -> str:
        return f"A practical answer is to reduce implementation risk in stages: {kwargs.get('transcript', '')}"

    live_avatar.process_transcript_with_gemini = fake_process
    live_avatar._generate_grounded_reply = fake_grounded_reply


@asynccontextmanager
async def patch_offline_brain():
    """Context manager that installs the offline brain and restores original handlers on exit."""
    orig_process = live_avatar.process_transcript_with_gemini
    orig_grounded = live_avatar._generate_grounded_reply
    try:
        install_offline_brain()
        yield
    finally:
        live_avatar.process_transcript_with_gemini = orig_process
        live_avatar._generate_grounded_reply = orig_grounded


def normalize_steps(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Accept the original flat `turns` fixture as a backwards-compatible form."""
    if config.get("steps"):
        return list(config["steps"])
    return [{"turn": {"participant_id": t["participant_id"], "speaker": t.get("speaker"), "say": t["text"], "interim": t.get("interim")}} for t in config.get("turns", [])]


async def settle(run: Dict[str, Any], extra_s: float = 0.05, wait_for_answer: bool = False) -> None:
    await asyncio.sleep(live_avatar.AGENT_TURN_MERGE_MS / 1000 + extra_s)
    if not wait_for_answer:
        return
    for _ in range(120):
        tasks = [task for task in run.get("background_tasks", set()) if not task.done()]
        active = run.get("active_response_task")
        if active and not active.done():
            tasks.append(active)
        if not tasks:
            return
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.sleep(0)


def matches(event: Dict[str, Any], expectation: Dict[str, Any]) -> bool:
    if expectation.get("kind") and event.get("kind") != expectation["kind"]:
        return False
    if expectation.get("phase") and event.get("phase") != expectation["phase"]:
        return False
    if expectation.get("source") and event.get("source") != expectation["source"]:
        return False
    text = str(event.get("text") or "").casefold()
    contains = expectation.get("contains", expectation.get("tom_says"))
    if isinstance(contains, dict):
        contains = contains.get("contains")
    if contains:
        needles = contains if isinstance(contains, list) else [contains]
        if not all(str(needle).casefold() in text for needle in needles):
            return False
    not_contains = expectation.get("not_contains")
    if not_contains and any(str(needle).casefold() in text for needle in (not_contains if isinstance(not_contains, list) else [not_contains])):
        return False
    return event.get("kind", "").startswith("tom_")


async def wait_for(events: List[Dict[str, Any]], expectation: Dict[str, Any], seen: int) -> int:
    timeout = float(expectation.get("timeout_s", expectation.get("timeout", 30)))
    deadline, cursor = time.monotonic() + timeout, seen
    while time.monotonic() < deadline:
        while cursor < len(events):
            if matches(events[cursor], expectation):
                return cursor + 1
            cursor += 1
        await asyncio.sleep(0.05)
    raise ReplayFailure(f"Timed out after {timeout:.1f}s waiting for {expectation!r}")


async def do_turn(run: Dict[str, Any], events: List[Dict[str, Any]], turn: Dict[str, Any], auto_barge_in: bool) -> None:
    participant_id, speaker = str(turn["participant_id"]), turn.get("speaker") or str(turn["participant_id"])
    text = str(turn.get("say", turn.get("text", ""))).strip()
    if auto_barge_in and run.get("state") in (AgentState.THINKING, AgentState.SPEAKING):
        await live_avatar._interrupt_avatar(run, participant_id)
    _event(events, "participant_turn", participant_id=participant_id, speaker=speaker, text=text)
    live_avatar._observe_interim_transcript(run, participant_id, speaker, turn.get("interim") or text, False)
    live_avatar._ingest_utterance(run, participant_id, speaker, {"text": text, "words": []})
    await settle(run)


async def _run_replay_steps(
    run: Dict[str, Any], events: List[Dict[str, Any]], config: Dict[str, Any], settings: Dict[str, Any]
) -> List[Dict[str, Any]]:
    auto_barge_in, started, seen = bool(settings.get("auto_barge_in", True)), time.monotonic(), 0
    for index, step in enumerate(normalize_steps(config), start=1):
        if "sleep" in step:
            await asyncio.sleep(float(step["sleep"]))
        if "turn" in step:
            await do_turn(run, events, step["turn"], auto_barge_in)
        if "barge_in" in step:
            action = step["barge_in"]
            await asyncio.sleep(float(action.get("after_s", 0)))
            participant_id = str(action.get("participant_id", "barge-in"))
            await live_avatar._interrupt_avatar(run, participant_id)
            if action.get("say"):
                await do_turn(run, events, {"participant_id": participant_id, "speaker": action.get("speaker", participant_id), "say": action["say"]}, False)
        for expectation in step.get("wait_for", []) + step.get("expect", []):
            seen = await wait_for(events, expectation, seen)
        if step.get("wait_for_idle"):
            await settle(run, extra_s=0.1, wait_for_answer=True)
        print(f"step {index}: ok")
    await settle(run, extra_s=0.25, wait_for_answer=True)
    for task in list(run.get("background_tasks", set())):
        if not task.done():
            task.cancel()
    events.sort(key=lambda item: item["at"])
    elapsed = time.monotonic() - started
    for event in events:
        event["elapsed_s"], event["at"] = round(event["at"] - started, 3), None
        event.pop("at", None)
    print(f"Replay complete: {len(normalize_steps(config))} steps in {elapsed:.2f}s")
    return events


async def replay(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    run, events = make_run(config), []
    settings = config.get("settings") or {}
    speech_scale = float(settings.get("speech_scale", 1.0))
    async with patch_recorder(run, events, speech_scale):
        if config.get("offline"):
            async with patch_offline_brain():
                return await _run_replay_steps(run, events, config, settings)
        else:
            return await _run_replay_steps(run, events, config, settings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay_file", type=Path)
    parser.add_argument("--json", action="store_true", help="print machine-readable event output")
    parser.add_argument("--offline", action="store_true", help="run deterministic local brain; never call Gemini or RAG")
    args = parser.parse_args()
    try:
        config = json.loads(args.replay_file.read_text(encoding="utf-8"))
        config["offline"] = config.get("offline", args.offline)
        events = asyncio.run(replay(config))
    except ReplayFailure as exc:
        print(f"REPLAY FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    if args.json:
        print(json.dumps(events, indent=2, ensure_ascii=False))
    else:
        for event in events:
            if event["kind"] == "participant_turn":
                print(f"[{event['elapsed_s']:>6.2f}s] {event['speaker']}: {event['text']}")
            elif event["kind"].startswith("tom_"):
                print(f"[{event['elapsed_s']:>6.2f}s] {event['kind'].upper()}: {event.get('text', '')}")


if __name__ == "__main__":
    main()
