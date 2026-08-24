import asyncio
from types import SimpleNamespace

from src import live_avatar
from src.agent_policy import AgentState, EchoSuppressor, FloorState, SpeechGovernor


def test_addressed_turn_flows_from_interim_wake_to_contextual_dispatch(monkeypatch):
    observed = {}
    dispatched = []

    async def fake_process(**kwargs):
        observed.update(kwargs)
        return "Yes — given the implementation-cost concern we discussed, that is the right direction."

    async def fake_dispatch(run, answer, turn_id, source="addressed"):
        dispatched.append((answer, turn_id, source))

    monkeypatch.setattr(live_avatar, "process_transcript_with_gemini", fake_process)
    monkeypatch.setattr(live_avatar, "_dispatch_reply", fake_dispatch)

    run = {
        "run_id": "flow-test",
        "bot_name": "Tom",
        "state": AgentState.LISTENING,
        "history": [{"speaker": "Client", "participant_id": "client", "text": "We are worried about implementation cost."}],
        "live_turns": {},
        "pending_turns": {},
        "meeting_preferences": {},
        "role_profile": "solution architect",
        "taught_facts": [],
        "echo": EchoSuppressor(),
        "floor": FloorState(),
        "governor": SpeechGovernor(),
        "user_context": {"company_name": "SpikedAI", "bot_name": "Tom"},
        "token": "token",
        "client_id": "client-1",
        "active_response_task": None,
        "watchdog_task": None,
    }

    async def drive():
        live_avatar._observe_interim_transcript(
            run, "client", "Client", "Tom, did you catch", False
        )
        assert run["live_turns"]["client"]["wake_candidate"] is True

        live_avatar._ingest_utterance(
            run,
            "client",
            "Client",
            {"text": "Tom, did you catch that?", "words": []},
        )
        await asyncio.sleep(0.35)
        task = run.get("active_response_task")
        if task:
            await task

    asyncio.run(drive())

    assert observed["transcript"] == "Tom, did you catch that?"
    assert observed["conversation_history"][-1]["text"] == "We are worried about implementation cost."
    assert dispatched and dispatched[0][0].startswith("Yes")
    assert run["history"][-1]["text"] == "Tom, did you catch that?"
