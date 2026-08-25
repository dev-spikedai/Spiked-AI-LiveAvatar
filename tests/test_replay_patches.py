import asyncio
from pathlib import Path

from scripts import replay_meeting
from src import live_avatar
from src.core import floor as floor_module


def test_replay_restores_monkeypatches():
    orig_speak = live_avatar._speak_chunk
    orig_floor_speak = floor_module._speak_chunk
    orig_dispatch = live_avatar._dispatch_reply
    orig_autospeak = live_avatar._take_floor_and_speak
    orig_gemini = live_avatar.process_transcript_with_gemini
    orig_grounded = live_avatar._generate_grounded_reply

    scenario = {
        "offline": True,
        "settings": {
            "bot_name": "Tom",
            "speech_scale": 0.01,
        },
        "steps": [
            {
                "turn": {
                    "participant_id": "rep",
                    "speaker": "Sai",
                    "say": "Tom, quick check",
                },
                "wait_for": [
                    {"kind": "tom_answer", "timeout_s": 5},
                ],
            }
        ],
    }

    events = asyncio.run(replay_meeting.replay(scenario))

    # Check monkeypatches restored
    assert live_avatar._speak_chunk is orig_speak
    assert floor_module._speak_chunk is orig_floor_speak
    assert live_avatar._dispatch_reply is orig_dispatch
    assert live_avatar._take_floor_and_speak is orig_autospeak
    assert live_avatar.process_transcript_with_gemini is orig_gemini
    assert live_avatar._generate_grounded_reply is orig_grounded

    # Check that tom_answer is emitted only once for the completed turn
    tom_answers = [e for e in events if e.get("kind") == "tom_answer"]
    assert len(tom_answers) == 1
    tom_chunks = [e for e in events if e.get("kind") == "tom_answer_chunk"]
    assert len(tom_chunks) >= 1
