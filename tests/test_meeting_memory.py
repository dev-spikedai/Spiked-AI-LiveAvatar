import asyncio
from types import SimpleNamespace

from src import live_avatar
from src.live_avatar import (
    detect_role_instruction,
    detect_speak_up_preference,
    format_meeting_instructions,
    proactive_speaking_enabled,
    remember_meeting_instruction,
)
from src.supabase_client import _is_active_memory_row


def test_explicit_speak_up_preference_is_detected():
    assert detect_speak_up_preference(
        "I prefer you to always speak up when you can help"
    ) is True


def test_explicit_wait_preference_is_detected():
    assert detect_speak_up_preference(
        "Please only speak when asked"
    ) is False


def test_ordinary_meeting_statement_does_not_change_preference():
    assert detect_speak_up_preference(
        "We should speak up about implementation cost"
    ) is None


def test_role_instruction_is_scoped_to_the_meeting():
    assert detect_role_instruction(
        "Tom, as a solution architect, how would you design this?"
    ) == "solution architect"


def test_memory_instruction_changes_proactive_policy_and_prompt_context():
    run = {
        "run_id": "run_test",
        "autospeak_enabled": False,
        "meeting_preferences": {},
        "role_profile": "",
    }
    assert not proactive_speaking_enabled(run)

    remember_meeting_instruction(
        run,
        "Always speak up when you can help",
        speak_up=True,
        role_profile="solution architect",
    )

    assert proactive_speaking_enabled(run)
    rendered = format_meeting_instructions(run)
    assert "speak up" in rendered
    assert "solution architect" in rendered


def test_expired_persistent_memory_is_not_hydrated():
    assert _is_active_memory_row({"expires_at": "2000-01-01T00:00:00Z"}) is False
    assert _is_active_memory_row({"expires_at": None}) is True


def test_first_turn_waits_for_startup_memory_before_classification(monkeypatch):
    captured = []

    class Models:
        async def generate_content(self, **kwargs):
            captured.append(kwargs)
            return SimpleNamespace(
                parsed=live_avatar.TurnAnalysisAndReply(
                    response_action="acknowledge",
                    intent="social",
                    resolved_query="",
                    corrections=[],
                )
            )

    monkeypatch.setattr(
        live_avatar,
        "gemini_client",
        SimpleNamespace(aio=SimpleNamespace(models=Models())),
    )

    async def scenario():
        run = {
            "active_mcp_context": None,
            "meeting_preferences": {},
            "role_profile": "",
            "persistent_memory": [],
            "user_context": {"company_name": "SpikedAI", "bot_name": "Tom"},
        }

        async def hydrate():
            await asyncio.sleep(0.01)
            run["meeting_preferences"]["speak_up_when_helpful"] = True
            run["role_profile"] = "solution architect"
            run["persistent_memory"] = [{
                "memory_type": "preference",
                "memory_key": "speak_up_when_helpful",
                "memory_value": True,
            }]

        run["memory_task"] = asyncio.create_task(hydrate())
        answer = await live_avatar.process_transcript_with_gemini(
            transcript="Tom, are you there?",
            speaker="Client",
            conversation_history=[],
            auth_token="token",
            user_context=run["user_context"],
            run=run,
        )
        return answer

    assert asyncio.run(scenario()) == "Understood."
    prompt = str(captured[0]["contents"])
    assert "speak up" in prompt
    assert "solution architect" in prompt
