"""prompts/avatar_prompt.md is the only place the avatar's persona is written.

These tests exist to keep it that way: the persona previously lived in three
inline copies that drifted, and a fourth consumer (the vendor system prompt)
was empty.
"""

from pathlib import Path

import pytest

from src.core import persona

SRC = Path("src")
# Phrases that identify the persona rather than merely mention the bot.
PERSONA_MARKERS = [
    "Solution Architect",
    "you never pitch",
    "Never invent facts",
]


def test_every_block_renders_with_the_documented_placeholders():
    values = dict(bot_name="Tom", company_name="SpikedAI", reply_word_limit=45, max_question_words=12)
    for name in persona.BLOCKS:
        rendered = persona.block(name, **values)
        assert rendered.strip(), f"{name} is empty"
        assert "{" not in rendered, f"{name} left an unfilled placeholder: {rendered[:80]}"
        assert "Tom" in rendered or "45" in rendered, f"{name} interpolated nothing"


def test_editor_notes_never_reach_a_model():
    for name, body in persona.BLOCKS.items():
        assert not body.lstrip().startswith(">"), f"{name} kept a blockquote note"
        assert "\n>" not in body, f"{name} kept a blockquote note"


def test_unknown_placeholder_is_rejected_at_load():
    """A typo'd placeholder must fail on load, not ship a literal brace to a model."""
    with pytest.raises(ValueError, match="unknown placeholder"):
        persona.load("### broken\n\nHello {not_a_real_placeholder}\n")


def test_a_valid_document_loads():
    blocks = persona.load("### greeting\n\n> a note\nHi {bot_name}, from {company_name}.\n")
    assert blocks == {"greeting": "Hi {bot_name}, from {company_name}."}


def test_missing_block_fails_loudly():
    with pytest.raises(KeyError, match="no block"):
        persona.block("does_not_exist", bot_name="Tom")


def test_the_persona_is_not_duplicated_anywhere_in_the_source():
    """The whole point of Priority 5. If this fails, someone has started a
    fifth copy of the persona instead of editing the prompt document."""
    offenders = []
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for marker in PERSONA_MARKERS:
            if marker in text:
                offenders.append(f"{path}: {marker!r}")
    assert not offenders, (
        "persona text found outside prompts/avatar_prompt.md:\n  " + "\n  ".join(offenders)
    )


def test_all_four_consumers_are_covered():
    """Each prompt the agent sends must have a block backing it."""
    for name in ("identity", "retrieval_hint", "interjection_framing",
                 "interjection_bar", "vendor_system_prompt"):
        assert name in persona.BLOCKS, f"{name} is missing from avatar_prompt.md"


def test_vendor_prompt_states_its_own_word_budget():
    """Delegated mode never reaches compose_reply, so the budget can only be
    stated in the prompt -- if it is dropped, native replies run unbounded."""
    rendered = persona.block(
        "vendor_system_prompt", bot_name="Tom", company_name="SpikedAI", reply_word_limit=45
    )
    assert "45 words" in rendered


def test_the_prompt_document_ships_in_the_container():
    """persona.py reads it at import, so a missing COPY is not a degraded
    persona -- it is a service that cannot start."""
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    assert "COPY ./prompts" in dockerfile
