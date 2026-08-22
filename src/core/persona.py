"""Loads prompts/avatar_prompt.md, the single source for the avatar's persona."""

import re
from pathlib import Path
from string import Formatter
from typing import Dict

PROMPT_FILE = Path(__file__).resolve().parent.parent.parent / "prompts" / "avatar_prompt.md"

ALLOWED_PLACEHOLDERS = {"bot_name", "company_name", "reply_word_limit", "max_question_words"}

_HEADING = re.compile(r"^### (?P<name>[a-z_]+)\s*$", re.MULTILINE)


def _parse(text: str) -> Dict[str, str]:
    blocks: Dict[str, str] = {}
    matches = list(_HEADING.finditer(text))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.end():end].strip("\n")
        blocks[match.group("name")] = body.strip()
    return blocks


def _strip_editor_notes(body: str) -> str:
    """Drop `>` blockquote lines: notes for editors, never sent to a model."""
    kept = [line for line in body.split("\n") if not line.lstrip().startswith(">")]
    return "\n".join(kept).strip()


def validate(blocks: Dict[str, str]) -> Dict[str, str]:
    """Reject unknown placeholders at load, not as a literal brace at runtime."""
    for name, body in blocks.items():
        used = {f for _, f, _, _ in Formatter().parse(body) if f}
        unknown = used - ALLOWED_PLACEHOLDERS
        if unknown:
            raise ValueError(
                f"avatar_prompt.md block '{name}' uses unknown placeholder(s): {sorted(unknown)}"
            )
    return blocks


def load(text: str) -> Dict[str, str]:
    return validate({name: _strip_editor_notes(body) for name, body in _parse(text).items()})


BLOCKS: Dict[str, str] = load(PROMPT_FILE.read_text(encoding="utf-8"))


def block(name: str, **values: object) -> str:
    """Render one named block from the prompt document."""
    if name not in BLOCKS:
        raise KeyError(f"avatar_prompt.md has no block '{name}' (have: {sorted(BLOCKS)})")
    return BLOCKS[name].format(**values)
