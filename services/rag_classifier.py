"""
RAG answer classifier.

Runs after the Gemini cognitive answer is generated. Uses a two-stage filter to
detect low-quality RAG outputs and persist them to `meeting_questions` for later
review:

  Stage 1 — regex prefilter: catches obvious "I don't know" answers cheaply.
  Stage 2 — Gemini judge: for answers that pass regex, classify into
            GROUNDED / NO_INFO / PARTIAL / HALLUCINATED via a small LLM call.

Only rows with a non-GROUNDED verdict are inserted. GROUNDED answers are dropped
so the table stays focused on failures worth analyzing.

The classifier is fire-and-forget from the caller's perspective — errors are
logged but never raised. If the judge itself errors or times out, we still log
the row with verdict='UNCLASSIFIED' so we don't silently lose data.
"""

import asyncio
import logging
import re
from datetime import datetime
from typing import Optional

from core.config import get_g_vars
from services.ai_helpers import call_gemini_flash_llm, parse_json_from_llm_response

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NO_INFO_SENTINEL = "NO_INFO_FROM_RAG"

# Verdicts we persist. GROUNDED is intentionally not in this set — we drop
# grounded answers so the table only contains rows worth reviewing.
PERSISTED_VERDICTS = {"NO_INFO", "PARTIAL", "HALLUCINATED", "UNCLASSIFIED"}

# Judge timeout — keep tight so a slow Gemini call can't back up the task queue.
GEMINI_JUDGE_TIMEOUT_SECONDS = 5.0

# Truncate context passed to the judge to keep the prompt small and cheap.
JUDGE_CONTEXT_CHAR_LIMIT = 3000


# ---------------------------------------------------------------------------
# Stage 1 — regex prefilter
# ---------------------------------------------------------------------------

# Applied case-insensitively to the stripped answer. A hit means the answer
# clearly said "no info" and we can skip the Gemini judge entirely.
_NO_INFO_PATTERNS = [
    r"i (don't|do not|cannot|can'?t|couldn'?t) (have|find|see|answer|provide)",
    r"(the )?(documents?|context|sources?|provided (info|material|context)) (don't|do not|does not) (contain|mention|include|have|provide)",
    r"no (relevant )?(information|data|details|answer|context) (available|found|provided|in the|to answer)",
    r"unable to (answer|find|determine|provide|locate)",
    r"not enough (context|information|data|detail)",
    r"based on the (provided|available) (context|documents?|sources?),? i (can'?t|cannot|don'?t|do not)",
    r"insufficient (information|context|data)",
    r"^i could not find relevant documents\.?$",   # existing hardcoded fallback
    r"^no relevant information found\.?$",         # existing hardcoded fallback
    r"the (provided )?(context|information) (is |does )?not (enough|sufficient|contain)",
]

_COMPILED_NO_INFO = [re.compile(p, re.IGNORECASE) for p in _NO_INFO_PATTERNS]


def classify_with_regex(answer: str) -> Optional[str]:
    """Return 'NO_INFO' if any pattern hits, else None.

    Cheap synchronous check — call before the judge.
    """
    if not answer:
        return "NO_INFO"

    text = answer.strip()
    for pattern in _COMPILED_NO_INFO:
        if pattern.search(text):
            return "NO_INFO"
    return None


# ---------------------------------------------------------------------------
# Stage 2 — Gemini judge
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_PROMPT = """You are a strict evaluator of RAG (retrieval-augmented generation) answers.
Given a QUESTION, the CONTEXT that was provided to a model, and the model's ANSWER,
classify the answer into exactly one of these verdicts:

- GROUNDED: The answer is fully supported by the context.
- NO_INFO: The model correctly refused because the context lacked the info.
- PARTIAL: The model gave an answer, but the context only partially supported it.
- HALLUCINATED: The model made claims that the context does not support.

Respond with ONLY a JSON object, no prose, no markdown fences:
{"verdict": "<one of the four>", "confidence": <float 0.0-1.0>, "reason": "<one short sentence>"}
"""


async def classify_with_gemini(
    question: str,
    context_text: str,
    answer: str,
) -> dict:
    """Call Gemini Flash to classify the answer. Returns a dict with
    verdict, confidence, reason. On any error or timeout, returns
    verdict='UNCLASSIFIED' with reason describing the failure.
    """
    truncated_context = (context_text or "")[:JUDGE_CONTEXT_CHAR_LIMIT]

    user_prompt = (
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{truncated_context}\n\n"
        f"ANSWER:\n{answer}"
    )

    try:
        raw = await asyncio.wait_for(
            call_gemini_flash_llm(prompt=user_prompt, system_prompt=_JUDGE_SYSTEM_PROMPT),
            timeout=GEMINI_JUDGE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning("RAG judge timed out after %ss", GEMINI_JUDGE_TIMEOUT_SECONDS)
        return {"verdict": "UNCLASSIFIED", "confidence": 0.0, "reason": "judge_timeout"}
    except Exception as e:
        logger.warning("RAG judge call failed: %s", e)
        return {"verdict": "UNCLASSIFIED", "confidence": 0.0, "reason": f"judge_error: {type(e).__name__}"}

    try:
        parsed = parse_json_from_llm_response(raw)
        verdict = str(parsed.get("verdict", "")).upper().strip()
        if verdict not in {"GROUNDED", "NO_INFO", "PARTIAL", "HALLUCINATED"}:
            logger.warning("RAG judge returned unknown verdict: %r", verdict)
            return {"verdict": "UNCLASSIFIED", "confidence": 0.0, "reason": f"unknown_verdict:{verdict}"}

        confidence = float(parsed.get("confidence", 0.0))
        reason = str(parsed.get("reason", ""))[:500]
        return {"verdict": verdict, "confidence": confidence, "reason": reason}

    except Exception as e:
        logger.warning("Failed to parse RAG judge response: %s | raw=%r", e, raw[:200])
        return {"verdict": "UNCLASSIFIED", "confidence": 0.0, "reason": "parse_error"}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _answer_for_verdict(verdict: str, original_answer: str) -> str:
    """NO_INFO gets a sentinel (the actual text is useless prose).
    PARTIAL / HALLUCINATED / UNCLASSIFIED get the real answer text so
    it can be reviewed.
    """
    if verdict == "NO_INFO":
        return NO_INFO_SENTINEL
    return original_answer


async def _insert_row(
    *,
    question: str,
    answer_text: str,
    verdict: str,
    endpoint: str,
    client_id: Optional[str],
    meeting_log_id: Optional[str],
) -> None:
    """Insert one row into meeting_questions. Best-effort; swallows errors."""
    g_vars = get_g_vars()
    supabase = g_vars.get("supabase")
    if supabase is None:
        logger.error("Supabase client not initialized; skipping rag_classifier insert")
        return

    payload = {
        "question": question,
        "answer_transcript": answer_text,
        "rag_verdict": verdict,
        "endpoint": endpoint,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
    }
    if client_id:
        payload["client_id"] = client_id
    if meeting_log_id:
        payload["meeting_log_id"] = meeting_log_id

    try:
        await asyncio.to_thread(
            lambda: supabase.table("meeting_questions").insert(payload).execute()
        )
        logger.info(
            "rag_classifier logged row: endpoint=%s verdict=%s client_id=%s",
            endpoint, verdict, client_id,
        )
    except Exception as e:
        logger.error("rag_classifier insert failed: %s", e, exc_info=True)


# ---------------------------------------------------------------------------
# Orchestrator — the one function callers use
# ---------------------------------------------------------------------------

async def classify_and_log(
    *,
    question: str,
    context_text: str,
    answer: str,
    endpoint: str,
    client_id: Optional[str],
    meeting_log_id: Optional[str] = None,
) -> None:
    """Classify a RAG answer and persist it to meeting_questions if it's not
    GROUNDED. Safe to fire-and-forget via asyncio.create_task — never raises.

    Args:
        question: the user's question
        context_text: the RAG context that was fed to the model
        answer: the final Gemini cognitive answer being classified
        endpoint: 'regular' or 'handsfree'
        client_id: caller's client_id (may be None)
        meeting_log_id: active meeting id if any (may be None)
    """
    try:
        if not answer or not answer.strip():
            # Empty answer — treat as NO_INFO without spending a judge call.
            await _insert_row(
                question=question,
                answer_text=NO_INFO_SENTINEL,
                verdict="NO_INFO",
                endpoint=endpoint,
                client_id=client_id,
                meeting_log_id=meeting_log_id,
            )
            return

        # Stage 1 — regex prefilter.
        regex_verdict = classify_with_regex(answer)
        if regex_verdict == "NO_INFO":
            await _insert_row(
                question=question,
                answer_text=NO_INFO_SENTINEL,
                verdict="NO_INFO",
                endpoint=endpoint,
                client_id=client_id,
                meeting_log_id=meeting_log_id,
            )
            return

        # Stage 2 — Gemini judge.
        result = await classify_with_gemini(question, context_text, answer)
        verdict = result["verdict"]

        # GROUNDED = drop, don't insert.
        if verdict == "GROUNDED":
            return

        if verdict not in PERSISTED_VERDICTS:
            # Defensive — shouldn't happen given classify_with_gemini's contract.
            logger.warning("Unexpected verdict %r from judge; skipping insert", verdict)
            return

        await _insert_row(
            question=question,
            answer_text=_answer_for_verdict(verdict, answer),
            verdict=verdict,
            endpoint=endpoint,
            client_id=client_id,
            meeting_log_id=meeting_log_id,
        )

    except Exception as e:
        # Never let a classifier failure bubble up — this is post-response work.
        logger.error("classify_and_log unexpected failure: %s", e, exc_info=True)
