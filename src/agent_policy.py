import re
import math
import struct
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import webrtcvad
except ImportError:  # pragma: no cover - energy fallback is used in minimal installs
    webrtcvad = None


class AgentState(str, Enum):
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTING = "interrupting"


@dataclass(frozen=True)
class InvocationDecision:
    addressed: bool
    matched_name: Optional[str] = None
    reason: str = "name_not_present"


@dataclass(frozen=True)
class TurnDecision:
    """Outcome of the floor-control gate for one assembled participant turn."""

    should_reply: bool
    reason: str
    matched_name: Optional[str] = None


@dataclass
class FloorState:
    """Who currently holds the conversational floor with the agent."""

    last_bot_finished_at: Optional[float] = None
    last_addressed_participant: Optional[str] = None
    consecutive_followups: int = 0


@dataclass
class FinalUtteranceBuffer:
    """Collect finalized Deepgram segments until the speaker finishes a turn."""

    segments: List[str] = field(default_factory=list)
    words: List[Dict[str, Any]] = field(default_factory=list)

    def add_result(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if data.get("type") == "UtteranceEnd":
            return self.flush()
        if data.get("type") != "Results" or not data.get("is_final"):
            return None

        alternatives = data.get("channel", {}).get("alternatives", [])
        if alternatives:
            alternative = alternatives[0]
            text = (alternative.get("transcript") or "").strip()
            if text:
                self.segments.append(text)
                self.words.extend(alternative.get("words") or [])

        if data.get("speech_final"):
            return self.flush()
        return None

    def flush(self) -> Optional[Dict[str, Any]]:
        text = " ".join(self.segments).strip()
        if not text:
            self.segments.clear()
            self.words.clear()
            return None
        result = {"text": text, "words": list(self.words)}
        self.segments.clear()
        self.words.clear()
        return result


def detect_invocation(text: str, bot_name: str) -> InvocationDecision:
    """Match the configured wake name, including conservative STT formatting variants."""

    def normalized_words(value: str) -> List[str]:
        # Deepgram may render a brand as either "SpikedAI", "Spiked AI", or
        # "Spiked A.I.". Normalize formatting without enabling broad fuzzy matches.
        value = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value)
        value = re.sub(r"[^a-zA-Z0-9]+", " ", value).casefold().strip()
        words = value.split()
        collapsed: List[str] = []
        index = 0
        while index < len(words):
            if index + 1 < len(words) and len(words[index]) == len(words[index + 1]) == 1:
                collapsed.append(words[index] + words[index + 1])
                index += 2
            else:
                collapsed.append(words[index])
                index += 1
        return collapsed

    configured = (bot_name or "Tom").strip()
    configured_words = normalized_words(configured) or ["tom"]
    aliases = {tuple(configured_words)}
    compact_name = "".join(configured_words)
    if compact_name == "tom":
        aliases.add(("thom",))
    if compact_name == "spikedai":
        # "spike AI" is a frequent, narrow STT repair for the SpikedAI brand.
        aliases.update({("spiked", "ai"), ("spike", "ai")})

    normalized_text = " ".join(normalized_words(text))
    for alias in sorted(aliases, key=lambda item: (len(item), len(" ".join(item))), reverse=True):
        name = " ".join(alias)
        if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", normalized_text):
            return InvocationDecision(True, name, "explicit_name")
    return InvocationDecision(False)


def build_entity_catalog(context: Dict[str, Any]) -> List[str]:
    values: List[str] = []
    for key in ("company_name", "bot_name"):
        value = context.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    for value in context.get("keywords") or []:
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    products = context.get("products_services")
    if isinstance(products, str):
        values.extend(part.strip() for part in re.split(r"[,;\n]", products) if part.strip())

    deduped: Dict[str, str] = {}
    for value in values:
        if 1 < len(value) <= 80:
            deduped.setdefault(value.casefold(), value)
    return list(deduped.values())[:100]


def closest_entities(text: str, catalog: Iterable[str], limit: int = 15) -> List[str]:
    normalized = text.casefold()
    scored = []
    for entity in catalog:
        entity_norm = entity.casefold()
        score = SequenceMatcher(None, normalized, entity_norm).ratio()
        if entity_norm in normalized:
            score += 1.0
        scored.append((score, entity))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [entity for _, entity in scored[:limit]]


def apply_validated_corrections(
    raw_text: str,
    corrections: Iterable[Dict[str, Any]],
    catalog: Iterable[str],
    min_confidence: float = 0.85,
) -> str:
    """Apply only high-confidence span replacements to verified catalog entities."""

    result = raw_text
    catalog_by_key = {item.casefold(): item for item in catalog}
    for correction in corrections:
        raw = str(correction.get("raw") or "").strip()
        replacement_key = str(correction.get("replacement") or "").strip().casefold()
        confidence = float(correction.get("confidence") or 0.0)
        replacement = catalog_by_key.get(replacement_key)
        if not raw or not replacement or confidence < min_confidence:
            continue
        if SequenceMatcher(None, raw.casefold(), replacement.casefold()).ratio() < 0.30:
            continue
        result = re.sub(re.escape(raw), replacement, result, count=1, flags=re.IGNORECASE)
    return result


def requires_company_knowledge(text: str, catalog: Iterable[str]) -> bool:
    normalized = text.casefold()
    factual_terms = (
        "price", "pricing", "cost", "product", "offering", "feature", "security",
        "architecture", "roadmap", "sla", "integration", "compliance", "service",
        "platform", "what do you offer", "what can you do", "your tool", "your company",
    )
    if any(term in normalized for term in factual_terms):
        return True
    return any(
        len(entity) >= 4 and entity.casefold() in normalized
        for entity in catalog
        if entity.casefold() not in {"tom", "thom"}
    )


# Verbs that make a nameless remainder an instruction to the agent rather than a
# remark about it.
_IMPERATIVE_STARTERS = frozenset({
    "tell", "explain", "elaborate", "describe", "summarize", "summarise",
    "list", "give", "show", "walk", "define", "compare",
})


_ANAPHORA = frozenset({
    "it", "its", "it's", "that", "this", "they", "them", "their", "those",
    "these", "he", "she", "him", "her", "his", "one", "ones",
})


def is_directly_addressed(text: str, bot_name: str) -> bool:
    """True when the turn opens by naming the agent, then asks something.

    "Tom, what is your pricing?" is unambiguous. "I asked Tom about pricing" puts
    the name mid-sentence and is a third-person mention, so it must not qualify —
    resolving that correctly needs the LLM addressee gate, not a regex.
    """

    invocation = detect_invocation(text, bot_name)
    if not invocation.addressed or not invocation.matched_name:
        return False

    normalized = _normalize_for_match(text)
    name = invocation.matched_name
    if not normalized.startswith(name):
        return False

    remainder = normalized[len(name):].split()
    if not remainder:
        return False
    # A trailing "?" is the strongest signal. Otherwise only a direct imperative
    # counts: "Tom is great at pricing" opens with the name and reads as
    # question-shaped to a looser check, but it is a statement about her.
    # Anything else falls through to the LLM gate, which costs latency, not
    # correctness.
    return text.strip().endswith("?") or remainder[0] in _IMPERATIVE_STARTERS


def needs_context_resolution(text: str) -> bool:
    """True when a turn leans on earlier context and needs LLM query repair.

    A self-contained question can go straight to retrieval, skipping a full
    model round trip. Anything referring back to prior turns cannot.

    NOTE: the fast path in live_avatar.py no longer calls this for the
    company-knowledge route — the answer model receives conversation history
    and can resolve pronouns on its own, saving one LLM round trip per turn.
    This function is still used by callers that need strict anaphora gating.
    """

    words = re.sub(r"[^a-z0-9' ]+", " ", (text or "").casefold()).split()
    return any(word in _ANAPHORA for word in words)


def normalize_reply(text: str, max_words: int = 45, max_sentences: int = 2) -> str:
    clean = re.sub(r"\s+", " ", (text or "").strip())
    clean = clean.replace("**", "").replace("__", "").replace("`", "")
    clean = re.sub(r"^#+\s*", "", clean)
    if not clean:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", clean)
    clean = " ".join(sentences[:max_sentences]).strip()
    words = clean.split()
    if len(words) > max_words:
        clean = " ".join(words[:max_words]).rstrip(",;:")
        if clean[-1:] not in ".!?":
            clean += "."
    return clean


WORDS_PER_SECOND = 2.5

# Openers that make a nameless turn read as a continuation of the agent's answer
# rather than as room chatter. Deliberately narrow: anything not listed here needs
# the wake name, matching the conservative default of the rest of this module.
_FOLLOWUP_STARTERS = frozenset({
    "what", "how", "why", "when", "where", "who", "which", "whose",
    "can", "could", "would", "will", "should", "do", "does", "did",
    "is", "are", "was", "were", "tell", "explain", "elaborate", "repeat",
})

# A finalized segment ending on one of these is a speaker pausing mid-thought,
# not a completed turn.
_DANGLING_TOKENS = frozenset({
    "and", "but", "so", "or", "if", "when", "because", "that", "which",
    "with", "for", "to", "of", "in", "on", "at", "the", "a", "an", "um", "uh",
})


def _normalize_for_match(text: str) -> str:
    collapsed = re.sub(r"[^a-z0-9 ]+", " ", (text or "").casefold())
    return re.sub(r"\s+", " ", collapsed).strip()


def estimate_speech_seconds(text: str, words_per_second: float = WORDS_PER_SECOND) -> float:
    """Approximate spoken duration, used for echo expiry and the speak watchdog."""

    words = len((text or "").split())
    return words / words_per_second if words else 0.0


def is_probably_incomplete(text: str) -> bool:
    """True when a finalized segment looks like a mid-thought pause."""

    clean = (text or "").strip()
    if not clean:
        return True
    if clean[-1] not in ".!?":
        return True
    words = _normalize_for_match(clean).split()
    return bool(words) and words[-1] in _DANGLING_TOKENS


def looks_like_followup(text: str) -> bool:
    """Question-shaped continuation that may skip the wake name inside the window."""

    clean = (text or "").strip()
    words = _normalize_for_match(clean).split()
    if len(words) < 2:
        return False
    if clean.endswith("?"):
        return True
    if words[0] in _FOLLOWUP_STARTERS:
        return True
    # "and what about pricing", "okay how does that work"
    if words[0] in {"and", "but", "so", "okay", "ok"} and words[1] in _FOLLOWUP_STARTERS:
        return True
    return False


class EchoSuppressor:
    """Reject transcripts that are the avatar's own voice returning through the room.

    Recall includes bot audio in the separate-audio stream, and a participant on
    speakers re-injects the avatar through their own microphone under their own
    participant id. Neither case can be caught by comparing participant names, so
    match against what the agent actually said instead.
    """

    def __init__(self, similarity_threshold: float = 0.72, tail_seconds: float = 2.5):
        self.similarity_threshold = similarity_threshold
        self.tail_seconds = tail_seconds
        self._entries: List[Tuple[float, str]] = []

    def note_bot_speech(self, text: str, now: float) -> None:
        normalized = _normalize_for_match(text)
        if not normalized:
            return
        expires_at = now + estimate_speech_seconds(text) + self.tail_seconds
        self._entries.append((expires_at, normalized))

    def _prune(self, now: float) -> None:
        self._entries = [entry for entry in self._entries if entry[0] > now]

    def is_echo(self, text: str, now: float) -> bool:
        self._prune(now)
        candidate = _normalize_for_match(text)
        if not candidate:
            return False
        candidate_words = candidate.split()
        for _, spoken in self._entries:
            # A microphone usually captures only a fragment of the avatar's reply.
            if len(candidate_words) >= 3 and candidate in spoken:
                return True
            if SequenceMatcher(None, candidate, spoken).ratio() >= self.similarity_threshold:
                return True
        return False

    def reset(self) -> None:
        self._entries.clear()


class SpeechGovernor:
    """Hard, non-LLM ceiling on how often and how repetitively the agent may speak.

    This is the structural guarantee against a talking spree: it holds even when
    the model misbehaves, a control event is lost, or turns arrive fragmented.
    """

    def __init__(
        self,
        cooldown_seconds: float = 2.0,
        max_replies_per_window: int = 4,
        window_seconds: float = 30.0,
        duplicate_threshold: float = 0.90,
        duplicate_window_seconds: float = 20.0,
    ):
        self.cooldown_seconds = cooldown_seconds
        self.max_replies_per_window = max_replies_per_window
        self.window_seconds = window_seconds
        self.duplicate_threshold = duplicate_threshold
        self.duplicate_window_seconds = duplicate_window_seconds
        self._reply_times: List[float] = []
        self._recent_replies: List[Tuple[float, str]] = []

    def _prune(self, now: float) -> None:
        self._reply_times = [item for item in self._reply_times if now - item <= self.window_seconds]
        self._recent_replies = [
            item for item in self._recent_replies if now - item[0] <= self.duplicate_window_seconds
        ]

    def allows_reply(self, now: float) -> Tuple[bool, str]:
        self._prune(now)
        if self._reply_times and now - self._reply_times[-1] < self.cooldown_seconds:
            return False, "cooldown"
        if len(self._reply_times) >= self.max_replies_per_window:
            return False, "rate_limited"
        return True, "allowed"

    def is_duplicate(self, text: str, now: float) -> bool:
        self._prune(now)
        candidate = _normalize_for_match(text)
        if not candidate:
            return False
        return any(
            SequenceMatcher(None, candidate, previous).ratio() >= self.duplicate_threshold
            for _, previous in self._recent_replies
        )

    def note_reply(self, text: str, now: float) -> None:
        self._reply_times.append(now)
        self._recent_replies.append((now, _normalize_for_match(text)))
        self._prune(now)


def evaluate_turn(
    text: str,
    participant_id: str,
    bot_name: str,
    floor: FloorState,
    now: float,
    followup_window_seconds: float = 8.0,
    max_consecutive_followups: int = 2,
    agent_is_idle: bool = True,
) -> TurnDecision:
    """Decide whether an assembled turn has the floor.

    The wake name always grants it. A bounded window after the agent finishes
    speaking also lets the same speaker ask one or two question-shaped follow-ups
    without repeating the name, so natural conversation does not require chanting.

    Set `max_consecutive_followups=0` to disable nameless follow-ups entirely and
    fall back to requiring the wake name on every single turn.
    """

    invocation = detect_invocation(text, bot_name)
    if invocation.addressed:
        return TurnDecision(True, "explicit_name", invocation.matched_name)
    if not agent_is_idle:
        # Room chatter overlapping the agent's own reply is never a follow-up.
        # Without the name it has no claim on the floor.
        return TurnDecision(False, "agent_not_idle")
    if floor.last_bot_finished_at is None:
        return TurnDecision(False, "name_not_present")
    if participant_id != floor.last_addressed_participant:
        return TurnDecision(False, "different_speaker")
    if now - floor.last_bot_finished_at > followup_window_seconds:
        return TurnDecision(False, "followup_window_expired")
    if floor.consecutive_followups >= max_consecutive_followups:
        return TurnDecision(False, "followup_limit")
    if not looks_like_followup(text):
        return TurnDecision(False, "not_followup_shaped")
    return TurnDecision(True, "followup_window")


class SustainedSpeechDetector:
    """20 ms voice-frame detector used only for barge-in, never invocation."""

    def __init__(self, sample_rate: int = 16000, threshold_ms: int = 700):
        self.sample_rate = sample_rate
        self.frame_ms = 20
        self.frame_bytes = sample_rate * self.frame_ms // 1000 * 2
        self.required_frames = max(1, threshold_ms // self.frame_ms)
        self._buffer = bytearray()
        self._consecutive_voiced = 0
        self._vad = webrtcvad.Vad(2) if webrtcvad else None

    def reset(self) -> None:
        self._buffer.clear()
        self._consecutive_voiced = 0

    def feed(self, pcm: bytes) -> bool:
        self._buffer.extend(pcm)
        while len(self._buffer) >= self.frame_bytes:
            frame = bytes(self._buffer[: self.frame_bytes])
            del self._buffer[: self.frame_bytes]
            if self._vad:
                voiced = self._vad.is_speech(frame, self.sample_rate)
            else:
                samples = struct.unpack(f"<{len(frame) // 2}h", frame)
                rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
                voiced = rms >= 350
            self._consecutive_voiced = self._consecutive_voiced + 1 if voiced else 0
            if self._consecutive_voiced >= self.required_frames:
                self._consecutive_voiced = 0
                return True
        return False
