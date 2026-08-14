import re
import math
import struct
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional

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
