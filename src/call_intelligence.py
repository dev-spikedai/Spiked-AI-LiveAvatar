"""Read-only client for the recall backend's conversation intelligence.

The sentiment backend already computes participant roles, concerns, qualification
progress, and call health on its own ~3 minute cadence. The agent consumes that
work rather than recomputing it in-process, which is what turns a name string
into a role the answer can actually be aimed at.

Two properties this module is built around:

* **Never on the speak path.** A background task refreshes the snapshot; the
  reply path only ever reads an already-materialized dict. A slow or dead
  sentiment backend degrades the agent to the name-only behavior that existed
  before this module, and never adds a millisecond to time-to-speech.
* **Auth is the join key.** The ``/sentiment/*`` component routes take no bot id;
  the backend resolves the caller's active bot from the JWT. So the only thing
  needed here is the user's token, which the run already holds.
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("LiveAvatar-Spiked")

# Meeting-bot backend only. Stealth/hot-mic sessions are registered on a
# different service and are out of scope for the avatar.
SENTIMENT_BACKEND_URL = os.getenv(
    "SPIKED_RECALL_BACKEND_URL",
    "https://recall-backend-production-409019309412.us-central1.run.app",
).rstrip("/")

# The backend recomputes roughly every 180s. Polling at half that keeps the
# snapshot at most one cycle stale without adding meaningful load.
REFRESH_INTERVAL_S = float(os.getenv("AGENT_INTEL_REFRESH_S", "90"))
FETCH_TIMEOUT_S = float(os.getenv("AGENT_INTEL_TIMEOUT_S", "8"))
# Runs are never evicted from the orchestrator's in-process registry, so an
# unbounded poll per run would outlive the meeting it belongs to. Cap the loop
# at a length no real call exceeds; a longer one degrades to a stale snapshot,
# which is the same failure mode as the backend being unreachable.
MAX_LIFETIME_S = float(os.getenv("AGENT_INTEL_MAX_LIFETIME_S", "14400"))

_COMPONENTS = {
    "participants": "/sentiment/participants",
    "medpic": "/sentiment/medpic",
    "alerts": "/sentiment/alerts",
    "health": "/sentiment/health",
}


def _first_name(name: str) -> str:
    return (name or "").strip().split(" ")[0].casefold()


class CallIntelligence:
    """A periodically refreshed snapshot of what the sentiment backend knows."""

    def __init__(self, auth_token: str):
        self._auth_token = auth_token
        self._snapshot: Dict[str, Any] = {}
        self._task: Optional[asyncio.Task] = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._refresh_loop())

    def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = None

    async def _refresh_loop(self) -> None:
        deadline = asyncio.get_running_loop().time() + MAX_LIFETIME_S
        while asyncio.get_running_loop().time() < deadline:
            try:
                await self.refresh()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("[Intel] Refresh cycle failed", exc_info=True)
            await asyncio.sleep(REFRESH_INTERVAL_S)
        logger.info("[Intel] Refresher retired after max lifetime; snapshot frozen")

    async def refresh(self) -> None:
        if not self._auth_token:
            return
        headers = {"Authorization": f"Bearer {self._auth_token}"}
        async with httpx.AsyncClient(timeout=FETCH_TIMEOUT_S) as client:

            async def fetch(name: str, path: str) -> None:
                try:
                    response = await client.get(f"{SENTIMENT_BACKEND_URL}{path}", headers=headers)
                    if response.status_code != 200:
                        # 401/403/404 are the normal shape of "no active bot for
                        # this user yet", not an error worth escalating.
                        logger.debug("[Intel] %s -> %s", name, response.status_code)
                        return
                    self._snapshot[name] = response.json() or {}
                except Exception:
                    logger.debug("[Intel] %s fetch failed", name, exc_info=True)

            await asyncio.gather(*(fetch(n, p) for n, p in _COMPONENTS.items()))

        logger.info(
            "[Intel] Snapshot refreshed components=%s participants=%d",
            sorted(self._snapshot.keys()),
            len(self.participant_cards()),
        )

    # -- accessors ---------------------------------------------------------

    def participant_cards(self) -> List[Dict[str, Any]]:
        cards = (self._snapshot.get("participants") or {}).get("participant_cards")
        return cards if isinstance(cards, list) else []

    def card_for(self, speaker: str) -> Optional[Dict[str, Any]]:
        """Match a Deepgram speaker label to a sentiment participant card.

        Display names differ between Recall and the sentiment backend more often
        than they match exactly, so this falls back to first name.
        """

        target = (speaker or "").strip().casefold()
        if not target:
            return None
        cards = self.participant_cards()
        for card in cards:
            if (card.get("speaker") or "").strip().casefold() == target:
                return card
        target_first = _first_name(speaker)
        for card in cards:
            if _first_name(card.get("speaker") or "") == target_first:
                return card
        return None

    # -- prompt blocks -----------------------------------------------------

    def speaker_dossier(self, speaker: str) -> str:
        """One line describing who is talking, or empty when unknown."""

        card = self.card_for(speaker)
        if not card:
            return ""
        bits = []
        role = (card.get("role") or "").strip()
        if role:
            bits.append(f"role: {role}")
        concerns = [c for c in (card.get("concerns_list") or []) if c]
        if concerns:
            bits.append("open concerns: " + "; ".join(str(c) for c in concerns[:3]))
        if not bits:
            return ""
        return f"{card.get('speaker') or speaker} ({', '.join(bits)})"

    def room_block(self) -> str:
        """Who else is on the call and what they each care about."""

        lines = []
        for card in self.participant_cards()[:8]:
            name = card.get("speaker") or "Participant"
            role = (card.get("role") or "").strip()
            lines.append(f"- {name}{f' — {role}' if role else ''}")
        return "\n".join(lines)

    def call_state_block(self) -> str:
        """What the call has established so far, from the backend's own analysis.

        This is the durable answer to "what has already been discussed". It does
        not restore verbatim recall of early turns — the in-process history is
        still capped — but it survives well past that cap and is what actually
        matters for not re-asking a question the room already settled.
        """

        lines = []

        medpic = (self._snapshot.get("medpic") or {}).get("medpic_progress") or {}
        covered, open_items = [], []
        for key, value in medpic.items():
            if not isinstance(value, dict):
                continue
            (covered if value.get("discussed") else open_items).append(key)
        if covered:
            lines.append("Already established: " + ", ".join(sorted(covered)))
        if open_items:
            lines.append("Not yet covered: " + ", ".join(sorted(open_items)))

        alerts = (self._snapshot.get("alerts") or {}).get("critical_alerts") or []
        live = [
            str(a.get("message") or a.get("title") or "").strip()
            for a in alerts
            if isinstance(a, dict) and not a.get("acknowledged")
        ]
        live = [item for item in live if item]
        if live:
            lines.append("Live concerns in the room: " + "; ".join(live[:3]))

        return "\n".join(lines)
