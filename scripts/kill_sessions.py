"""Stop every active LiveAvatar session and evict any Recall bot still in a call.

Two independent leaks strand resources:

* LiveAvatar allows 1 concurrent session (sandbox and live share the pool), so an
  orphan blocks all further starts with `4032 Session concurrency limit reached`.
* The frontend keeps `botId` in local state, so a refresh loses the disconnect
  button while the bot keeps recording server-side.

Run this to clear both:

    .venv/Scripts/python.exe scripts/kill_sessions.py
"""

import os
import sys
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("LIVEAVATAR_API_KEY", "")
BASE_URL = os.getenv("LIVEAVATAR_API_URL", "https://api.liveavatar.com")
HEADERS = {"X-API-KEY": API_KEY, "Content-Type": "application/json"}

RECALL_API_KEY = os.getenv("RECALL_API_KEY", "")
RECALL_BASE_URL = os.getenv("RECALL_BASE_URL", "https://us-west-2.recall.ai").rstrip("/")

# Terminal states. Anything else means the bot is still occupying the meeting.
FINISHED_STATUSES = {"done", "fatal", "call_ended", "media_expired"}


def evict_recall_bots() -> None:
    if not RECALL_API_KEY:
        print("RECALL_API_KEY is not set; skipping bot cleanup.")
        return

    headers = {"Authorization": f"Token {RECALL_API_KEY}"}
    response = httpx.get(f"{RECALL_BASE_URL}/api/v1/bot/", headers=headers, timeout=15)
    response.raise_for_status()
    payload = response.json()
    bots = payload.get("results", payload) if isinstance(payload, dict) else payload

    stranded = []
    for bot in bots:
        changes = bot.get("status_changes") or []
        status = changes[-1].get("code") if changes else "unknown"
        if status not in FINISHED_STATUSES:
            stranded.append((bot.get("id"), status))

    if not stranded:
        print("No bots in a call.")
        return

    for bot_id, status in stranded:
        left = httpx.post(
            f"{RECALL_BASE_URL}/api/v1/bot/{bot_id}/leave_call/",
            headers=headers,
            timeout=15,
        )
        print(f"{bot_id} ({status}) -> leave_call {left.status_code}")


def sweep() -> None:
    if not API_KEY:
        raise SystemExit("LIVEAVATAR_API_KEY is not set")

    listing = httpx.get(
        f"{BASE_URL}/v1/sessions",
        headers=HEADERS,
        params={"type": "active"},
        timeout=15,
    )
    listing.raise_for_status()
    sessions = (listing.json().get("data") or {}).get("results") or []

    if not sessions:
        print("No active sessions.")
    for session in sessions:
        session_id = session.get("id")
        response = httpx.post(
            f"{BASE_URL}/v1/sessions/stop",
            headers=HEADERS,
            json={"session_id": session_id},
            timeout=15,
        )
        print(f"{session_id} sandbox={session.get('is_sandbox')} -> {response.status_code}")

    evict_recall_bots()


def main() -> None:
    # --watch keeps sweeping so a run that crashes mid-meeting frees its slot
    # without you having to notice and intervene.
    if "--watch" in sys.argv:
        interval = 20
        print(f"Watching; sweeping every {interval}s. Ctrl+C to stop.")
        while True:
            try:
                sweep()
            except Exception as error:  # keep watching through transient API errors
                print(f"sweep failed: {error}")
            time.sleep(interval)
    else:
        sweep()


if __name__ == "__main__":
    main()
