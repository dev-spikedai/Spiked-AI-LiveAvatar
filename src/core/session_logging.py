"""Opt-in, uniquely named application log files."""

from __future__ import annotations

import contextvars
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


_LOG_DIR = Path(os.getenv("APP_LOG_DIR", Path(__file__).resolve().parents[2] / "logs"))
_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

_current_session_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "current_session_id", default=None
)


def set_current_session_id(session_id: Optional[str]) -> None:
    """Set the active session/run ID in the current async execution context."""
    _current_session_id.set(session_id)


def get_current_session_id() -> Optional[str]:
    """Get the active session/run ID from the current async execution context."""
    return _current_session_id.get()


class SessionLogFilter(logging.Filter):
    """Filter that only permits records matching the bound session/run identifier."""

    def __init__(self, session_id: str):
        super().__init__()
        self.session_id = session_id

    def filter(self, record: logging.LogRecord) -> bool:
        # Check explicit attributes attached to the record
        record_id = getattr(record, "session_id", None) or getattr(record, "run_id", None)
        if record_id is not None:
            return record_id == self.session_id
        # Fall back to task-local context variable
        active_id = _current_session_id.get()
        if active_id is not None:
            return active_id == self.session_id
        return False


def logs_enabled() -> bool:
    return os.getenv("ENABLE_APP_LOGS", "false").strip().lower() in {"1", "true", "yes", "on"}


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value or "session")[:120].strip("-") or "session"


def open_session_log(session_id: str) -> Optional[logging.FileHandler]:
    if not logs_enabled():
        return None
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")[:-3]
    path = _LOG_DIR / f"app-{_safe_id(session_id)}-{timestamp}.log"
    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    handler.addFilter(SessionLogFilter(session_id))
    logging.getLogger().addHandler(handler)
    return handler


def close_session_log(handler: Optional[logging.FileHandler]) -> None:
    if not handler:
        return
    root = logging.getLogger()
    root.removeHandler(handler)
    handler.close()
