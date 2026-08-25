"""Opt-in, uniquely named application log files."""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


_LOG_DIR = Path(os.getenv("APP_LOG_DIR", Path(__file__).resolve().parents[2] / "logs"))
_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


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
    logging.getLogger().addHandler(handler)
    return handler


def close_session_log(handler: Optional[logging.FileHandler]) -> None:
    if not handler:
        return
    root = logging.getLogger()
    root.removeHandler(handler)
    handler.close()
