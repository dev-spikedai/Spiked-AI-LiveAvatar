import asyncio
import logging
import os
from pathlib import Path

import pytest

from src.core.session_logging import (
    close_session_log,
    open_session_log,
    set_current_session_id,
)


def test_concurrent_sessions_have_isolated_logs(monkeypatch, tmp_path):
    monkeypatch.setenv("ENABLE_APP_LOGS", "true")
    monkeypatch.setenv("APP_LOG_DIR", str(tmp_path))
    monkeypatch.setattr("src.core.session_logging._LOG_DIR", tmp_path)

    logger = logging.getLogger("test_isolated_logger")
    logger.setLevel(logging.INFO)

    session_a = "session_aaa_123"
    session_b = "session_bbb_456"

    handler_a = open_session_log(session_a)
    handler_b = open_session_log(session_b)

    assert handler_a is not None
    assert handler_b is not None

    async def run_session_a():
        set_current_session_id(session_a)
        await asyncio.sleep(0.01)
        logger.info("Secret transcript content from meeting A")
        await asyncio.sleep(0.01)
        logger.info("Another message from meeting A")

    async def run_session_b():
        set_current_session_id(session_b)
        await asyncio.sleep(0.01)
        logger.info("Confidential financial figures from meeting B")
        await asyncio.sleep(0.01)
        logger.info("Another message from meeting B")

    async def drive():
        await asyncio.gather(run_session_a(), run_session_b())

    asyncio.run(drive())

    close_session_log(handler_a)
    close_session_log(handler_b)

    # Read log files
    log_files = list(tmp_path.glob("*.log"))
    assert len(log_files) == 2

    file_a = next(f for f in log_files if session_a in f.name)
    file_b = next(f for f in log_files if session_b in f.name)

    content_a = file_a.read_text(encoding="utf-8")
    content_b = file_b.read_text(encoding="utf-8")

    # Verify meeting A has meeting A content and no meeting B content
    assert "Secret transcript content from meeting A" in content_a
    assert "Another message from meeting A" in content_a
    assert "Confidential financial figures from meeting B" not in content_a

    # Verify meeting B has meeting B content and no meeting A content
    assert "Confidential financial figures from meeting B" in content_b
    assert "Another message from meeting B" in content_b
    assert "Secret transcript content from meeting A" not in content_b
