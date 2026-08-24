from src.core.mcp_tools import READ_ONLY_TOOLS, choose_tool


def test_mcp_policy_only_selects_live_read_tools():
    assert choose_tool("What questions have they asked?", "meeting_context", True) == "get_live_questions"
    assert choose_tool("What did Alice just say?", "meeting_context", True) == "get_livenote"
    assert choose_tool("How are they feeling?", "meeting_context", True) == "get_meeting_sentiment"
    assert choose_tool("What are our meeting goals?", "meeting_context", True) == "get_meeting_goals"


def test_mcp_policy_never_selects_a_tool_without_a_live_session_for_live_state():
    assert choose_tool("What did they discuss?", "meeting_context", False) is None


def test_mcp_allowlist_contains_no_write_tools():
    assert READ_ONLY_TOOLS
    assert not any(name.startswith(("create", "update", "delete", "send", "edit")) for name in READ_ONLY_TOOLS)
