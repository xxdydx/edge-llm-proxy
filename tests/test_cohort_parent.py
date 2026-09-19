from edgeproxy.cohort_parent import is_cohort_parent_candidate


AGENT = {"name": "Agent", "input_schema": {"type": "object"}}
REMINDER = "<system-reminder><total_tokens>14999733 tokens left</total_tokens></system-reminder>"


def request(*, role="system", content=None, tools=None):
    return {
        "tools": [AGENT] if tools is None else tools,
        "messages": [{"role": role, "content": REMINDER if content is None else content}],
    }


def test_accepts_last_text_block_of_root_system_reminder():
    content = [
        {"type": "text", "text": "<system-reminder>other context</system-reminder>"},
        {"type": "text", "text": REMINDER, "cache_control": {"type": "ephemeral"}},
    ]
    assert is_cohort_parent_candidate(request(content=content), {})


def test_rejects_same_shape_from_subagent():
    assert not is_cohort_parent_candidate(
        request(), {"x-claude-code-agent-id": "child-1"}
    )


def test_rejects_user_message_and_generic_system_reminder():
    assert not is_cohort_parent_candidate(request(role="user"), {})
    assert not is_cohort_parent_candidate(
        request(content="<system-reminder>ordinary</system-reminder>"), {}
    )


def test_rejects_request_without_agent_tool():
    assert not is_cohort_parent_candidate(request(tools=[]), {})
