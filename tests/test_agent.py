"""Tests for the three submit_answer_status paths in Agent.ask() — the
model calling submit_answer cleanly, calling it with malformed
arguments, and not calling it at all. The LLM call itself is mocked
(SimpleNamespace fakes matching the openai SDK response shape) so these
run fast, deterministically, with no API key or network dependency —
embeddings still run for real locally (fastembed, no network either).
"""
from types import SimpleNamespace

from src.agent import FEW_SHOT_EXAMPLE, Agent


def _fake_response(tool_calls=None, content=None):
    message = SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _fake_tool_call(name, arguments_json, call_id="call_1"):
    function = SimpleNamespace(name=name, arguments=arguments_json)
    return SimpleNamespace(id=call_id, function=function)


def _agent_with_mocked_llm(monkeypatch, fake_chat):
    agent = Agent(tenant_id="tenant_a")
    monkeypatch.setattr(agent.llm, "chat", fake_chat)
    monkeypatch.setattr("src.agent.write_audit_record", lambda record: _CAPTURED.append(record))
    return agent


_CAPTURED: list = []


def setup_function(_):
    _CAPTURED.clear()


def test_submit_answer_status_ok(monkeypatch):
    call = _fake_tool_call("submit_answer", '{"answer": "The fee is $129.", "citations": []}')
    agent = _agent_with_mocked_llm(monkeypatch, lambda messages, tools=None, tool_choice="auto": _fake_response(tool_calls=[call]))

    result = agent.ask("What is the fee?")

    assert result["answer"] == "The fee is $129."
    assert result["citations"] == []
    assert _CAPTURED[-1].submit_answer_status == "ok"
    assert _CAPTURED[-1].submit_answer_error is None


def test_submit_answer_status_malformed_missing_required_field(monkeypatch):
    # Valid JSON, but missing the required "answer" key.
    call = _fake_tool_call("submit_answer", '{"citations": []}')
    agent = _agent_with_mocked_llm(monkeypatch, lambda messages, tools=None, tool_choice="auto": _fake_response(tool_calls=[call]))

    result = agent.ask("What is the fee?")

    assert _CAPTURED[-1].submit_answer_status == "malformed"
    assert _CAPTURED[-1].submit_answer_error is not None
    assert result["answer"]  # never crashes, never returns None/empty


def test_submit_answer_status_malformed_invalid_json(monkeypatch):
    call = _fake_tool_call("submit_answer", "not valid json{")
    agent = _agent_with_mocked_llm(monkeypatch, lambda messages, tools=None, tool_choice="auto": _fake_response(tool_calls=[call]))

    result = agent.ask("What is the fee?")

    assert _CAPTURED[-1].submit_answer_status == "malformed"
    assert result["answer"]


def test_submit_answer_status_not_called(monkeypatch):
    agent = _agent_with_mocked_llm(
        monkeypatch,
        lambda messages, tools=None, tool_choice="auto": _fake_response(tool_calls=None, content="The fee is $129."),
    )

    result = agent.ask("What is the fee?")

    assert result["answer"] == "The fee is $129."
    assert _CAPTURED[-1].submit_answer_status == "not_called"


def test_few_shot_example_present_in_constructed_messages(monkeypatch):
    captured_messages = []

    def fake_chat(messages, tools=None, tool_choice="auto"):
        captured_messages.append(messages)
        call = _fake_tool_call("submit_answer", '{"answer": "ok", "citations": []}')
        return _fake_response(tool_calls=[call])

    agent = _agent_with_mocked_llm(monkeypatch, fake_chat)
    agent.ask("test question")

    messages = captured_messages[0]
    assert messages[0]["role"] == "system"
    # the synthetic few-shot turns appear between the system prompt and
    # the real user question
    assert messages[1 : 1 + len(FEW_SHOT_EXAMPLE)] == FEW_SHOT_EXAMPLE
    assert messages[-1] == {"role": "user", "content": "test question"}
