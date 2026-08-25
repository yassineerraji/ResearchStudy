"""Unit tests for integrations/llm_client.py: the tool-call loop, retries, and secret-free request assembly, against a fake stand-in for the OpenAI client."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import openai
import pytest

from supply_chain_simulator.integrations.llm_client import (
    DecisionKey,
    FakeLLMClient,
    LLMIntegrationError,
    LLMToolLoopRequest,
    OpenAIResponsesClient,
    ReplayLLMClient,
    ReplayTraceError,
    ToolSpec,
    interaction_to_dict,
)

_REQUEST = httpx.Request("POST", "https://example.test/v1/responses")


def _decision_key(shipment_id: str = "shipment_001_001", day: int = 21) -> DecisionKey:
    return DecisionKey(
        experiment_id="exp",
        scenario_id="scenario",
        replication=1,
        run_kind="DISRUPTED",
        day=day,
        shipment_id=shipment_id,
        observation_hash="a" * 64,
    )


def _request(**overrides: object) -> LLMToolLoopRequest:
    defaults: dict[str, object] = {
        "decision_key": _decision_key(),
        "system_prompt": "system",
        "user_message": "user",
        "tools": (
            ToolSpec(name="get_shipment_context", description="d", parameters_schema={}),
            ToolSpec(name="submit_action", description="d", parameters_schema={}),
        ),
        "submit_tool_name": "submit_action",
        "model": "gpt-5.4-mini",
        "temperature": 0.0,
        "max_output_tokens": 1000,
        "max_tool_calls": 8,
        "request_timeout_seconds": 60,
        "max_retries": 3,
    }
    defaults.update(overrides)
    return LLMToolLoopRequest(**defaults)  # type: ignore[arg-type]


@dataclass
class _FakeFunctionCall:
    call_id: str
    name: str
    arguments: str
    type: str = "function_call"


@dataclass
class _FakeUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass
class _FakeResponse:
    output: list[_FakeFunctionCall] = field(default_factory=list)
    usage: _FakeUsage | None = None

    def model_dump(self, mode: str = "json") -> dict[str, object]:
        return {"output": [call.name for call in self.output], "mode": mode}


def _function_call_response(call_id: str, name: str, arguments: dict[str, object]) -> _FakeResponse:
    return _FakeResponse(
        output=[_FakeFunctionCall(call_id=call_id, name=name, arguments=json.dumps(arguments))],
        usage=_FakeUsage(input_tokens=10, output_tokens=5, total_tokens=15),
    )


def _no_call_response() -> _FakeResponse:
    return _FakeResponse(output=[], usage=_FakeUsage(input_tokens=3, output_tokens=1, total_tokens=4))


class _FakeResponsesEndpoint:
    def __init__(self, script: list[object]) -> None:
        self._script = list(script)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> _FakeResponse:
        self.calls.append(kwargs)
        item = self._script.pop(0)
        if isinstance(item, BaseException):
            raise item
        assert isinstance(item, _FakeResponse)
        return item


class _FakeOpenAIClient:
    def __init__(self, script: list[object]) -> None:
        self.responses = _FakeResponsesEndpoint(script)


def _echo_tool(name: str, arguments: dict[str, object]) -> dict[str, object]:
    return {"echoed": name, "arguments": arguments}


class TestOpenAIResponsesClientToolLoop:
    def test_normal_submission_records_both_calls_and_stops(self) -> None:
        fake_client = _FakeOpenAIClient(
            [
                _function_call_response("call_1", "get_shipment_context", {}),
                _function_call_response(
                    "call_2",
                    "submit_action",
                    {
                        "shipment_id": "shipment_001_001",
                        "action_type": "WAIT",
                        "route_id": None,
                        "reason_code": "LOWER_ESTIMATED_COST",
                        "rationale": "cheapest option",
                    },
                ),
            ]
        )
        client = OpenAIResponsesClient(api_key="unused", client=fake_client)  # type: ignore[arg-type]

        result = client.run_tool_loop(_request(), _echo_tool)

        assert result.stop_reason == "submitted"
        assert result.submitted_action == {
            "shipment_id": "shipment_001_001",
            "action_type": "WAIT",
            "route_id": None,
            "reason_code": "LOWER_ESTIMATED_COST",
            "rationale": "cheapest option",
        }
        assert [call.name for call in result.tool_calls] == [
            "get_shipment_context",
            "submit_action",
        ]
        assert [out.name for out in result.tool_outputs] == [
            "get_shipment_context",
            "submit_action",
        ]
        assert result.tool_outputs[0].output == {
            "echoed": "get_shipment_context",
            "arguments": {},
        }
        assert result.token_usage == {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30}
        assert result.attempt_count == 2
        assert result.decision_key == _decision_key()
        assert len(fake_client.responses.calls) == 2

    def test_store_false_and_no_secrets_in_request(self) -> None:
        fake_client = _FakeOpenAIClient([_no_call_response()])
        client = OpenAIResponsesClient(api_key="sk-should-never-appear", client=fake_client)  # type: ignore[arg-type]

        result = client.run_tool_loop(_request(), _echo_tool)

        assert fake_client.responses.calls[0]["store"] is False
        serialized = json.dumps(result.request_without_secrets)
        assert "sk-should-never-appear" not in serialized

    def test_no_submission_when_model_never_calls_a_tool(self) -> None:
        fake_client = _FakeOpenAIClient([_no_call_response()])
        client = OpenAIResponsesClient(api_key="unused", client=fake_client)  # type: ignore[arg-type]

        result = client.run_tool_loop(_request(), _echo_tool)

        assert result.stop_reason == "no_submission"
        assert result.submitted_action is None
        assert result.tool_calls == ()

    def test_tool_limit_reached_stops_without_submission(self) -> None:
        fake_client = _FakeOpenAIClient(
            [
                _function_call_response("call_1", "get_shipment_context", {}),
                _function_call_response("call_2", "get_destination_context", {}),
            ]
        )
        client = OpenAIResponsesClient(api_key="unused", client=fake_client)  # type: ignore[arg-type]

        result = client.run_tool_loop(_request(max_tool_calls=1), _echo_tool)

        assert result.stop_reason == "tool_limit_reached"
        assert result.submitted_action is None
        assert len(result.tool_calls) == 1
        assert len(fake_client.responses.calls) == 1


class TestOpenAIResponsesClientRetries:
    def test_retries_transient_error_then_succeeds(self) -> None:
        connection_error = openai.APIConnectionError(request=_REQUEST)
        fake_client = _FakeOpenAIClient(
            [
                connection_error,
                _function_call_response(
                    "call_1",
                    "submit_action",
                    {
                        "shipment_id": "s1",
                        "action_type": "ABSTAIN",
                        "route_id": None,
                        "reason_code": "INSUFFICIENT_INFORMATION",
                        "rationale": "n/a",
                    },
                ),
            ]
        )
        sleeps: list[float] = []
        client = OpenAIResponsesClient(
            api_key="unused", client=fake_client, sleep=sleeps.append  # type: ignore[arg-type]
        )

        result = client.run_tool_loop(_request(), _echo_tool)

        assert result.stop_reason == "submitted"
        assert result.attempt_count == 2
        assert sleeps == [1.0]

    def test_gives_up_after_max_retries_and_raises_llm_integration_error(self) -> None:
        connection_error = openai.APIConnectionError(request=_REQUEST)
        fake_client = _FakeOpenAIClient([connection_error, connection_error, connection_error])
        sleeps: list[float] = []
        client = OpenAIResponsesClient(
            api_key="unused", client=fake_client, sleep=sleeps.append  # type: ignore[arg-type]
        )

        with pytest.raises(LLMIntegrationError):
            client.run_tool_loop(_request(max_retries=2), _echo_tool)

        assert sleeps == [1.0, 2.0]
        assert len(fake_client.responses.calls) == 3

    def test_does_not_retry_bad_request_error(self) -> None:
        response = httpx.Response(400, request=_REQUEST)
        bad_request = openai.BadRequestError("malformed", response=response, body=None)
        fake_client = _FakeOpenAIClient([bad_request])
        client = OpenAIResponsesClient(api_key="unused", client=fake_client)  # type: ignore[arg-type]

        with pytest.raises(LLMIntegrationError):
            client.run_tool_loop(_request(), _echo_tool)

        assert len(fake_client.responses.calls) == 1


class TestReplayLLMClient:
    def _write_trace(self, tmp_path: Path, lines: list[dict[str, object]]) -> Path:
        trace_path = tmp_path / "llm_interactions.jsonl"
        trace_path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
        return trace_path

    def _entry(self, key: DecisionKey, action_type: str = "WAIT") -> dict[str, object]:
        return {
            "decision_key": {
                "experiment_id": key.experiment_id,
                "scenario_id": key.scenario_id,
                "replication": key.replication,
                "run_kind": key.run_kind,
                "day": key.day,
                "shipment_id": key.shipment_id,
                "observation_hash": key.observation_hash,
            },
            "model": "gpt-5.4-mini",
            "request_without_secrets": {},
            "tool_calls": [],
            "tool_outputs": [],
            "provider_response": None,
            "submitted_action": {
                "shipment_id": key.shipment_id,
                "action_type": action_type,
                "route_id": None,
                "reason_code": "LOWER_ESTIMATED_COST",
                "rationale": "replayed",
            },
            "token_usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            "latency_ms": 12.5,
            "attempt_count": 1,
        }

    def test_matches_recorded_decision_key(self, tmp_path: Path) -> None:
        key = _decision_key()
        trace_path = self._write_trace(tmp_path, [self._entry(key)])
        client = ReplayLLMClient(trace_path)

        result = client.run_tool_loop(_request(decision_key=key), _echo_tool)

        assert result.stop_reason == "submitted"
        assert result.submitted_action is not None
        assert result.submitted_action["action_type"] == "WAIT"

    def test_missing_key_raises_replay_trace_error(self, tmp_path: Path) -> None:
        trace_path = self._write_trace(tmp_path, [self._entry(_decision_key(shipment_id="other"))])
        client = ReplayLLMClient(trace_path)

        with pytest.raises(ReplayTraceError):
            client.run_tool_loop(_request(decision_key=_decision_key()), _echo_tool)

    def test_duplicate_key_raises_replay_trace_error_at_load(self, tmp_path: Path) -> None:
        key = _decision_key()
        trace_path = self._write_trace(tmp_path, [self._entry(key), self._entry(key)])

        with pytest.raises(ReplayTraceError):
            ReplayLLMClient(trace_path)

    def test_unreadable_file_raises_replay_trace_error(self, tmp_path: Path) -> None:
        with pytest.raises(ReplayTraceError):
            ReplayLLMClient(tmp_path / "does_not_exist.jsonl")

    def test_replay_never_calls_execute_tool(self, tmp_path: Path) -> None:
        key = _decision_key()
        trace_path = self._write_trace(tmp_path, [self._entry(key)])
        client = ReplayLLMClient(trace_path)
        calls: list[str] = []

        def _record(name: str, args: dict[str, object]) -> dict[str, object]:
            calls.append(name)
            return {}

        client.run_tool_loop(_request(decision_key=key), _record)

        assert calls == []


class TestFakeLLMClient:
    def test_returns_configured_submission_and_runs_scripted_tool_calls(self) -> None:
        client = FakeLLMClient(
            submitted_action={"shipment_id": "s1", "action_type": "WAIT"},
            tool_call_sequence=(("get_shipment_context", {}),),
        )
        seen: list[str] = []

        result = client.run_tool_loop(
            _request(), lambda name, args: (seen.append(name), {"ok": True})[1]
        )

        assert seen == ["get_shipment_context"]
        assert result.submitted_action == {"shipment_id": "s1", "action_type": "WAIT"}
        assert result.stop_reason == "submitted"
        assert len(result.tool_calls) == 1

    def test_default_has_no_tool_calls(self) -> None:
        client = FakeLLMClient()
        result = client.run_tool_loop(_request(), _echo_tool)
        assert result.tool_calls == ()
        assert result.submitted_action is None


class TestInteractionToDict:
    def test_round_trips_the_fields_replay_needs(self) -> None:
        client = FakeLLMClient(submitted_action={"shipment_id": "s1", "action_type": "WAIT"})
        result = client.run_tool_loop(_request(), _echo_tool)

        payload = interaction_to_dict(result, prompt_hash="deadbeef")

        assert payload["prompt_hash"] == "deadbeef"
        assert payload["decision_key"]["shipment_id"] == "shipment_001_001"  # type: ignore[index]
        assert payload["submitted_action"] == {"shipment_id": "s1", "action_type": "WAIT"}
        assert json.dumps(payload)  # fully JSON-serializable
