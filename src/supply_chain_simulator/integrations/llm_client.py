"""Implements LLMClient three ways: OpenAIResponsesClient (live API), ReplayLLMClient (replays a recorded interaction), FakeLLMClient (test double). The only file that knows the OpenAI wire format."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar

import openai

StopReason = Literal["submitted", "tool_limit_reached", "no_submission"]

_ResponseT = TypeVar("_ResponseT")

_TRANSIENT_ERRORS: tuple[type[Exception], ...] = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)
_RETRY_BACKOFF_SECONDS: tuple[float, ...] = (1.0, 2.0, 4.0)


class LLMIntegrationError(Exception):
    """Raised when the configured LLM provider cannot be built or reached.

    A live provider failure must fail the whole experiment rather than
    being silently swapped for a fallback policy, so this is allowed to
    propagate all the way to cli.py's top-level mapping.
    """


class ReplayTraceError(LLMIntegrationError):
    """Raised when a replay trace file is unreadable, malformed, has a
    duplicate decision_key, or has no recorded entry for a requested one.
    """


@dataclass(frozen=True, slots=True)
class DecisionKey:
    """The seven fields defining one LLM decision's identity."""

    experiment_id: str
    scenario_id: str
    replication: int
    run_kind: str
    day: int
    shipment_id: str
    observation_hash: str

    def as_tuple(self) -> tuple[str, str, int, str, int, str, str]:
        return (
            self.experiment_id,
            self.scenario_id,
            self.replication,
            self.run_kind,
            self.day,
            self.shipment_id,
            self.observation_hash,
        )


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """One approved tool's name, description, and strict JSON-schema parameters."""

    name: str
    description: str
    parameters_schema: dict[str, object]


@dataclass(frozen=True, slots=True)
class LLMToolLoopRequest:
    """Everything an `LLMClient` needs to run one bounded decision, and nothing
    about simulation state: the caller (policies/llm_agent.py) has already
    reduced one `DecisionObservation` down to a system prompt, a short user
    message, and a fixed set of tool schemas.
    """

    decision_key: DecisionKey
    system_prompt: str
    user_message: str
    tools: tuple[ToolSpec, ...]
    submit_tool_name: str
    model: str
    temperature: float
    max_output_tokens: int
    max_tool_calls: int
    request_timeout_seconds: int
    max_retries: int


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    tool_call_id: str
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True, slots=True)
class ToolOutputRecord:
    tool_call_id: str
    name: str
    output: dict[str, object]


@dataclass(frozen=True, slots=True)
class LLMInteractionResult:
    """The complete, secret-free audit trail of one decision's tool loop,
    matching every field llm_interactions.jsonl needs except `prompt_hash`,
    which only policies/llm_agent.py (the owner of the prompt constant) can
    supply.
    """

    decision_key: DecisionKey
    model: str
    request_without_secrets: dict[str, object]
    tool_calls: tuple[ToolCallRecord, ...]
    tool_outputs: tuple[ToolOutputRecord, ...]
    provider_response: dict[str, object] | None
    submitted_action: dict[str, object] | None
    stop_reason: StopReason
    token_usage: dict[str, int] | None
    latency_ms: float
    attempt_count: int


class LLMClient(Protocol):
    """Structural contract every LLM backend (live, replay, fake) implements."""

    def run_tool_loop(
        self,
        request: LLMToolLoopRequest,
        execute_tool: Callable[[str, dict[str, object]], dict[str, object]],
    ) -> LLMInteractionResult: ...


def decision_key_to_dict(key: DecisionKey) -> dict[str, object]:
    return {
        "experiment_id": key.experiment_id,
        "scenario_id": key.scenario_id,
        "replication": key.replication,
        "run_kind": key.run_kind,
        "day": key.day,
        "shipment_id": key.shipment_id,
        "observation_hash": key.observation_hash,
    }


def interaction_to_dict(result: LLMInteractionResult, prompt_hash: str) -> dict[str, object]:
    """Converts one interaction into the exact record shape
    `llm_interactions.jsonl` requires, so
    `data_io/writers.py:append_llm_interactions` can write it without
    knowing anything about `LLMInteractionResult` itself.
    """
    return {
        "decision_key": decision_key_to_dict(result.decision_key),
        "model": result.model,
        "prompt_hash": prompt_hash,
        "request_without_secrets": result.request_without_secrets,
        "tool_calls": [
            {"tool_call_id": call.tool_call_id, "name": call.name, "arguments": call.arguments}
            for call in result.tool_calls
        ],
        "tool_outputs": [
            {"tool_call_id": out.tool_call_id, "name": out.name, "output": out.output}
            for out in result.tool_outputs
        ],
        "provider_response": result.provider_response,
        "submitted_action": result.submitted_action,
        "token_usage": result.token_usage,
        "latency_ms": round(result.latency_ms, 6),
        "attempt_count": result.attempt_count,
    }


def _decision_key_from_dict(data: Mapping[str, object]) -> tuple[str, str, int, str, int, str, str]:
    return (
        str(data["experiment_id"]),
        str(data["scenario_id"]),
        int(data["replication"]),  # type: ignore[call-overload]
        str(data["run_kind"]),
        int(data["day"]),  # type: ignore[call-overload]
        str(data["shipment_id"]),
        str(data["observation_hash"]),
    )


def _tool_spec_to_param(tool: ToolSpec) -> dict[str, object]:
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters_schema,
        "strict": True,
    }


class OpenAIResponsesClient:
    """Live `LLMClient` backed by the OpenAI Responses API.

    `client` is accepted as an injectable dependency purely for testability
    (no test in this project makes a real API call); production code
    should only ever pass `api_key` and let this constructor
    build the real `openai.OpenAI` client.
    """

    def __init__(
        self,
        api_key: str,
        client: openai.OpenAI | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client if client is not None else openai.OpenAI(api_key=api_key)
        self._sleep = sleep

    def run_tool_loop(
        self,
        request: LLMToolLoopRequest,
        execute_tool: Callable[[str, dict[str, object]], dict[str, object]],
    ) -> LLMInteractionResult:
        start = time.perf_counter()
        tools_param = [_tool_spec_to_param(tool) for tool in request.tools]
        conversation: list[dict[str, object]] = [
            {"role": "user", "content": request.user_message}
        ]
        request_without_secrets: dict[str, object] = {
            "model": request.model,
            "instructions": request.system_prompt,
            "input": list(conversation),
            "tools": tools_param,
            "temperature": request.temperature,
            "max_output_tokens": request.max_output_tokens,
            "max_tool_calls": request.max_tool_calls,
            "store": False,
        }

        tool_calls: list[ToolCallRecord] = []
        tool_outputs: list[ToolOutputRecord] = []
        total_attempts = 0
        token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        last_raw_response: dict[str, object] | None = None
        submitted_action: dict[str, object] | None = None
        stop_reason: StopReason = "no_submission"

        while len(tool_calls) < request.max_tool_calls:
            response, attempts = self._call_with_retries(
                lambda: self._client.responses.create(
                    model=request.model,
                    instructions=request.system_prompt,
                    input=conversation,  # type: ignore[arg-type]
                    tools=tools_param,  # type: ignore[arg-type]
                    temperature=request.temperature,
                    max_output_tokens=request.max_output_tokens,
                    store=False,
                    timeout=request.request_timeout_seconds,
                ),
                request.max_retries,
            )
            total_attempts += attempts
            last_raw_response = response.model_dump(mode="json")
            _accumulate_usage(token_usage, response.usage)

            function_calls = [item for item in response.output if item.type == "function_call"]
            if not function_calls:
                stop_reason = "no_submission"
                break

            submitted = False
            for call in function_calls:
                if len(tool_calls) >= request.max_tool_calls:
                    stop_reason = "tool_limit_reached"
                    break

                arguments = json.loads(call.arguments)
                conversation.append(
                    {
                        "type": "function_call",
                        "call_id": call.call_id,
                        "name": call.name,
                        "arguments": call.arguments,
                    }
                )
                tool_calls.append(
                    ToolCallRecord(tool_call_id=call.call_id, name=call.name, arguments=arguments)
                )

                if call.name == request.submit_tool_name:
                    submitted_action = arguments
                    output: dict[str, object] = {"acknowledged": True}
                    submitted = True
                else:
                    output = execute_tool(call.name, arguments)

                tool_outputs.append(
                    ToolOutputRecord(tool_call_id=call.call_id, name=call.name, output=output)
                )
                conversation.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(output),
                    }
                )

                if submitted:
                    stop_reason = "submitted"
                    break

            if submitted or stop_reason == "tool_limit_reached":
                break
        else:
            stop_reason = "tool_limit_reached"

        latency_ms = (time.perf_counter() - start) * 1000.0
        return LLMInteractionResult(
            decision_key=request.decision_key,
            model=request.model,
            request_without_secrets=request_without_secrets,
            tool_calls=tuple(tool_calls),
            tool_outputs=tuple(tool_outputs),
            provider_response=last_raw_response,
            submitted_action=submitted_action,
            stop_reason=stop_reason,
            token_usage=token_usage,
            latency_ms=latency_ms,
            attempt_count=total_attempts,
        )

    def _call_with_retries(
        self, make_request: Callable[[], _ResponseT], max_retries: int
    ) -> tuple[_ResponseT, int]:
        """Retries transient errors 3x with 1s/2s/4s backoff; never retries
        invalid-request/auth-style errors.
        """
        attempt = 0
        while True:
            try:
                return make_request(), attempt + 1
            except _TRANSIENT_ERRORS as exc:
                if attempt >= max_retries:
                    raise LLMIntegrationError(
                        f"OpenAI request failed after {attempt + 1} attempt(s): {exc}"
                    ) from exc
                wait_seconds = _RETRY_BACKOFF_SECONDS[
                    min(attempt, len(_RETRY_BACKOFF_SECONDS) - 1)
                ]
                self._sleep(wait_seconds)
                attempt += 1
            except openai.OpenAIError as exc:
                raise LLMIntegrationError(f"OpenAI request failed: {exc}") from exc


def _accumulate_usage(totals: dict[str, int], usage: object) -> None:
    if usage is None:
        return
    totals["input_tokens"] += getattr(usage, "input_tokens", 0) or 0
    totals["output_tokens"] += getattr(usage, "output_tokens", 0) or 0
    totals["total_tokens"] += getattr(usage, "total_tokens", 0) or 0


class ReplayLLMClient:
    """Reproduces previously-recorded LLM decisions with no network call:
    loads an entire `llm_interactions.jsonl` at construction time, indexes
    it by `DecisionKey`, and returns the exact recorded interaction for a
    matching request.
    """

    def __init__(self, trace_path: Path) -> None:
        self._trace_path = trace_path
        self._recorded: dict[tuple[str, str, int, str, int, str, str], dict[str, Any]] = {}

        try:
            raw_text = trace_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ReplayTraceError(f"cannot read replay trace file {trace_path}: {exc}") from exc

        for line_number, line in enumerate(raw_text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReplayTraceError(
                    f"{trace_path}:{line_number}: invalid JSON in replay trace: {exc}"
                ) from exc
            key = _decision_key_from_dict(entry["decision_key"])
            if key in self._recorded:
                raise ReplayTraceError(f"{trace_path}:{line_number}: duplicate decision_key {key}")
            self._recorded[key] = entry

    def run_tool_loop(
        self,
        request: LLMToolLoopRequest,
        execute_tool: Callable[[str, dict[str, object]], dict[str, object]],
    ) -> LLMInteractionResult:
        del execute_tool  # replay never re-executes tools; it returns the recorded outcome
        key = request.decision_key.as_tuple()
        entry = self._recorded.get(key)
        if entry is None:
            raise ReplayTraceError(
                f"no recorded interaction for decision_key {key} in {self._trace_path}"
            )

        submitted_action = entry.get("submitted_action")
        tool_calls = tuple(
            ToolCallRecord(tool_call_id=call["tool_call_id"], name=call["name"], arguments=call["arguments"])
            for call in entry.get("tool_calls", [])
        )
        tool_outputs = tuple(
            ToolOutputRecord(tool_call_id=out["tool_call_id"], name=out["name"], output=out["output"])
            for out in entry.get("tool_outputs", [])
        )
        return LLMInteractionResult(
            decision_key=request.decision_key,
            model=str(entry.get("model", request.model)),
            request_without_secrets=dict(entry.get("request_without_secrets") or {}),
            tool_calls=tool_calls,
            tool_outputs=tool_outputs,
            provider_response=entry.get("provider_response"),
            submitted_action=submitted_action,
            stop_reason="submitted" if submitted_action is not None else "no_submission",
            token_usage=entry.get("token_usage"),
            latency_ms=0.0,
            attempt_count=0,
        )


class FakeLLMClient:
    """Test-only `LLMClient`: returns a pre-programmed result, no network
    call and no OpenAI dependency involved.
    """

    def __init__(
        self,
        submitted_action: dict[str, object] | None = None,
        stop_reason: StopReason = "submitted",
        tool_call_sequence: Sequence[tuple[str, dict[str, object]]] = (),
    ) -> None:
        self._submitted_action = submitted_action
        self._stop_reason = stop_reason
        self._tool_call_sequence = tuple(tool_call_sequence)

    def run_tool_loop(
        self,
        request: LLMToolLoopRequest,
        execute_tool: Callable[[str, dict[str, object]], dict[str, object]],
    ) -> LLMInteractionResult:
        tool_calls: list[ToolCallRecord] = []
        tool_outputs: list[ToolOutputRecord] = []
        for index, (name, arguments) in enumerate(self._tool_call_sequence):
            call_id = f"fake_call_{index}"
            tool_calls.append(ToolCallRecord(tool_call_id=call_id, name=name, arguments=arguments))
            output = execute_tool(name, arguments)
            tool_outputs.append(ToolOutputRecord(tool_call_id=call_id, name=name, output=output))

        return LLMInteractionResult(
            decision_key=request.decision_key,
            model=request.model,
            request_without_secrets={
                "model": request.model,
                "system_prompt": request.system_prompt,
                "user_message": request.user_message,
            },
            tool_calls=tuple(tool_calls),
            tool_outputs=tuple(tool_outputs),
            provider_response=None,
            submitted_action=self._submitted_action,
            stop_reason=self._stop_reason,
            token_usage=None,
            latency_ms=0.0,
            attempt_count=0,
        )
