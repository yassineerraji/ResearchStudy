"""The bounded LLM decision agent: one shipment, five tools, one submission.

Inside the policies package, this module implements CLAUDE.md sections 11.15
and 23: a versioned system prompt, the five approved read-only/submission
tools (each a deterministic local view over one immutable
`DecisionObservation`, never simulation state), and `LLMAgentPolicy`, which
turns one observation into one `DecisionAction` by running an `LLMClient`'s
tool loop and interpreting whatever it returns. In the full system, this is
the research question's other half — the same `Policy` protocol the
heuristic implements, so it is held to exactly the same observation,
validation, and fallback rules. It never calls a provider directly (that is
integrations/llm_client.py's job) and never repairs an invalid submission —
an unparseable or infeasible one becomes ABSTAIN and flows into the same
fallback chain every policy uses.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable

from supply_chain_simulator.decisions.observation import (
    DecisionObservation,
    compute_observation_hash,
    destination_context_to_dict,
    route_option_to_dict,
    shipment_context_to_dict,
)
from supply_chain_simulator.domain.actions import (
    MAX_RATIONALE_LENGTH,
    ActionType,
    DecisionAction,
    ReasonCode,
)
from supply_chain_simulator.integrations.llm_client import (
    DecisionKey,
    LLMClient,
    LLMInteractionResult,
    LLMToolLoopRequest,
    ToolSpec,
)
from supply_chain_simulator.simulation.engine import RunIdentity

_SUBMIT_ACTION_TOOL_NAME = "submit_action"

SYSTEM_PROMPT = """You are a bounded supply-chain disruption-response agent. Your objective is \
to choose the action for one shipment that minimizes the total cost of disruption, while \
respecting the physical and operational limits described by the tools available to you.

Rules you must follow:
- Use only the facts you retrieve through your tools. Never invent a route, a capacity, a \
cost, or a shock that no tool reported to you.
- You control exactly one shipment and must choose exactly one action: WAIT, REROUTE, \
EXPEDITE, or ABSTAIN.
- REROUTE and EXPEDITE must use a route_id that list_route_options or inspect_route actually \
returned to you. Never invent a route_id.
- action_type must match that route's contains_emergency_edge flag exactly: choose REROUTE \
only for a route where contains_emergency_edge is false, and EXPEDITE only for a route where \
contains_emergency_edge is true. Never submit EXPEDITE for a normal route, and never submit \
REROUTE for an emergency route.
- If the information available to you is insufficient to decide with confidence, choose \
ABSTAIN rather than guessing.
- You cannot modify the simulation, create shipments, change demand, or repair an unavailable \
route. You may only propose one action for the shipment you were given.
- Submit your final decision by calling submit_action exactly once, as your last tool call.
- Do not reveal your private reasoning. Your rationale must be a short, factual audit note \
(at most 300 characters), never a chain-of-thought explanation.
"""


def compute_prompt_hash() -> str:
    """CLAUDE.md section 23.6: a SHA-256 of the versioned system prompt, written
    to the manifest so any prompt change is visible as a policy change.
    """
    return hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()


_NO_PARAMS_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {},
    "required": [],
    "additionalProperties": False,
}

_TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="get_shipment_context",
        description=(
            "Returns the triggered shipment's own facts: id, product, quantity, current "
            "node, destination, release/due day, days until due, capacity wait days, and "
            "its remaining planned route edges."
        ),
        parameters_schema=_NO_PARAMS_SCHEMA,
    ),
    ToolSpec(
        name="get_destination_context",
        description=(
            "Returns the demand destination's current inventory on hand, backlog, mean "
            "daily demand, and days of supply."
        ),
        parameters_schema=_NO_PARAMS_SCHEMA,
    ),
    ToolSpec(
        name="list_route_options",
        description=(
            "Returns a short summary of every approved candidate route: route_id, "
            "estimated total cost, estimated arrival day, whether it uses an emergency "
            "lane, and whether it can dispatch today."
        ),
        parameters_schema=_NO_PARAMS_SCHEMA,
    ),
    ToolSpec(
        name="inspect_route",
        description=(
            "Returns full cost and lead-time detail for one approved route_id (as "
            "returned by list_route_options)."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "route_id": {
                    "type": "string",
                    "description": "One of the route_ids returned by list_route_options.",
                }
            },
            "required": ["route_id"],
            "additionalProperties": False,
        },
    ),
    ToolSpec(
        name=_SUBMIT_ACTION_TOOL_NAME,
        description=(
            "Submits the final decision for this shipment. Call exactly once, as your "
            "last tool call."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "shipment_id": {"type": "string"},
                "action_type": {
                    "type": "string",
                    "enum": [action_type.value for action_type in ActionType],
                    "description": (
                        "REROUTE requires a route_id whose contains_emergency_edge is "
                        "false; EXPEDITE requires a route_id whose contains_emergency_edge "
                        "is true. Never mix these up."
                    ),
                },
                "route_id": {
                    "type": ["string", "null"],
                    "description": (
                        "Required (non-null) for REROUTE/EXPEDITE, using an approved "
                        "route_id whose contains_emergency_edge flag matches action_type; "
                        "must be null for WAIT/ABSTAIN."
                    ),
                },
                "reason_code": {
                    "type": "string",
                    "enum": [reason_code.value for reason_code in ReasonCode],
                },
                "rationale": {
                    "type": "string",
                    "maxLength": MAX_RATIONALE_LENGTH,
                    "description": "A short factual audit note, never chain-of-thought.",
                },
            },
            "required": ["shipment_id", "action_type", "route_id", "reason_code", "rationale"],
            "additionalProperties": False,
        },
    ),
)


def _build_user_message(observation: DecisionObservation) -> str:
    shipment = observation.shipment
    return (
        f"Observation {observation.observation_id} (day {observation.day}). "
        f"Decide the action for shipment {shipment.shipment_id}, currently at "
        f"{shipment.current_node_id}, destination {shipment.destination_node_id}, "
        f"due day {shipment.due_day} ({shipment.days_until_due} days from now). "
        "Use your tools to gather the facts you need, then call submit_action."
    )


def _list_route_options(observation: DecisionObservation) -> dict[str, object]:
    return {
        "route_options": [
            {
                "route_id": option.route_id,
                "estimated_total_cost": option.estimated_total_cost,
                "estimated_arrival_day": option.estimated_arrival_day,
                "contains_emergency_edge": option.contains_emergency_edge,
                "currently_dispatchable": option.currently_dispatchable,
            }
            for option in observation.route_options
        ]
    }


def _inspect_route(observation: DecisionObservation, arguments: dict[str, object]) -> dict[str, object]:
    route_id = arguments.get("route_id")
    for option in observation.route_options:
        if option.route_id == route_id:
            return route_option_to_dict(option)
    return {"error": f"route_id {route_id!r} is not one of the approved route options"}


def make_tool_executor(
    observation: DecisionObservation,
) -> Callable[[str, dict[str, object]], dict[str, object]]:
    """Builds the local, deterministic, read-only tool dispatch for one
    observation. `submit_action` is deliberately absent: the LLMClient
    handles it directly as the loop's terminal tool and never routes it
    through this executor (CLAUDE.md section 11.15/11.16).
    """
    executors: dict[str, Callable[[dict[str, object]], dict[str, object]]] = {
        "get_shipment_context": lambda _args: shipment_context_to_dict(observation.shipment),
        "get_destination_context": lambda _args: destination_context_to_dict(
            observation.destination
        ),
        "list_route_options": lambda _args: _list_route_options(observation),
        "inspect_route": lambda args: _inspect_route(observation, args),
    }

    def execute(name: str, arguments: dict[str, object]) -> dict[str, object]:
        executor = executors.get(name)
        if executor is None:
            return {"error": f"unknown tool {name!r}"}
        return executor(arguments)

    return execute


def _abstain(shipment_id: str, reason_code: ReasonCode, rationale: str) -> DecisionAction:
    return DecisionAction(
        shipment_id=shipment_id,
        action_type=ActionType.ABSTAIN,
        route_id=None,
        reason_code=reason_code,
        rationale=rationale[:MAX_RATIONALE_LENGTH],
    )


def _interpret_result(shipment_id: str, result: LLMInteractionResult) -> DecisionAction:
    if result.stop_reason == "tool_limit_reached":
        return _abstain(
            shipment_id,
            ReasonCode.TOOL_LIMIT_REACHED,
            "tool-call budget exhausted before submission",
        )
    submitted = result.submitted_action
    if result.stop_reason == "no_submission" or submitted is None:
        return _abstain(
            shipment_id,
            ReasonCode.POLICY_OUTPUT_INVALID,
            "model produced no action submission",
        )

    try:
        raw_route_id = submitted["route_id"]
        route_id = str(raw_route_id) if raw_route_id is not None else None
        return DecisionAction(
            shipment_id=str(submitted["shipment_id"]),
            action_type=ActionType(submitted["action_type"]),
            route_id=route_id,
            reason_code=ReasonCode(submitted["reason_code"]),
            rationale=str(submitted["rationale"]),
        )
    except (KeyError, ValueError, TypeError) as exc:
        return _abstain(
            shipment_id,
            ReasonCode.POLICY_OUTPUT_INVALID,
            f"submitted action failed to parse: {exc}",
        )


class LLMAgentPolicy:
    """Bounded LLM policy: builds one tool-loop request per observation and
    interprets whatever the configured `LLMClient` returns into a
    `DecisionAction`. Never validates or falls back itself — that is shared
    with every other policy via decisions/validator.py and
    policies/fallback.py.
    """

    def __init__(
        self,
        client: LLMClient,
        model: str,
        temperature: float,
        max_tool_calls: int,
        max_output_tokens: int,
        request_timeout_seconds: int,
        max_retries: int,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_tool_calls = max_tool_calls
        self._max_output_tokens = max_output_tokens
        self._request_timeout_seconds = request_timeout_seconds
        self._max_retries = max_retries
        self._run_identity: RunIdentity | None = None
        self.last_interaction: LLMInteractionResult | None = None

    @property
    def name(self) -> str:
        return "llm_agent"

    def configure_run_context(self, run_identity: RunIdentity) -> None:
        """Called once per run by simulation/engine.py (a duck-typed, opt-in
        hook — see engine.py's docstring) so every decision's DecisionKey can
        include experiment_id/scenario_id/replication/run_kind, which the
        observation alone does not carry.
        """
        self._run_identity = run_identity

    def decide(self, observation: DecisionObservation) -> DecisionAction:
        if self._run_identity is None:
            raise ValueError(
                "LLMAgentPolicy.decide called before configure_run_context; "
                "SimulationEngine.run must configure run context before the day loop"
            )

        decision_key = DecisionKey(
            experiment_id=self._run_identity.experiment_id,
            scenario_id=self._run_identity.scenario_id,
            replication=self._run_identity.replication,
            run_kind=self._run_identity.run_kind,
            day=observation.day,
            shipment_id=observation.shipment.shipment_id,
            observation_hash=compute_observation_hash(observation),
        )
        request = LLMToolLoopRequest(
            decision_key=decision_key,
            system_prompt=SYSTEM_PROMPT,
            user_message=_build_user_message(observation),
            tools=_TOOL_SPECS,
            submit_tool_name=_SUBMIT_ACTION_TOOL_NAME,
            model=self._model,
            temperature=self._temperature,
            max_output_tokens=self._max_output_tokens,
            max_tool_calls=self._max_tool_calls,
            request_timeout_seconds=self._request_timeout_seconds,
            max_retries=self._max_retries,
        )
        result = self._client.run_tool_loop(request, make_tool_executor(observation))
        self.last_interaction = result
        return _interpret_result(observation.shipment.shipment_id, result)
