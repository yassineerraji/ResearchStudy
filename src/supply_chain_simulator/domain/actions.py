"""Policy actions and the shared record of whether one was accepted.

Inside the domain package, this module defines the four-action vocabulary
every policy shares (WAIT, REROUTE, EXPEDITE, ABSTAIN), the structured
DecisionAction a policy returns, and the ValidationCode/ValidationResult pair
the shared validator uses to accept or reject it. In the full system, this is
the common contract that makes the heuristic and the LLM agent comparable:
both produce exactly the same action type, and both are held to the same
validation outcomes. It does not implement validation logic itself (that is
decisions/validator.py) and does not execute actions against simulation
state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

MAX_RATIONALE_LENGTH = 300


class ActionType(Enum):
    """The four actions every policy may choose between."""

    WAIT = "WAIT"
    REROUTE = "REROUTE"
    EXPEDITE = "EXPEDITE"
    ABSTAIN = "ABSTAIN"


class ReasonCode(Enum):
    """Audit reason attached to a proposed or fallback action."""

    CURRENT_ROUTE_BLOCKED = "CURRENT_ROUTE_BLOCKED"
    REDUCE_LATENESS = "REDUCE_LATENESS"
    LOWER_ESTIMATED_COST = "LOWER_ESTIMATED_COST"
    NO_FEASIBLE_ALTERNATIVE = "NO_FEASIBLE_ALTERNATIVE"
    INSUFFICIENT_INFORMATION = "INSUFFICIENT_INFORMATION"
    POLICY_OUTPUT_INVALID = "POLICY_OUTPUT_INVALID"
    TOOL_LIMIT_REACHED = "TOOL_LIMIT_REACHED"


@dataclass(frozen=True, slots=True)
class DecisionAction:
    shipment_id: str
    action_type: ActionType
    route_id: str | None
    reason_code: ReasonCode
    rationale: str

    def __post_init__(self) -> None:
        if self.action_type in (ActionType.WAIT, ActionType.ABSTAIN) and self.route_id is not None:
            raise ValueError(f"{self.action_type.value} requires route_id=None")
        if (
            self.action_type in (ActionType.REROUTE, ActionType.EXPEDITE)
            and self.route_id is None
        ):
            raise ValueError(f"{self.action_type.value} requires a route_id")
        if len(self.rationale) > MAX_RATIONALE_LENGTH:
            raise ValueError(f"rationale exceeds {MAX_RATIONALE_LENGTH} characters")


class ValidationCode(Enum):
    """Every possible outcome of validating a proposed DecisionAction."""

    VALID = "VALID"
    SHIPMENT_NOT_FOUND = "SHIPMENT_NOT_FOUND"
    SHIPMENT_NOT_AT_NODE = "SHIPMENT_NOT_AT_NODE"
    SHIPMENT_ALREADY_DELIVERED = "SHIPMENT_ALREADY_DELIVERED"
    ACTION_SHIPMENT_MISMATCH = "ACTION_SHIPMENT_MISMATCH"
    ROUTE_REQUIRED = "ROUTE_REQUIRED"
    ROUTE_NOT_ALLOWED = "ROUTE_NOT_ALLOWED"
    ROUTE_NOT_FOUND = "ROUTE_NOT_FOUND"
    ROUTE_WRONG_ORIGIN = "ROUTE_WRONG_ORIGIN"
    ROUTE_WRONG_DESTINATION = "ROUTE_WRONG_DESTINATION"
    ROUTE_USES_UNAVAILABLE_COMPONENT = "ROUTE_USES_UNAVAILABLE_COMPONENT"
    ROUTE_STATIC_CAPACITY_TOO_SMALL = "ROUTE_STATIC_CAPACITY_TOO_SMALL"
    REROUTE_USES_EMERGENCY_EDGE = "REROUTE_USES_EMERGENCY_EDGE"
    EXPEDITE_HAS_NO_EMERGENCY_EDGE = "EXPEDITE_HAS_NO_EMERGENCY_EDGE"
    INVALID_ACTION_SCHEMA = "INVALID_ACTION_SCHEMA"


@dataclass(frozen=True, slots=True)
class ValidationResult:
    code: ValidationCode
    detail: str

    @property
    def is_valid(self) -> bool:
        return self.code is ValidationCode.VALID
