"""Provides WaitFallbackPolicy, HeuristicFallbackPolicy, and resolve_action: validate, then fallback, then terminal WAIT, keeping an invalid or abstaining proposal from ever reaching transition.py."""

from __future__ import annotations

from dataclasses import dataclass

from supply_chain_simulator.decisions.observation import DecisionObservation
from supply_chain_simulator.decisions.validator import validate_action
from supply_chain_simulator.domain.actions import (
    ActionType,
    DecisionAction,
    ReasonCode,
    ValidationResult,
)
from supply_chain_simulator.domain.state import SimulationState
from supply_chain_simulator.policies.base import Policy
from supply_chain_simulator.policies.heuristic import HeuristicPolicy


class WaitFallbackPolicy:
    """Unconditionally proposes WAIT; always valid for any non-delivered, AT_NODE shipment."""

    @property
    def name(self) -> str:
        return "wait_fallback"

    def decide(self, observation: DecisionObservation) -> DecisionAction:
        return DecisionAction(
            shipment_id=observation.shipment.shipment_id,
            action_type=ActionType.WAIT,
            route_id=None,
            reason_code=ReasonCode.NO_FEASIBLE_ALTERNATIVE,
            rationale="fallback policy: unconditional wait",
        )


class HeuristicFallbackPolicy:
    """Delegates to a wrapped HeuristicPolicy, distinguished by name for audit."""

    def __init__(self, heuristic: HeuristicPolicy) -> None:
        self._heuristic = heuristic

    @property
    def name(self) -> str:
        return "heuristic_fallback"

    def decide(self, observation: DecisionObservation) -> DecisionAction:
        return self._heuristic.decide(observation)


@dataclass(frozen=True, slots=True)
class FallbackResolution:
    proposed_action: DecisionAction
    proposal_validation: ValidationResult
    fallback_invoked: bool
    fallback_action: DecisionAction | None
    fallback_validation: ValidationResult | None
    executed_action: DecisionAction


def _terminal_safe_wait(shipment_id: str) -> DecisionAction:
    return DecisionAction(
        shipment_id=shipment_id,
        action_type=ActionType.WAIT,
        route_id=None,
        reason_code=ReasonCode.POLICY_OUTPUT_INVALID,
        rationale="terminal safe WAIT: fallback action was also invalid",
    )


def resolve_action(
    proposed_action: DecisionAction,
    observation: DecisionObservation,
    state: SimulationState,
    fallback_policy: Policy,
) -> FallbackResolution:
    """Runs CLAUDE.md section 11.14's fallback chain for one shipment's decision."""
    proposal_validation = validate_action(proposed_action, observation, state)
    needs_fallback = (
        proposed_action.action_type is ActionType.ABSTAIN
        or not proposal_validation.is_valid
    )

    if not needs_fallback:
        return FallbackResolution(
            proposed_action=proposed_action,
            proposal_validation=proposal_validation,
            fallback_invoked=False,
            fallback_action=None,
            fallback_validation=None,
            executed_action=proposed_action,
        )

    fallback_action = fallback_policy.decide(observation)
    fallback_validation = validate_action(fallback_action, observation, state)
    if (
        fallback_validation.is_valid
        and fallback_action.action_type is not ActionType.ABSTAIN
    ):
        executed_action = fallback_action
    else:
        executed_action = _terminal_safe_wait(proposed_action.shipment_id)

    return FallbackResolution(
        proposed_action=proposed_action,
        proposal_validation=proposal_validation,
        fallback_invoked=True,
        fallback_action=fallback_action,
        fallback_validation=fallback_validation,
        executed_action=executed_action,
    )
