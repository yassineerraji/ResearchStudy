"""The transparent, deterministic classical benchmark policy.

Inside the policies package, this module implements the exact rule CLAUDE.md
sections 11.13 and 22 specify: build a WAIT candidate from the current plan, a
REROUTE candidate for every non-emergency route option, and an EXPEDITE
candidate for every emergency route option (kept only when the current plan
is already at least `expedite_trigger_lateness_days` late, or when no
non-emergency route exists), then pick the lowest estimated-total-cost
candidate, tie-breaking WAIT before REROUTE before EXPEDITE and then by the
lexicographically smallest route ID. In the full system, this is the
research question's classical baseline, held to the same observation and
validation rules as the LLM agent. It never looks at anything beyond the
observation it is given and never mutates simulation state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from supply_chain_simulator.decisions.observation import (
    DecisionObservation,
    RouteOption,
)
from supply_chain_simulator.domain.actions import ActionType, DecisionAction, ReasonCode

_ACTION_PRIORITY: dict[ActionType, int] = {
    ActionType.WAIT: 0,
    ActionType.REROUTE: 1,
    ActionType.EXPEDITE: 2,
}


@dataclass(frozen=True, slots=True)
class _Candidate:
    action_type: ActionType
    route_id: str | None
    estimated_total_cost: float
    reason_code: ReasonCode


def _cost_or_infinite(estimated_total_cost: float | None) -> float:
    return math.inf if estimated_total_cost is None else estimated_total_cost


def _tie_break_key(candidate: _Candidate) -> tuple[int, str]:
    return (_ACTION_PRIORITY[candidate.action_type], candidate.route_id or "")


class HeuristicPolicy:
    def __init__(
        self, expedite_trigger_lateness_days: int, cost_tolerance: float
    ) -> None:
        self._expedite_trigger_lateness_days = expedite_trigger_lateness_days
        self._cost_tolerance = cost_tolerance

    @property
    def name(self) -> str:
        return "heuristic"

    def decide(self, observation: DecisionObservation) -> DecisionAction:
        candidates = self._build_candidates(observation)
        chosen = self._select(candidates)
        return DecisionAction(
            shipment_id=observation.shipment.shipment_id,
            action_type=chosen.action_type,
            route_id=chosen.route_id,
            reason_code=chosen.reason_code,
            rationale=self._rationale(chosen),
        )

    def _build_candidates(self, observation: DecisionObservation) -> list[_Candidate]:
        route_options = observation.route_options
        wait_reason = (
            ReasonCode.NO_FEASIBLE_ALTERNATIVE
            if not route_options
            else ReasonCode.LOWER_ESTIMATED_COST
        )
        candidates = [
            _Candidate(
                action_type=ActionType.WAIT,
                route_id=None,
                estimated_total_cost=_cost_or_infinite(
                    observation.current_plan.estimated_total_cost
                ),
                reason_code=wait_reason,
            )
        ]

        reroute_options = [
            option for option in route_options if not option.contains_emergency_edge
        ]
        expedite_options = [
            option for option in route_options if option.contains_emergency_edge
        ]

        candidates.extend(
            _Candidate(
                action_type=ActionType.REROUTE,
                route_id=option.route_id,
                estimated_total_cost=_cost_or_infinite(option.estimated_total_cost),
                reason_code=ReasonCode.LOWER_ESTIMATED_COST,
            )
            for option in reroute_options
        )

        if self._expedite_is_eligible(observation, reroute_options):
            expedite_reason = (
                ReasonCode.REDUCE_LATENESS
                if self._wait_lateness_triggers_expedite(observation)
                else ReasonCode.NO_FEASIBLE_ALTERNATIVE
            )
            candidates.extend(
                _Candidate(
                    action_type=ActionType.EXPEDITE,
                    route_id=option.route_id,
                    estimated_total_cost=_cost_or_infinite(option.estimated_total_cost),
                    reason_code=expedite_reason,
                )
                for option in expedite_options
            )

        return candidates

    def _wait_lateness_triggers_expedite(
        self, observation: DecisionObservation
    ) -> bool:
        lateness = observation.current_plan.estimated_lateness_days
        return lateness is not None and lateness >= self._expedite_trigger_lateness_days

    def _expedite_is_eligible(
        self, observation: DecisionObservation, reroute_options: list[RouteOption]
    ) -> bool:
        return self._wait_lateness_triggers_expedite(observation) or not reroute_options

    def _select(self, candidates: list[_Candidate]) -> _Candidate:
        best = candidates[0]
        for candidate in candidates[1:]:
            if (
                candidate.estimated_total_cost
                < best.estimated_total_cost - self._cost_tolerance
                or (
                    abs(candidate.estimated_total_cost - best.estimated_total_cost)
                    <= self._cost_tolerance
                    and _tie_break_key(candidate) < _tie_break_key(best)
                )
            ):
                best = candidate
        return best

    def _rationale(self, candidate: _Candidate) -> str:
        if candidate.action_type is ActionType.WAIT:
            return "lowest estimated cost among eligible actions is to wait on the current plan"
        cost_text = (
            "unknown"
            if math.isinf(candidate.estimated_total_cost)
            else f"{candidate.estimated_total_cost:.2f}"
        )
        return (
            f"{candidate.action_type.value} via {candidate.route_id} has the lowest "
            f"estimated total cost ({cost_text})"
        )
