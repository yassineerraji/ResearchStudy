"""Defines the Policy protocol (name and decide()) both policies satisfy, and PolicyDecisionRecord, which times and records a proposed action without executing it."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from supply_chain_simulator.decisions.observation import DecisionObservation
from supply_chain_simulator.domain.actions import DecisionAction
from supply_chain_simulator.integrations.llm_client import LLMInteractionResult


class Policy(Protocol):
    """Structural contract every policy (heuristic, LLM agent, fallback) implements."""

    @property
    def name(self) -> str: ...

    def decide(self, observation: DecisionObservation) -> DecisionAction: ...


@dataclass(frozen=True, slots=True)
class PolicyDecisionRecord:
    policy_name: str
    observation_id: str
    proposed_action: DecisionAction
    decision_latency_ms: float
    llm_interaction: LLMInteractionResult | None = None


def make_decision_record(
    policy: Policy, observation: DecisionObservation
) -> PolicyDecisionRecord:
    """Calls `policy.decide(observation)` once, timing it for the audit trail."""
    start = time.perf_counter()
    action = policy.decide(observation)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return PolicyDecisionRecord(
        policy_name=policy.name,
        observation_id=observation.observation_id,
        proposed_action=action,
        decision_latency_ms=elapsed_ms,
        llm_interaction=getattr(policy, "last_interaction", None),
    )
