"""The shared contract every decision-maker implements, and its timing wrapper.

Inside the policies package, this module defines the `Policy` protocol both
the heuristic and the LLM agent structurally satisfy (a name and a `decide`
method that turns one read-only `DecisionObservation` into one
`DecisionAction`), plus `PolicyDecisionRecord`, which captures a proposed
action together with its timing and the policy that made it, without
executing anything. In the full system, this is what lets
simulation/engine.py invoke any policy identically and lets
experiments/runner.py measure and compare them fairly. It does not validate,
apply, or fall back on an action — those are decisions/validator.py's and
policies/fallback.py's jobs.

`make_decision_record` also reads an optional `last_interaction` attribute
off the policy after calling `decide()`, via `getattr(..., None)` rather than
an `isinstance` check. Only `policies/llm_agent.py`'s `LLMAgentPolicy` sets
this; every other policy simply doesn't define it, so the read is a no-op for
them. This is how `LLMInteractionResult` (CLAUDE.md section 27.6's
llm_interactions.jsonl) reaches simulation/engine.py without engine.py or
this module ever branching on a concrete policy type.
"""

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
