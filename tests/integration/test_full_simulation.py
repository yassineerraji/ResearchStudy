"""Drives the engine end to end over a tiny, hand-calculable fixture (undisrupted, disrupted, and with the real heuristic wired in) and checks every number by hand."""

from __future__ import annotations

from pathlib import Path

import pytest

from supply_chain_simulator.data_io.loaders import (
    NetworkConfig,
    ScenarioConfig,
    ShockConfig,
    build_initial_state,
    build_network_definition,
    load_network_config,
    load_scenario_config,
)
from supply_chain_simulator.decisions.observation import DecisionObservation
from supply_chain_simulator.domain.actions import ActionType, DecisionAction, ReasonCode
from supply_chain_simulator.domain.events import (
    DayEvents,
    EventTape,
    ShipmentReleaseEvent,
    Shock,
    ShockType,
    TargetType,
)
from supply_chain_simulator.domain.models import NetworkDefinition
from supply_chain_simulator.domain.state import (
    ShipmentStatus,
    SimulationResult,
    SimulationState,
)
from supply_chain_simulator.experiments.event_tape import (
    build_disrupted_event_tape,
    build_undisrupted_event_tape,
)
from supply_chain_simulator.policies.base import Policy
from supply_chain_simulator.policies.fallback import WaitFallbackPolicy
from supply_chain_simulator.policies.heuristic import HeuristicPolicy
from supply_chain_simulator.simulation.engine import (
    DecisionTraceEntry,
    RunIdentity,
    SimulationEngine,
)
from supply_chain_simulator.simulation.transition import SimulationInvariantError

REPO_ROOT = Path(__file__).resolve().parents[2]
TINY_NETWORK_CONFIG = REPO_ROOT / "tests/fixtures/tiny_network.yaml"
TINY_SCENARIO_CONFIG = REPO_ROOT / "tests/fixtures/tiny_scenario.yaml"

HORIZON_DAYS = 3
DRAIN_DAYS = 2
REROUTE_COST_PER_UNIT = 1.0
EXPEDITE_PREMIUM_PER_UNIT = 2.0
TERMINAL_PENALTY_DAYS = 30


def _run_identity(run_kind: str) -> RunIdentity:
    return RunIdentity(
        experiment_id="tiny_experiment",
        scenario_id="tiny_edge_closure",
        replication=1,
        policy_name="none_decisions_disabled",
        run_kind=run_kind,
    )


def _build_day_zero() -> tuple[NetworkConfig, NetworkDefinition, SimulationState]:
    network_config = load_network_config(TINY_NETWORK_CONFIG)
    network_definition = build_network_definition(network_config)
    state = build_initial_state(network_definition, network_config)
    return network_config, network_definition, state


class TestFullSimulationUndisrupted:
    def _run(self) -> SimulationResult:
        network_config, network_definition, initial_state = _build_day_zero()
        scenario_config = load_scenario_config(TINY_SCENARIO_CONFIG)
        disrupted_tape = build_disrupted_event_tape(
            network_definition=network_definition,
            demand_process=network_config.demand_process,
            replenishment_plan=network_config.replenishment_plan,
            scenario_config=scenario_config,
            replication=1,
            base_seed=42,
            horizon_days=HORIZON_DAYS,
            drain_days=DRAIN_DAYS,
        )
        undisrupted_tape = build_undisrupted_event_tape(disrupted_tape)

        return SimulationEngine().run(
            initial_state=initial_state,
            event_tape=undisrupted_tape,
            start_day=1,
            horizon_day=HORIZON_DAYS,
            drain_days=DRAIN_DAYS,
            decision_enabled=False,
            run_identity=_run_identity("UNDISRUPTED"),
            reroute_cost_per_unit=REROUTE_COST_PER_UNIT,
            expedite_premium_per_unit=EXPEDITE_PREMIUM_PER_UNIT,
            terminal_penalty_days=TERMINAL_PENALTY_DAYS,
        )

    def test_shipment_positions_and_delivery_days(self) -> None:
        result = self._run()
        shipments = result.final_state.shipments
        assert {s.status for s in shipments.values()} == {ShipmentStatus.DELIVERED}

        by_id = {sid: s for sid, s in shipments.items()}
        assert by_id["shipment_001_001"].delivered_day == 3
        assert by_id["shipment_002_001"].delivered_day == 4
        assert by_id["shipment_003_001"].delivered_day == 5
        assert all(s.quantity == 5 for s in shipments.values())
        assert all(s.current_node_id == "plant_1" for s in shipments.values())

    def test_inventory_and_backlog(self) -> None:
        result = self._run()
        assert result.final_state.inventory["plant_1"]["widget"] == 10
        assert all(
            quantity == 0
            for node_backlog in result.final_state.backlog.values()
            for quantity in node_backlog.values()
        )

    def test_costs_match_hand_calculation(self) -> None:
        result = self._run()
        costs = result.final_state.costs
        assert costs.transport == 30.0
        assert costs.holding == 2.0
        assert costs.reroute == 0.0
        assert costs.expedite == 0.0
        assert costs.backlog == 0.0
        assert costs.late == 0.0
        assert costs.terminal == 0.0

    def test_final_metrics(self) -> None:
        result = self._run()
        service = result.final_state.service
        assert service.total_demand_units == 15
        assert service.same_day_fulfilled_units == 15
        assert service.backlog_fulfilled_units == 0
        assert service.delivered_shipment_units == 15
        assert service.late_delivered_units == 0
        assert service.decision_count == 0  # decisions were disabled throughout
        assert result.final_day == 5
        assert result.terminated_with_unresolved_state is False
        assert (
            len(result.daily_metrics) == 5
        )  # early-stopped exactly at max_day, no gap

    def test_result_is_deterministic_across_repeated_runs(self) -> None:
        first = self._run()
        second = self._run()
        assert first == second


class TestFullSimulationDisrupted:
    """Exercises the edge-closure -> block -> reopen -> terminal-cost path.

    Hand-calculated trace: days 1-2 are identical to the undisrupted case.
    On day 3 the supplier_to_hub closure (days 3-4) becomes active and known;
    s002 (already in transit on that edge) is unaffected and arrives on
    schedule, but s003's departure on that same edge is blocked both days.
    The edge reopens on day 5, so s003 finally departs then, arriving day 6
    -- past the max drain day (5) -- so the run ends unresolved and charges
    terminal cost: 5 * 0.50 * max(1, 5 - 8) = 2.5.
    """

    def _run(self) -> SimulationResult:
        network_config, network_definition, initial_state = _build_day_zero()
        scenario_config = load_scenario_config(TINY_SCENARIO_CONFIG)
        disrupted_tape = build_disrupted_event_tape(
            network_definition=network_definition,
            demand_process=network_config.demand_process,
            replenishment_plan=network_config.replenishment_plan,
            scenario_config=scenario_config,
            replication=1,
            base_seed=42,
            horizon_days=HORIZON_DAYS,
            drain_days=DRAIN_DAYS,
        )

        return SimulationEngine().run(
            initial_state=initial_state,
            event_tape=disrupted_tape,
            start_day=1,
            horizon_day=HORIZON_DAYS,
            drain_days=DRAIN_DAYS,
            decision_enabled=False,
            run_identity=_run_identity("DISRUPTED"),
            reroute_cost_per_unit=REROUTE_COST_PER_UNIT,
            expedite_premium_per_unit=EXPEDITE_PREMIUM_PER_UNIT,
            terminal_penalty_days=TERMINAL_PENALTY_DAYS,
        )

    def test_blocked_shipment_departs_only_after_reopening(self) -> None:
        result = self._run()
        shipments = result.final_state.shipments

        assert shipments["shipment_001_001"].status is ShipmentStatus.DELIVERED
        assert shipments["shipment_001_001"].delivered_day == 3
        assert shipments["shipment_002_001"].status is ShipmentStatus.DELIVERED
        assert shipments["shipment_002_001"].delivered_day == 4

        s003 = shipments["shipment_003_001"]
        assert s003.status is ShipmentStatus.IN_TRANSIT
        assert (
            s003.edge_entry_day == 5
        )  # blocked on days 3-4, departs once reopened on day 5
        assert s003.edge_arrival_day == 6

    def test_inventory_reflects_only_the_two_delivered_shipments(self) -> None:
        result = self._run()
        # initial 10 - 15 consumed by demand + 10 delivered (s001, s002) = 5
        assert result.final_state.inventory["plant_1"]["widget"] == 5

    def test_costs_and_terminal_charge_match_hand_calculation(self) -> None:
        result = self._run()
        costs = result.final_state.costs
        assert costs.transport == 25.0
        assert costs.holding == 1.5
        assert costs.late == 0.0
        assert costs.terminal == 2.5
        assert result.terminated_with_unresolved_state is True

    def test_known_shock_is_recorded_and_active_shocks_clear_after_end_day(
        self,
    ) -> None:
        result = self._run()
        assert "close_supplier_to_hub" in result.final_state.known_shock_ids
        assert (
            result.final_state.active_shock_ids == set()
        )  # shock ended on day 4; day 5 is clear


class TestFullSimulationHeuristicPolicy:
    """Milestone 6: the real HeuristicPolicy, wired in through the engine.

    s003 is the only shipment ever triggered (its route is blocked by the
    edge closure on days 3-4, matching TestFullSimulationDisrupted's known
    active shock). On both triggered days, the only candidate route is the
    emergency supplier_to_plant_air lane (the normal route's first edge is
    closed, so it is excluded from route_options entirely), and waiting for
    the known day-5 reopening (arrival day 7) is still on time against the
    due day (8) and far cheaper than the emergency premium, so the heuristic
    chooses WAIT both times -- reproducing the exact same physical outcome
    as the disrupted decision_enabled=False run, but now via a real policy
    decision that was proposed, validated, and found valid on the first try.
    """

    def _run(self) -> SimulationResult:
        network_config, network_definition, initial_state = _build_day_zero()
        scenario_config = load_scenario_config(TINY_SCENARIO_CONFIG)
        disrupted_tape = build_disrupted_event_tape(
            network_definition=network_definition,
            demand_process=network_config.demand_process,
            replenishment_plan=network_config.replenishment_plan,
            scenario_config=scenario_config,
            replication=1,
            base_seed=42,
            horizon_days=HORIZON_DAYS,
            drain_days=DRAIN_DAYS,
        )

        return SimulationEngine().run(
            initial_state=initial_state,
            event_tape=disrupted_tape,
            start_day=1,
            horizon_day=HORIZON_DAYS,
            drain_days=DRAIN_DAYS,
            decision_enabled=True,
            run_identity=_run_identity("DISRUPTED"),
            reroute_cost_per_unit=REROUTE_COST_PER_UNIT,
            expedite_premium_per_unit=EXPEDITE_PREMIUM_PER_UNIT,
            terminal_penalty_days=TERMINAL_PENALTY_DAYS,
            policy=HeuristicPolicy(
                expedite_trigger_lateness_days=2, cost_tolerance=1e-9
            ),
            fallback_policy=WaitFallbackPolicy(),
            mean_daily_demand=network_config.demand_process.mean_daily_demand,
        )

    def test_heuristic_waits_out_the_closure_instead_of_paying_the_emergency_premium(
        self,
    ) -> None:
        result = self._run()
        service = result.final_state.service
        # s003 is triggered on both blocked days (3 and 4); the heuristic's own
        # candidate-cost comparison chooses WAIT both times (see class docstring).
        assert service.decision_count == 2
        assert service.wait_count == 2
        assert service.valid_action_count == 2
        assert service.reroute_count == 0
        assert service.expedite_count == 0
        assert service.invalid_action_count == 0
        assert service.abstention_count == 0
        assert service.fallback_count == 0

        costs = result.final_state.costs
        assert costs.transport == 25.0
        assert costs.holding == 1.5
        assert costs.terminal == 2.5
        assert result.terminated_with_unresolved_state is True

        s003 = result.final_state.shipments["shipment_003_001"]
        assert s003.status is ShipmentStatus.IN_TRANSIT
        assert s003.edge_entry_day == 5
        assert s003.edge_arrival_day == 6

    def test_decision_trace_sink_records_both_wait_decisions(self) -> None:
        network_config, network_definition, initial_state = _build_day_zero()
        scenario_config = load_scenario_config(TINY_SCENARIO_CONFIG)
        disrupted_tape = build_disrupted_event_tape(
            network_definition=network_definition,
            demand_process=network_config.demand_process,
            replenishment_plan=network_config.replenishment_plan,
            scenario_config=scenario_config,
            replication=1,
            base_seed=42,
            horizon_days=HORIZON_DAYS,
            drain_days=DRAIN_DAYS,
        )
        sink: list[DecisionTraceEntry] = []

        SimulationEngine().run(
            initial_state=initial_state,
            event_tape=disrupted_tape,
            start_day=1,
            horizon_day=HORIZON_DAYS,
            drain_days=DRAIN_DAYS,
            decision_enabled=True,
            run_identity=_run_identity("DISRUPTED"),
            reroute_cost_per_unit=REROUTE_COST_PER_UNIT,
            expedite_premium_per_unit=EXPEDITE_PREMIUM_PER_UNIT,
            terminal_penalty_days=TERMINAL_PENALTY_DAYS,
            policy=HeuristicPolicy(
                expedite_trigger_lateness_days=2, cost_tolerance=1e-9
            ),
            fallback_policy=WaitFallbackPolicy(),
            mean_daily_demand=network_config.demand_process.mean_daily_demand,
            decision_trace_sink=sink,
        )

        assert [entry.day for entry in sink] == [3, 4]
        assert all(entry.shipment_id == "shipment_003_001" for entry in sink)
        assert all(
            entry.executed_action.action_type is ActionType.WAIT for entry in sink
        )
        assert all(entry.fallback_invoked is False for entry in sink)
        assert all(entry.proposal_validation.is_valid for entry in sink)
        assert all(len(entry.observation_hash) == 64 for entry in sink)
        assert all(entry.decision_latency_ms >= 0.0 for entry in sink)

    def test_missing_policy_with_decisions_enabled_raises(self) -> None:
        network_config, network_definition, initial_state = _build_day_zero()
        scenario_config = load_scenario_config(TINY_SCENARIO_CONFIG)
        disrupted_tape = build_disrupted_event_tape(
            network_definition=network_definition,
            demand_process=network_config.demand_process,
            replenishment_plan=network_config.replenishment_plan,
            scenario_config=scenario_config,
            replication=1,
            base_seed=42,
            horizon_days=HORIZON_DAYS,
            drain_days=DRAIN_DAYS,
        )
        with pytest.raises(ValueError, match="policy and fallback_policy are required"):
            SimulationEngine().run(
                initial_state=initial_state,
                event_tape=disrupted_tape,
                start_day=1,
                horizon_day=HORIZON_DAYS,
                drain_days=DRAIN_DAYS,
                decision_enabled=True,
                run_identity=_run_identity("DISRUPTED"),
                reroute_cost_per_unit=REROUTE_COST_PER_UNIT,
                expedite_premium_per_unit=EXPEDITE_PREMIUM_PER_UNIT,
                terminal_penalty_days=TERMINAL_PENALTY_DAYS,
            )


class TestObservationsOnlyRevealKnownShocks:
    """Regression test for the known_shock_ids filtering fix.

    Builds a one-day event tape by hand: a shock targets hub_to_plant (on
    s001's route) but its information_day is day 5, while s001's due_day is
    set to its own release day so a decision is triggered on day 1 purely by
    the lateness trigger (CLAUDE.md section 19 trigger 3) -- independent of
    the shock. Before the fix, engine.py handed every shock in the event
    tape to build_observation regardless of state.known_shock_ids, so the
    day-1 observation would have leaked the shock's existence four days
    before policies are supposed to know about it (CLAUDE.md section 20).
    """

    def _run(self) -> list[DecisionTraceEntry]:
        _, network_definition, initial_state = _build_day_zero()

        future_shock = Shock(
            shock_id="future_leak_shock",
            shock_type=ShockType.EDGE_CLOSURE,
            target_type=TargetType.EDGE,
            target_id="hub_to_plant",
            physical_start_day=5,
            physical_end_day=6,
            information_day=5,
        )
        release_event = ShipmentReleaseEvent(
            day=1,
            shipment_id="shipment_001_001",
            product_id="widget",
            quantity=5,
            origin_node_id="supplier_1",
            destination_node_id="plant_1",
            due_day=1,  # already "due" on release day: forces the lateness trigger
            initial_route_edge_ids=("supplier_to_hub", "hub_to_plant"),
        )
        day_one = DayEvents(
            day=1,
            demand_events=(),
            shipment_release_events=(release_event,),
            edge_extra_delay_days=dict.fromkeys(network_definition.edges, 0),
            newly_known_shock_ids=(),  # information_day is 5, not 1
        )
        tape = EventTape(
            scenario_id="future_leak_scenario",
            replication=1,
            seed=1,
            days=(day_one,),
            shocks=(future_shock,),
        )

        sink: list[DecisionTraceEntry] = []
        SimulationEngine().run(
            initial_state=initial_state,
            event_tape=tape,
            start_day=1,
            horizon_day=1,
            drain_days=0,
            decision_enabled=True,
            run_identity=_run_identity("DISRUPTED"),
            reroute_cost_per_unit=REROUTE_COST_PER_UNIT,
            expedite_premium_per_unit=EXPEDITE_PREMIUM_PER_UNIT,
            terminal_penalty_days=TERMINAL_PENALTY_DAYS,
            policy=HeuristicPolicy(
                expedite_trigger_lateness_days=2, cost_tolerance=1e-9
            ),
            fallback_policy=WaitFallbackPolicy(),
            mean_daily_demand=5.0,
            decision_trace_sink=sink,
        )
        return sink

    def test_decision_is_triggered_by_lateness_not_the_shock(self) -> None:
        sink = self._run()
        assert [entry.shipment_id for entry in sink] == ["shipment_001_001"]

    def test_future_shock_is_absent_from_the_day_one_observation(self) -> None:
        sink = self._run()
        observation = sink[0].observation
        shock_ids = {shock.shock_id for shock in observation.relevant_shocks}
        assert "future_leak_shock" not in shock_ids
        assert observation.relevant_shocks == ()


class _AbstainingPolicy:
    """Test double: always abstains, forcing the engine's fallback chain."""

    @property
    def name(self) -> str:
        return "always_abstain"

    def decide(self, observation: DecisionObservation) -> DecisionAction:
        return DecisionAction(
            shipment_id=observation.shipment.shipment_id,
            action_type=ActionType.ABSTAIN,
            route_id=None,
            reason_code=ReasonCode.INSUFFICIENT_INFORMATION,
            rationale="test double: always abstains",
        )


class _InvalidRerouteThenWaitPolicy:
    """Test double: proposes a REROUTE to a route_id that was never offered.

    Distinct from _AbstainingPolicy: this proposal is INVALID (fails
    decisions/validator.py's ROUTE_NOT_ALLOWED check), not an ABSTAIN, so it
    exercises invalid_action_count rather than abstention_count.
    """

    @property
    def name(self) -> str:
        return "always_invalid_reroute"

    def decide(self, observation: DecisionObservation) -> DecisionAction:
        return DecisionAction(
            shipment_id=observation.shipment.shipment_id,
            action_type=ActionType.REROUTE,
            route_id="not_a_real_route",
            reason_code=ReasonCode.LOWER_ESTIMATED_COST,
            rationale="test double: never a valid route",
        )


class _AlwaysExpeditePolicy:
    """Test double: always expedites via the first emergency route option."""

    @property
    def name(self) -> str:
        return "always_expedite"

    def decide(self, observation: DecisionObservation) -> DecisionAction:
        emergency_option = next(
            option
            for option in observation.route_options
            if option.contains_emergency_edge
        )
        return DecisionAction(
            shipment_id=observation.shipment.shipment_id,
            action_type=ActionType.EXPEDITE,
            route_id=emergency_option.route_id,
            reason_code=ReasonCode.REDUCE_LATENESS,
            rationale="test double: always expedites",
        )


class TestFullSimulationFallbackChain:
    """Exercises the engine's fallback-chain wiring: an abstaining or invalid
    policy must fall back to WaitFallbackPolicy's WAIT, incrementing the
    matching service counters without changing the physical outcome at all
    (WAIT is a no-op either way).
    """

    def _run(self, policy: Policy) -> SimulationResult:
        network_config, network_definition, initial_state = _build_day_zero()
        scenario_config = load_scenario_config(TINY_SCENARIO_CONFIG)
        disrupted_tape = build_disrupted_event_tape(
            network_definition=network_definition,
            demand_process=network_config.demand_process,
            replenishment_plan=network_config.replenishment_plan,
            scenario_config=scenario_config,
            replication=1,
            base_seed=42,
            horizon_days=HORIZON_DAYS,
            drain_days=DRAIN_DAYS,
        )

        return SimulationEngine().run(
            initial_state=initial_state,
            event_tape=disrupted_tape,
            start_day=1,
            horizon_day=HORIZON_DAYS,
            drain_days=DRAIN_DAYS,
            decision_enabled=True,
            run_identity=_run_identity("DISRUPTED"),
            reroute_cost_per_unit=REROUTE_COST_PER_UNIT,
            expedite_premium_per_unit=EXPEDITE_PREMIUM_PER_UNIT,
            terminal_penalty_days=TERMINAL_PENALTY_DAYS,
            policy=policy,
            fallback_policy=WaitFallbackPolicy(),
            mean_daily_demand=network_config.demand_process.mean_daily_demand,
        )

    def test_abstention_falls_back_to_wait_and_counters_reflect_it(self) -> None:
        result = self._run(_AbstainingPolicy())
        service = result.final_state.service
        assert service.decision_count == 2
        assert service.abstention_count == 2
        assert service.fallback_count == 2
        assert service.invalid_action_count == 0
        assert service.wait_count == 2

        # Fallback resolved to WAIT both times, so the physical trace matches
        # the heuristic-WAIT and decisions-disabled runs exactly.
        costs = result.final_state.costs
        assert costs.transport == 25.0
        assert costs.terminal == 2.5
        assert result.terminated_with_unresolved_state is True

    def test_invalid_proposal_falls_back_to_wait_and_counters_reflect_it(self) -> None:
        result = self._run(_InvalidRerouteThenWaitPolicy())
        service = result.final_state.service
        assert service.decision_count == 2
        assert service.invalid_action_count == 2
        assert service.fallback_count == 2
        assert service.abstention_count == 0
        assert service.wait_count == 2

        costs = result.final_state.costs
        assert costs.transport == 25.0
        assert costs.reroute == 0.0
        assert costs.terminal == 2.5
        assert result.terminated_with_unresolved_state is True


class TestFullSimulationExecutesExpedite:
    """A policy that always proposes EXPEDITE must have that route actually
    resolved and applied by the engine (simulation/engine.py's `_route_for`),
    diverging s003 onto the emergency air lane instead of waiting out the
    closure.
    """

    def _run(self) -> SimulationResult:
        network_config, network_definition, initial_state = _build_day_zero()
        scenario_config = load_scenario_config(TINY_SCENARIO_CONFIG)
        disrupted_tape = build_disrupted_event_tape(
            network_definition=network_definition,
            demand_process=network_config.demand_process,
            replenishment_plan=network_config.replenishment_plan,
            scenario_config=scenario_config,
            replication=1,
            base_seed=42,
            horizon_days=HORIZON_DAYS,
            drain_days=DRAIN_DAYS,
        )

        return SimulationEngine().run(
            initial_state=initial_state,
            event_tape=disrupted_tape,
            start_day=1,
            horizon_day=HORIZON_DAYS,
            drain_days=DRAIN_DAYS,
            decision_enabled=True,
            run_identity=_run_identity("DISRUPTED"),
            reroute_cost_per_unit=REROUTE_COST_PER_UNIT,
            expedite_premium_per_unit=EXPEDITE_PREMIUM_PER_UNIT,
            terminal_penalty_days=TERMINAL_PENALTY_DAYS,
            policy=_AlwaysExpeditePolicy(),
            fallback_policy=WaitFallbackPolicy(),
            mean_daily_demand=network_config.demand_process.mean_daily_demand,
        )

    def test_expedite_diverts_shipment_onto_the_emergency_route(self) -> None:
        result = self._run()
        service = result.final_state.service
        assert service.decision_count == 1  # expedited on day 3; never triggered again
        assert service.expedite_count == 1
        assert service.expedited_units == 5

        s003 = result.final_state.shipments["shipment_003_001"]
        assert s003.status is ShipmentStatus.DELIVERED
        assert s003.planned_route_edge_ids == ("supplier_to_plant_air",)
        assert s003.delivered_day == 4  # departs day 3, 1-day air lead time

        costs = result.final_state.costs
        assert costs.expedite == pytest.approx(10.0)  # 5 units * 2.0 premium
        # s001/s002 contribute 20.0 as usual; s003 pays the 5.00/unit air rate
        # instead of the 1.00/unit road rate: 20.0 + 5 * 5.00 == 45.0.
        assert costs.transport == pytest.approx(45.0)
        assert costs.terminal == 0.0
        assert result.terminated_with_unresolved_state is False


class TestFullSimulationSupplierClosureDefersReleases:
    """V2 §V2.3.7: closing the supplier itself (NODE_CLOSURE on a SUPPLIER
    node) makes release_shipments defer instead of raising. Product balance
    is asserted every day by simulation/engine.py's _assert_invariants, so a
    run that completes at all across a deferred-then-retried release is
    already proof the corrected total_released bookkeeping (V2.7's engine.py
    correction) holds.
    """

    def _run(self) -> SimulationResult:
        network_config, network_definition, initial_state = _build_day_zero()
        scenario_config = ScenarioConfig(
            schema_version=1,
            scenario_id="tiny_supplier_closure",
            description="Supplier closed for the first two days.",
            shocks=[
                ShockConfig(
                    shock_id="close_supplier",
                    shock_type="NODE_CLOSURE",
                    target_type="NODE",
                    target_id="supplier_1",
                    planned_start_day=1,
                    start_day_jitter_days=0,
                    minimum_duration_days=2,
                    duration_mean_days=2,
                    duration_std_days=0,
                    maximum_duration_days=2,
                    max_information_delay_days=0,
                )
            ],
        )
        disrupted_tape = build_disrupted_event_tape(
            network_definition=network_definition,
            demand_process=network_config.demand_process,
            replenishment_plan=network_config.replenishment_plan,
            scenario_config=scenario_config,
            replication=1,
            base_seed=42,
            horizon_days=HORIZON_DAYS,
            drain_days=DRAIN_DAYS,
        )

        return SimulationEngine().run(
            initial_state=initial_state,
            event_tape=disrupted_tape,
            start_day=1,
            horizon_day=HORIZON_DAYS,
            drain_days=DRAIN_DAYS,
            decision_enabled=False,
            run_identity=_run_identity("DISRUPTED"),
            reroute_cost_per_unit=REROUTE_COST_PER_UNIT,
            expedite_premium_per_unit=EXPEDITE_PREMIUM_PER_UNIT,
            terminal_penalty_days=TERMINAL_PENALTY_DAYS,
        )

    def test_run_completes_without_invariant_violation(self) -> None:
        # If total_released ever overcounted a deferred release, the daily
        # product-balance assertion in engine.py would raise before this
        # returns -- completing at all is the real assertion here.
        result = self._run()
        assert result.final_state.pending_releases == []

    def test_deferred_releases_keep_their_originally_scheduled_due_day(self) -> None:
        result = self._run()
        shipments = result.final_state.shipments
        # s001 and s002 were scheduled on days 1 and 2, both blocked by the
        # closure, and only actually release once supplier_1 reopens on day 3
        # -- but due_day stays anchored to the original schedule (day + 5).
        assert shipments["shipment_001_001"].due_day == 6
        assert shipments["shipment_002_001"].due_day == 7

    def test_delayed_release_still_departs_and_is_eventually_delivered(self) -> None:
        result = self._run()
        s001 = result.final_state.shipments["shipment_001_001"]
        assert s001.status is ShipmentStatus.DELIVERED
        assert s001.delivered_day is not None
        assert s001.delivered_day > 3  # released day 3 at the earliest, not day 1


class TestEngineRejectsAnUndersizedEventTape:
    def test_requesting_a_day_beyond_the_tape_raises(self) -> None:
        network_config, network_definition, initial_state = _build_day_zero()
        scenario_config = load_scenario_config(TINY_SCENARIO_CONFIG)
        short_tape = build_undisrupted_event_tape(
            build_disrupted_event_tape(
                network_definition=network_definition,
                demand_process=network_config.demand_process,
                replenishment_plan=network_config.replenishment_plan,
                scenario_config=scenario_config,
                replication=1,
                base_seed=42,
                horizon_days=HORIZON_DAYS,
                drain_days=0,  # tape only covers days 1..3
            )
        )

        with pytest.raises(SimulationInvariantError, match="no entry for day"):
            SimulationEngine().run(
                initial_state=initial_state,
                event_tape=short_tape,
                start_day=1,
                horizon_day=HORIZON_DAYS,
                drain_days=DRAIN_DAYS,  # but the run asks for 2 extra drain days
                decision_enabled=False,
                run_identity=_run_identity("UNDISRUPTED"),
                reroute_cost_per_unit=REROUTE_COST_PER_UNIT,
                expedite_premium_per_unit=EXPEDITE_PREMIUM_PER_UNIT,
                terminal_penalty_days=TERMINAL_PENALTY_DAYS,
            )
