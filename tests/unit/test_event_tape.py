"""Unit tests for experiments/event_tape.py: reproducibility and pairing.

Inside tests/unit, this file checks that stream-seed derivation, demand
generation, edge-delay generation, and shipment-release generation are all
deterministic given a fixed seed, that a built EventTape reflects the tiny
fixture's scenario correctly (shock conversion, known-shock-id timing, day
coverage through the drain period), and that the undisrupted tape produced
from a disrupted one differs only in its shocks. It does not test how these
events are later applied to a SimulationState, since the engine does not
exist yet at this stage of the build.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from supply_chain_simulator.data_io.loaders import (
    ReplenishmentPlanConfig,
    build_network_definition,
    load_network_config,
    load_scenario_config,
)
from supply_chain_simulator.domain.events import EventTape, ShockType, TargetType
from supply_chain_simulator.domain.models import NetworkDefinition
from supply_chain_simulator.experiments.event_tape import (
    DEMAND_STREAM,
    EDGE_DELAYS_STREAM,
    build_disrupted_event_tape,
    build_undisrupted_event_tape,
    derive_stream_seed,
    generate_demand_events,
    generate_edge_delay_draws,
    generate_shipment_release_events,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TINY_NETWORK_CONFIG = REPO_ROOT / "tests/fixtures/tiny_network.yaml"
TINY_SCENARIO_CONFIG = REPO_ROOT / "tests/fixtures/tiny_scenario.yaml"


def _tiny_network_definition() -> NetworkDefinition:
    return build_network_definition(load_network_config(TINY_NETWORK_CONFIG))


class TestDeriveStreamSeed:
    def test_deterministic_for_same_inputs(self) -> None:
        assert derive_stream_seed(100, DEMAND_STREAM) == derive_stream_seed(100, DEMAND_STREAM)

    def test_differs_by_stream_name(self) -> None:
        assert derive_stream_seed(100, DEMAND_STREAM) != derive_stream_seed(100, EDGE_DELAYS_STREAM)

    def test_differs_by_replication_seed(self) -> None:
        assert derive_stream_seed(100, DEMAND_STREAM) != derive_stream_seed(101, DEMAND_STREAM)


class TestGenerateDemandEvents:
    def test_zero_variance_is_deterministic_and_exact(self) -> None:
        demand_process = load_network_config(TINY_NETWORK_CONFIG).demand_process
        rng = random.Random(derive_stream_seed(1, DEMAND_STREAM))
        events = generate_demand_events(demand_process, horizon_days=5, rng=rng)

        assert [event.day for event in events] == [1, 2, 3, 4, 5]
        assert all(event.quantity == 5 for event in events)
        assert all(event.destination_node_id == "plant_1" for event in events)
        assert all(event.product_id == "widget" for event in events)

    def test_quantities_are_clamped_to_configured_bounds(self) -> None:
        demand_process = load_network_config(TINY_NETWORK_CONFIG).demand_process
        wide_demand_process = demand_process.model_copy(
            update={"mean_daily_demand": 5.0, "standard_deviation": 100.0}
        )
        rng = random.Random(derive_stream_seed(1, DEMAND_STREAM))
        events = generate_demand_events(wide_demand_process, horizon_days=50, rng=rng)
        assert all(
            demand_process.minimum_daily_demand <= event.quantity <= demand_process.maximum_daily_demand
            for event in events
        )

    def test_same_seed_reproduces_identical_events(self) -> None:
        demand_process = load_network_config(TINY_NETWORK_CONFIG).demand_process.model_copy(
            update={"standard_deviation": 2.0}
        )
        first = generate_demand_events(
            demand_process, horizon_days=10, rng=random.Random(derive_stream_seed(7, DEMAND_STREAM))
        )
        second = generate_demand_events(
            demand_process, horizon_days=10, rng=random.Random(derive_stream_seed(7, DEMAND_STREAM))
        )
        assert first == second


class TestGenerateEdgeDelayDraws:
    def test_fully_reliable_edges_never_delay(self) -> None:
        network_definition = _tiny_network_definition()
        rng = random.Random(derive_stream_seed(1, EDGE_DELAYS_STREAM))
        draws = generate_edge_delay_draws(network_definition, max_day=10, rng=rng)

        assert set(draws) == set(range(1, 11))
        for day_draws in draws.values():
            assert set(day_draws) == set(network_definition.edges)
            assert all(delay == 0 for delay in day_draws.values())

    def test_same_seed_reproduces_identical_draws(self) -> None:
        network_definition = _tiny_network_definition()
        first = generate_edge_delay_draws(
            network_definition, max_day=10, rng=random.Random(derive_stream_seed(3, EDGE_DELAYS_STREAM))
        )
        second = generate_edge_delay_draws(
            network_definition, max_day=10, rng=random.Random(derive_stream_seed(3, EDGE_DELAYS_STREAM))
        )
        assert first == second


class TestGenerateShipmentReleaseEvents:
    def _plan(self) -> ReplenishmentPlanConfig:
        return load_network_config(TINY_NETWORK_CONFIG).replenishment_plan

    def test_one_release_per_day_within_horizon(self) -> None:
        events = generate_shipment_release_events(self._plan(), horizon_days=3)
        assert [event.day for event in events] == [1, 2, 3]
        assert [event.shipment_id for event in events] == [
            "shipment_001_001",
            "shipment_002_001",
            "shipment_003_001",
        ]
        assert all(event.quantity == 5 for event in events)
        assert all(event.due_day == event.day + 5 for event in events)
        assert all(
            event.initial_route_edge_ids == ("supplier_to_hub", "hub_to_plant") for event in events
        )

    def test_no_releases_after_horizon(self) -> None:
        plan = self._plan().model_copy(update={"first_release_day": 10})
        events = generate_shipment_release_events(plan, horizon_days=5)
        assert events == ()


class TestBuildDisruptedEventTape:
    def _build(self, replication: int = 1) -> EventTape:
        network_config = load_network_config(TINY_NETWORK_CONFIG)
        scenario_config = load_scenario_config(TINY_SCENARIO_CONFIG)
        network_definition = _tiny_network_definition()
        return build_disrupted_event_tape(
            network_definition=network_definition,
            demand_process=network_config.demand_process,
            replenishment_plan=network_config.replenishment_plan,
            scenario_config=scenario_config,
            replication=replication,
            base_seed=1000,
            horizon_days=10,
            drain_days=5,
        )

    def test_replication_below_one_rejected(self) -> None:
        with pytest.raises(ValueError, match="replication numbering starts at 1"):
            self._build(replication=0)

    def test_covers_horizon_plus_drain_days(self) -> None:
        tape = self._build()
        assert len(tape.days) == 15
        assert [day_events.day for day_events in tape.days] == list(range(1, 16))

    def test_shocks_converted_correctly(self) -> None:
        tape = self._build()
        assert len(tape.shocks) == 1
        shock = tape.shocks[0]
        assert shock.shock_id == "close_supplier_to_hub"
        assert shock.shock_type is ShockType.EDGE_CLOSURE
        assert shock.target_type is TargetType.EDGE
        assert shock.target_id == "supplier_to_hub"

    def test_known_shock_ids_appear_only_on_information_day(self) -> None:
        tape = self._build()
        by_day = {day_events.day: day_events.newly_known_shock_ids for day_events in tape.days}
        assert by_day[3] == ("close_supplier_to_hub",)
        assert by_day[1] == ()
        assert by_day[2] == ()
        assert by_day[4] == ()

    def test_demand_and_release_events_present_only_within_horizon(self) -> None:
        tape = self._build()
        for day_events in tape.days:
            if day_events.day <= 10:
                assert len(day_events.demand_events) == 1
                assert len(day_events.shipment_release_events) == 1
            else:
                assert day_events.demand_events == ()
                assert day_events.shipment_release_events == ()

    def test_edge_delay_days_present_for_every_day_including_drain(self) -> None:
        tape = self._build()
        network_definition = _tiny_network_definition()
        for day_events in tape.days:
            assert set(day_events.edge_extra_delay_days) == set(network_definition.edges)

    def test_reproducible_for_same_replication_and_seed(self) -> None:
        assert self._build(replication=2) == self._build(replication=2)

    def test_differs_across_replications(self) -> None:
        assert self._build(replication=1).seed != self._build(replication=2).seed


class TestBuildUndisruptedEventTape:
    def test_removes_shocks_but_keeps_ordinary_events_identical(self) -> None:
        network_config = load_network_config(TINY_NETWORK_CONFIG)
        scenario_config = load_scenario_config(TINY_SCENARIO_CONFIG)
        network_definition = _tiny_network_definition()
        disrupted = build_disrupted_event_tape(
            network_definition=network_definition,
            demand_process=network_config.demand_process,
            replenishment_plan=network_config.replenishment_plan,
            scenario_config=scenario_config,
            replication=1,
            base_seed=1000,
            horizon_days=10,
            drain_days=5,
        )
        undisrupted = build_undisrupted_event_tape(disrupted)

        assert undisrupted.shocks == ()
        assert undisrupted.scenario_id == disrupted.scenario_id
        assert undisrupted.replication == disrupted.replication
        assert undisrupted.seed == disrupted.seed

        for before, after in zip(disrupted.days, undisrupted.days, strict=True):
            assert after.day == before.day
            assert after.demand_events == before.demand_events
            assert after.shipment_release_events == before.shipment_release_events
            assert after.edge_extra_delay_days == before.edge_extra_delay_days
            assert after.newly_known_shock_ids == ()
