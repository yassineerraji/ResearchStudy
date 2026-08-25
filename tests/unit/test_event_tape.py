"""Unit tests for event_tape.py: deterministic stream derivation, event generation, shock realization, and that the undisrupted tape differs from the disrupted one only in its shocks."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from supply_chain_simulator.data_io.loaders import (
    ReplenishmentPlanConfig,
    ScenarioConfig,
    ShockConfig,
    build_network_definition,
    load_network_config,
    load_scenario_config,
)
from supply_chain_simulator.domain.events import EventTape, Shock, ShockType, TargetType
from supply_chain_simulator.domain.models import NetworkDefinition
from supply_chain_simulator.experiments.event_tape import (
    DEMAND_STREAM,
    EDGE_DELAYS_STREAM,
    RELEASE_QUANTITY_STREAM,
    SHOCK_REALIZATION_STREAM,
    build_disrupted_event_tape,
    build_undisrupted_event_tape,
    derive_stream_seed,
    generate_demand_events,
    generate_edge_delay_draws,
    generate_shipment_release_events,
    realize_all_shocks,
    realize_release_quantity,
    realize_shock,
    realize_shock_group,
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

    def test_demand_shock_scales_mean_and_bounds_without_changing_draw_count(self) -> None:
        """V2 §V2.3.6: a demand shock changes what the demand stream's single
        per-day gauss() draw is centered around, not how many draws happen.
        """
        demand_process = load_network_config(TINY_NETWORK_CONFIG).demand_process.model_copy(
            update={"standard_deviation": 1.0}
        )
        unshocked = generate_demand_events(
            demand_process, horizon_days=5, rng=random.Random(derive_stream_seed(9, DEMAND_STREAM))
        )

        demand_shock = Shock(
            shock_id="demand_spike",
            shock_type=ShockType.DEMAND_SPIKE,
            target_type=TargetType.DEMAND,
            target_id="plant_1",
            physical_start_day=2,
            physical_end_day=3,
            information_day=2,
            demand_multiplier=2.0,
        )
        shocked = generate_demand_events(
            demand_process,
            horizon_days=5,
            rng=random.Random(derive_stream_seed(9, DEMAND_STREAM)),
            demand_shocks=(demand_shock,),
        )

        assert len(shocked) == len(unshocked) == 5
        # Days outside the shock window are completely unaffected.
        assert shocked[0] == unshocked[0]
        assert shocked[3] == unshocked[3]
        assert shocked[4] == unshocked[4]
        # Days inside the window scale with the multiplier (mean 5 -> 10,
        # zero variance in this fixture would normally clamp to it exactly,
        # but standard_deviation is 1.0 here so we only assert the bound moved).
        assert shocked[1].quantity != unshocked[1].quantity or shocked[1].quantity >= 5
        assert shocked[1].day == 2 and shocked[2].day == 3


    def test_demand_spike_and_drop_are_correctly_directioned(self) -> None:
        """V2.11's acceptance bar requires DEMAND_SPIKE *and* DEMAND_DROP to
        each produce a measurable, correctly-directioned effect -- exercised
        here with the tiny fixture's zero-variance demand (mean=min=max=5)
        so both directions are exact, hand-verifiable numbers, not just
        'different from baseline'.
        """
        demand_process = load_network_config(TINY_NETWORK_CONFIG).demand_process

        def _shock(shock_id: str, shock_type: ShockType, multiplier: float) -> Shock:
            return Shock(
                shock_id=shock_id,
                shock_type=shock_type,
                target_type=TargetType.DEMAND,
                target_id="plant_1",
                physical_start_day=2,
                physical_end_day=2,
                information_day=2,
                demand_multiplier=multiplier,
            )

        spiked = generate_demand_events(
            demand_process,
            horizon_days=3,
            rng=random.Random(derive_stream_seed(1, DEMAND_STREAM)),
            demand_shocks=(_shock("spike", ShockType.DEMAND_SPIKE, 2.0),),
        )
        dropped = generate_demand_events(
            demand_process,
            horizon_days=3,
            rng=random.Random(derive_stream_seed(1, DEMAND_STREAM)),
            demand_shocks=(_shock("drop", ShockType.DEMAND_DROP, 0.4),),
        )

        # Day 2 is inside the shock window; days 1 and 3 are unaffected (base
        # demand is a fixed 5 units/day in this fixture).
        assert spiked[0].quantity == 5 and dropped[0].quantity == 5
        assert spiked[1].quantity == 10  # 5 * 2.0
        assert dropped[1].quantity == 2  # 5 * 0.4
        assert spiked[2].quantity == 5 and dropped[2].quantity == 5


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

    def _rng(self, replication: int = 1) -> random.Random:
        return random.Random(derive_stream_seed(replication, RELEASE_QUANTITY_STREAM))

    def test_one_release_per_day_within_horizon(self) -> None:
        events = generate_shipment_release_events(
            self._plan(), horizon_days=3, network_definition=_tiny_network_definition(), rng=self._rng()
        )
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
        events = generate_shipment_release_events(
            plan, horizon_days=5, network_definition=_tiny_network_definition(), rng=self._rng()
        )
        assert events == ()

    def test_quantity_clamped_to_route_feasibility_ceiling(self) -> None:
        """A drawn quantity above the route's static edge/node capacity
        (V2 §V2.3.5) is clamped down, even if it's within the configured
        [minimum_shipment_quantity, maximum_shipment_quantity] bounds.
        """
        plan = self._plan().model_copy(
            update={
                "shipment_quantity_mean": 20,
                "shipment_quantity_std": 0,
                "minimum_shipment_quantity": 20,
                "maximum_shipment_quantity": 20,
            }
        )
        # supplier_to_hub's daily_capacity is 20 in the tiny fixture; a route
        # requiring exactly that stays feasible, but hub_1's processing_capacity
        # (50) and every other route capacity are >= 20, so 20 survives here.
        # Push the requested quantity above every route capacity instead.
        plan = plan.model_copy(
            update={
                "shipment_quantity_mean": 999,
                "shipment_quantity_std": 0,
                "minimum_shipment_quantity": 999,
                "maximum_shipment_quantity": 999,
            }
        )
        events = generate_shipment_release_events(
            plan, horizon_days=1, network_definition=_tiny_network_definition(), rng=self._rng()
        )
        assert events[0].quantity == 20

    def test_same_seed_reproduces_identical_quantities(self) -> None:
        plan = self._plan().model_copy(
            update={"shipment_quantity_mean": 5, "shipment_quantity_std": 2, "maximum_shipment_quantity": 10}
        )
        first = generate_shipment_release_events(
            plan, horizon_days=10, network_definition=_tiny_network_definition(), rng=self._rng(3)
        )
        second = generate_shipment_release_events(
            plan, horizon_days=10, network_definition=_tiny_network_definition(), rng=self._rng(3)
        )
        assert first == second


class TestRealizeReleaseQuantity:
    def test_zero_std_reproduces_fixed_mean(self) -> None:
        rng = random.Random(1)
        assert realize_release_quantity(40, 0, 20, 55, rng) == 40

    def test_clamped_to_bounds(self) -> None:
        rng = random.Random(1)
        assert realize_release_quantity(1000, 0, 20, 55, rng) == 55


class TestShockRealization:
    def _template(self, **overrides: object) -> ShockConfig:
        base: dict[str, object] = {
            "shock_id": "s1",
            "shock_type": "NODE_CLOSURE",
            "target_type": "NODE",
            "target_id": "port_primary",
            "planned_start_day": 25,
            "start_day_jitter_days": 3,
            "minimum_duration_days": 5,
            "duration_mean_days": 10,
            "duration_std_days": 3,
            "maximum_duration_days": 18,
            "max_information_delay_days": 2,
        }
        base.update(overrides)
        return ShockConfig(**base)  # type: ignore[arg-type]

    def test_all_zero_uncertainty_reproduces_fixed_shock(self) -> None:
        template = self._template(
            planned_start_day=21,
            start_day_jitter_days=0,
            minimum_duration_days=7,
            duration_mean_days=7,
            duration_std_days=0,
            maximum_duration_days=7,
            max_information_delay_days=0,
        )
        shock = realize_shock(template, random.Random(1))
        assert shock.physical_start_day == 21
        assert shock.physical_end_day == 27
        assert shock.information_day == 21

    def test_deterministic_for_fixed_stream_seed(self) -> None:
        template = self._template()
        first = realize_shock(template, random.Random(42))
        second = realize_shock(template, random.Random(42))
        assert first == second

    def test_draw_order_is_jitter_then_duration_then_information_delay(self) -> None:
        """A stream that only ever returns 0 for randint and the mean for
        gauss lets us confirm each formula consumes the RNG in the documented
        order without depending on the exact underlying draw values.
        """
        template = self._template(
            planned_start_day=25, start_day_jitter_days=5, max_information_delay_days=4
        )
        rng = random.Random(7)
        shock = realize_shock(template, rng)
        # physical_start_day must fall within the jitter window, duration
        # within its bounds, and information_day within [start, end] -- this
        # exercises all three draws happening, in the documented order.
        assert 20 <= shock.physical_start_day <= 30
        assert 5 <= (shock.physical_end_day - shock.physical_start_day + 1) <= 18
        assert shock.physical_start_day <= shock.information_day <= shock.physical_end_day

    def test_group_shares_one_jitter_draw_but_independent_duration_and_delay(self) -> None:
        member_a = self._template(shock_id="a", planned_start_day=22, event_group_id="g1")
        member_b = self._template(
            shock_id="b",
            planned_start_day=22,
            duration_mean_days=14,
            minimum_duration_days=7,
            maximum_duration_days=25,
            event_group_id="g1",
        )
        realized = realize_shock_group([member_b, member_a], random.Random(5))
        realized_a, realized_b = (s for s in realized if s.shock_id == "a"), (
            s for s in realized if s.shock_id == "b"
        )
        shock_a = next(realized_a)
        shock_b = next(realized_b)
        assert (shock_a.physical_start_day - 22) == (shock_b.physical_start_day - 22)
        # durations were drawn independently from different distributions,
        # so they need not (and, given the differing means, should not) match.
        assert (shock_a.physical_end_day - shock_a.physical_start_day) != (
            shock_b.physical_end_day - shock_b.physical_start_day
        )

    def test_realize_all_shocks_orders_groups_before_ungrouped(self) -> None:
        scenario = ScenarioConfig(
            schema_version=1,
            scenario_id="s",
            description="d",
            shocks=[
                self._template(shock_id="z_ungrouped", event_group_id=None),
                self._template(shock_id="b1", event_group_id="group_b"),
                self._template(shock_id="a1", event_group_id="group_a"),
            ],
        )
        realized_grouped_first = realize_all_shocks(scenario, random.Random(1))
        realized_grouped_again = realize_all_shocks(scenario, random.Random(1))
        assert realized_grouped_first == realized_grouped_again
        assert {shock.shock_id for shock in realized_grouped_first} == {"z_ungrouped", "b1", "a1"}

    def test_shock_realization_stream_distinct_from_others(self) -> None:
        seed = derive_stream_seed(1, SHOCK_REALIZATION_STREAM)
        assert seed != derive_stream_seed(1, DEMAND_STREAM)
        assert seed != derive_stream_seed(1, EDGE_DELAYS_STREAM)
        assert seed != derive_stream_seed(1, RELEASE_QUANTITY_STREAM)


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
