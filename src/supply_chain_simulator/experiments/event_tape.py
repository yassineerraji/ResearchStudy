"""Generates the policy-independent random future shared by every branch.

Inside the experiments package, this module derives per-replication random
streams from a base seed, draws realized demand and ordinary transport
delays from them, computes the fixed shipment-release schedule, and bundles
all of it together with the designed shocks into one EventTape per
replication. In the full system, this is what guarantees fairness: every
random draw happens once, before any policy makes a single decision, and the
undisrupted counterfactual reuses the exact same draws with only the designed
shocks removed. It does not decide what a policy does with these events, and
it does not apply them to a SimulationState — that is simulation/engine.py's
job.
"""

from __future__ import annotations

import hashlib
import random

from supply_chain_simulator.data_io.loaders import (
    DemandProcessConfig,
    ReplenishmentPlanConfig,
    ScenarioConfig,
    build_shocks,
)
from supply_chain_simulator.domain.events import (
    DayEvents,
    DemandEvent,
    EventTape,
    ShipmentReleaseEvent,
)
from supply_chain_simulator.domain.models import NetworkDefinition

DEMAND_STREAM = "demand"
EDGE_DELAYS_STREAM = "edge_delays"


def derive_stream_seed(replication_seed: int, stream_name: str) -> int:
    digest = hashlib.sha256(f"{replication_seed}:{stream_name}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big")


def generate_demand_events(
    demand_process: DemandProcessConfig,
    horizon_days: int,
    rng: random.Random,
) -> tuple[DemandEvent, ...]:
    events = []
    for day in range(1, horizon_days + 1):
        raw_quantity = round(rng.gauss(demand_process.mean_daily_demand, demand_process.standard_deviation))
        quantity = max(
            demand_process.minimum_daily_demand,
            min(demand_process.maximum_daily_demand, raw_quantity),
        )
        events.append(
            DemandEvent(
                day=day,
                destination_node_id=demand_process.destination_node_id,
                product_id=demand_process.product_id,
                quantity=quantity,
            )
        )
    return tuple(events)


def generate_edge_delay_draws(
    network_definition: NetworkDefinition,
    max_day: int,
    rng: random.Random,
) -> dict[int, dict[str, int]]:
    edge_ids = sorted(network_definition.edges)
    draws: dict[int, dict[str, int]] = {}
    for day in range(1, max_day + 1):
        day_draws: dict[str, int] = {}
        for edge_id in edge_ids:
            reliability = network_definition.get_edge(edge_id).reliability
            day_draws[edge_id] = 0 if rng.random() <= reliability else 1
        draws[day] = day_draws
    return draws


def generate_shipment_release_events(
    replenishment_plan: ReplenishmentPlanConfig,
    horizon_days: int,
) -> tuple[ShipmentReleaseEvent, ...]:
    events = []
    day = replenishment_plan.first_release_day
    while day <= horizon_days:
        events.append(
            ShipmentReleaseEvent(
                day=day,
                shipment_id=f"shipment_{day:03d}_001",
                product_id=replenishment_plan.product_id,
                quantity=replenishment_plan.shipment_quantity,
                origin_node_id=replenishment_plan.origin_node_id,
                destination_node_id=replenishment_plan.destination_node_id,
                due_day=day + replenishment_plan.due_offset_days,
                initial_route_edge_ids=tuple(replenishment_plan.initial_route_edge_ids),
            )
        )
        day += replenishment_plan.release_every_days
    return tuple(events)


def build_disrupted_event_tape(
    network_definition: NetworkDefinition,
    demand_process: DemandProcessConfig,
    replenishment_plan: ReplenishmentPlanConfig,
    scenario_config: ScenarioConfig,
    replication: int,
    base_seed: int,
    horizon_days: int,
    drain_days: int,
) -> EventTape:
    if replication < 1:
        raise ValueError("replication numbering starts at 1")

    replication_seed = base_seed + replication
    demand_rng = random.Random(derive_stream_seed(replication_seed, DEMAND_STREAM))
    edge_delay_rng = random.Random(derive_stream_seed(replication_seed, EDGE_DELAYS_STREAM))

    demand_events_by_day: dict[int, DemandEvent] = {
        event.day: event for event in generate_demand_events(demand_process, horizon_days, demand_rng)
    }
    release_events_by_day: dict[int, tuple[ShipmentReleaseEvent, ...]] = {}
    for event in generate_shipment_release_events(replenishment_plan, horizon_days):
        release_events_by_day[event.day] = (*release_events_by_day.get(event.day, ()), event)

    max_day = horizon_days + drain_days
    edge_delay_draws = generate_edge_delay_draws(network_definition, max_day, edge_delay_rng)

    shocks = build_shocks(scenario_config)
    known_shock_ids_by_day: dict[int, tuple[str, ...]] = {}
    for shock in shocks:
        known_shock_ids_by_day[shock.information_day] = (
            *known_shock_ids_by_day.get(shock.information_day, ()),
            shock.shock_id,
        )

    days = tuple(
        DayEvents(
            day=day,
            demand_events=((demand_events_by_day[day],) if day in demand_events_by_day else ()),
            shipment_release_events=release_events_by_day.get(day, ()),
            edge_extra_delay_days=edge_delay_draws[day],
            newly_known_shock_ids=tuple(sorted(known_shock_ids_by_day.get(day, ()))),
        )
        for day in range(1, max_day + 1)
    )

    return EventTape(
        scenario_id=scenario_config.scenario_id,
        replication=replication,
        seed=replication_seed,
        days=days,
        shocks=shocks,
    )


def build_undisrupted_event_tape(disrupted_tape: EventTape) -> EventTape:
    """Strips the designed shocks; demand, releases, and delays are unchanged."""
    undisrupted_days = tuple(
        DayEvents(
            day=day_events.day,
            demand_events=day_events.demand_events,
            shipment_release_events=day_events.shipment_release_events,
            edge_extra_delay_days=day_events.edge_extra_delay_days,
            newly_known_shock_ids=(),
        )
        for day_events in disrupted_tape.days
    )
    return EventTape(
        scenario_id=disrupted_tape.scenario_id,
        replication=disrupted_tape.replication,
        seed=disrupted_tape.seed,
        days=undisrupted_days,
        shocks=(),
    )
