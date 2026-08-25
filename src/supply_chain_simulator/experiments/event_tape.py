"""Derives per-replication random streams and draws demand, transport delays, realized shocks, and shipment quantities from them into one EventTape, the policy-independent random future every branch shares."""

from __future__ import annotations

import hashlib
import random

from supply_chain_simulator.data_io.loaders import (
    DemandProcessConfig,
    ReplenishmentPlanConfig,
    ScenarioConfig,
    ShockConfig,
)
from supply_chain_simulator.domain.events import (
    DayEvents,
    DemandEvent,
    EventTape,
    ShipmentReleaseEvent,
    Shock,
    ShockType,
    TargetType,
)
from supply_chain_simulator.domain.models import NetworkDefinition

DEMAND_STREAM = "demand"
EDGE_DELAYS_STREAM = "edge_delays"
SHOCK_REALIZATION_STREAM = "shock_realization"
RELEASE_QUANTITY_STREAM = "release_quantity"


def derive_stream_seed(replication_seed: int, stream_name: str) -> int:
    digest = hashlib.sha256(f"{replication_seed}:{stream_name}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big")


def _draw_truncated_normal(mean: float, std: float, minimum: int, maximum: int, rng: random.Random) -> int:
    raw = round(rng.gauss(mean, std))
    return max(minimum, min(maximum, raw))


def generate_demand_events(
    demand_process: DemandProcessConfig,
    horizon_days: int,
    rng: random.Random,
    demand_shocks: tuple[Shock, ...] = (),
) -> tuple[DemandEvent, ...]:
    events = []
    for day in range(1, horizon_days + 1):
        multiplier = 1.0
        for shock in demand_shocks:
            if (
                shock.target_type is TargetType.DEMAND
                and shock.target_id == demand_process.destination_node_id
                and shock.physical_start_day <= day <= shock.physical_end_day
            ):
                multiplier *= shock.demand_multiplier

        mean = demand_process.mean_daily_demand * multiplier
        minimum_bound = round(demand_process.minimum_daily_demand * multiplier)
        maximum_bound = round(demand_process.maximum_daily_demand * multiplier)
        quantity = _draw_truncated_normal(
            mean, demand_process.standard_deviation, minimum_bound, maximum_bound, rng
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


def realize_release_quantity(mean: float, std: float, minimum: int, maximum: int, rng: random.Random) -> int:
    return _draw_truncated_normal(mean, std, minimum, maximum, rng)


def generate_shipment_release_events(
    replenishment_plan: ReplenishmentPlanConfig,
    horizon_days: int,
    network_definition: NetworkDefinition,
    rng: random.Random,
) -> tuple[ShipmentReleaseEvent, ...]:
    """Draws each release's quantity from the release_quantity stream (V2 §V2.3.5),
    then clamps it to the initial route's static edge/node capacity so every
    generated event stays structurally feasible, exactly as V1's fixed
    40-unit quantity always was by construction.
    """
    route_edge_ids = tuple(replenishment_plan.initial_route_edge_ids)
    route_edges = [network_definition.get_edge(edge_id) for edge_id in route_edge_ids]
    route_node_ids = {edge.origin_node_id for edge in route_edges} | {
        edge.destination_node_id for edge in route_edges
    }
    feasibility_ceiling = min(
        [edge.daily_capacity for edge in route_edges]
        + [network_definition.get_node(node_id).processing_capacity for node_id in route_node_ids]
    )

    events = []
    day = replenishment_plan.first_release_day
    while day <= horizon_days:
        raw_quantity = realize_release_quantity(
            replenishment_plan.shipment_quantity_mean,
            replenishment_plan.shipment_quantity_std,
            replenishment_plan.minimum_shipment_quantity,
            replenishment_plan.maximum_shipment_quantity,
            rng,
        )
        quantity = min(raw_quantity, feasibility_ceiling)
        events.append(
            ShipmentReleaseEvent(
                day=day,
                shipment_id=f"shipment_{day:03d}_001",
                product_id=replenishment_plan.product_id,
                quantity=quantity,
                origin_node_id=replenishment_plan.origin_node_id,
                destination_node_id=replenishment_plan.destination_node_id,
                due_day=day + replenishment_plan.due_offset_days,
                initial_route_edge_ids=route_edge_ids,
            )
        )
        day += replenishment_plan.release_every_days
    return tuple(events)


# --- shock template realization (V2 §V2.3.3, §V2.3.4) --------------------------


def _realize_from_start_day(template: ShockConfig, physical_start_day: int, rng: random.Random) -> Shock:
    duration_days = _draw_truncated_normal(
        template.duration_mean_days,
        template.duration_std_days,
        template.minimum_duration_days,
        template.maximum_duration_days,
        rng,
    )
    physical_end_day = physical_start_day + duration_days - 1

    information_delay = rng.randint(0, template.max_information_delay_days)
    information_day = min(physical_start_day + information_delay, physical_end_day)

    return Shock(
        shock_id=template.shock_id,
        shock_type=ShockType(template.shock_type),
        target_type=TargetType(template.target_type),
        target_id=template.target_id,
        physical_start_day=physical_start_day,
        physical_end_day=physical_end_day,
        information_day=information_day,
        capacity_multiplier=template.capacity_multiplier,
        lead_time_multiplier=template.lead_time_multiplier,
        cost_multiplier=template.cost_multiplier,
        demand_multiplier=template.demand_multiplier,
        event_group_id=template.event_group_id,
    )


def realize_shock(template: ShockConfig, rng: random.Random) -> Shock:
    """Draws start-day jitter, then duration, then information delay — that
    fixed order, per V2 §V2.3.3.
    """
    start_day_jitter = rng.randint(-template.start_day_jitter_days, template.start_day_jitter_days)
    return _realize_from_start_day(template, template.planned_start_day + start_day_jitter, rng)


def realize_shock_group(templates: list[ShockConfig], rng: random.Random) -> list[Shock]:
    """One shared start-day jitter draw for the whole group; each member's
    duration and information delay are still drawn independently, in
    ascending shock_id order (V2 §V2.3.4).
    """
    ordered = sorted(templates, key=lambda t: t.shock_id)
    shared_jitter = rng.randint(-ordered[0].start_day_jitter_days, ordered[0].start_day_jitter_days)
    return [
        _realize_from_start_day(template, template.planned_start_day + shared_jitter, rng)
        for template in ordered
    ]


def realize_all_shocks(scenario_config: ScenarioConfig, rng: random.Random) -> tuple[Shock, ...]:
    """Realizes every shock template in one scenario. Groups (by
    event_group_id) are processed in ascending lexicographic group-id order;
    ungrouped templates are each their own one-member group and are realized
    afterward, in ascending shock_id order — V2 §V2.3.4's "null sorts last"
    rule. The returned tuple is re-sorted by shock_id for a stable output
    order; that sort happens only after every draw has already been made in
    the order above, so it does not affect reproducibility.
    """
    grouped: dict[str, list[ShockConfig]] = {}
    ungrouped: list[ShockConfig] = []
    for template in scenario_config.shocks:
        if template.event_group_id is None:
            ungrouped.append(template)
        else:
            grouped.setdefault(template.event_group_id, []).append(template)

    realized: list[Shock] = []
    for group_id in sorted(grouped):
        realized.extend(realize_shock_group(grouped[group_id], rng))
    for template in sorted(ungrouped, key=lambda t: t.shock_id):
        realized.append(realize_shock(template, rng))

    return tuple(sorted(realized, key=lambda shock: shock.shock_id))


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
    shock_realization_rng = random.Random(derive_stream_seed(replication_seed, SHOCK_REALIZATION_STREAM))
    release_quantity_rng = random.Random(derive_stream_seed(replication_seed, RELEASE_QUANTITY_STREAM))

    shocks = realize_all_shocks(scenario_config, shock_realization_rng)
    demand_shocks = tuple(shock for shock in shocks if shock.target_type is TargetType.DEMAND)

    demand_events_by_day: dict[int, DemandEvent] = {
        event.day: event
        for event in generate_demand_events(demand_process, horizon_days, demand_rng, demand_shocks)
    }
    release_events_by_day: dict[int, tuple[ShipmentReleaseEvent, ...]] = {}
    for event in generate_shipment_release_events(
        replenishment_plan, horizon_days, network_definition, release_quantity_rng
    ):
        release_events_by_day[event.day] = (*release_events_by_day.get(event.day, ()), event)

    max_day = horizon_days + drain_days
    edge_delay_draws = generate_edge_delay_draws(network_definition, max_day, edge_delay_rng)

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
