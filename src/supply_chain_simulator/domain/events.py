"""Exogenous-event definitions: demand, releases, shocks, and the event tape.

Inside the domain package, this module defines the events that happen to the
simulation regardless of which policy is deciding — realized demand,
scheduled shipment releases, and designed disruptions (shocks) — plus the
per-day bundle (DayEvents) and the full per-replication EventTape that holds
them. In the full system, this is the shared vocabulary that lets the
experiment package generate one policy-independent random future and replay
it unchanged across every branch of a paired comparison. It does not generate
any of these events itself (that is experiments/event_tape.py) and does not
know anything about how an event is later applied to a SimulationState.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ShockType(Enum):
    """Kind of physical effect a designed disruption has on the network."""

    NODE_CLOSURE = "NODE_CLOSURE"
    EDGE_CLOSURE = "EDGE_CLOSURE"
    NODE_CAPACITY_REDUCTION = "NODE_CAPACITY_REDUCTION"
    EDGE_CAPACITY_REDUCTION = "EDGE_CAPACITY_REDUCTION"
    EDGE_LEAD_TIME_INCREASE = "EDGE_LEAD_TIME_INCREASE"
    EDGE_COST_INCREASE = "EDGE_COST_INCREASE"
    DEMAND_SPIKE = "DEMAND_SPIKE"
    DEMAND_DROP = "DEMAND_DROP"
    SUPPLIER_CAPACITY_REDUCTION = "SUPPLIER_CAPACITY_REDUCTION"


class TargetType(Enum):
    """Whether a shock targets a node, an edge, or the demand process."""

    NODE = "NODE"
    EDGE = "EDGE"
    DEMAND = "DEMAND"


@dataclass(frozen=True, slots=True)
class DemandEvent:
    day: int
    destination_node_id: str
    product_id: str
    quantity: int

    def __post_init__(self) -> None:
        if self.quantity < 0:
            raise ValueError("DemandEvent quantity must be non-negative")


@dataclass(frozen=True, slots=True)
class ShipmentReleaseEvent:
    day: int
    shipment_id: str
    product_id: str
    quantity: int
    origin_node_id: str
    destination_node_id: str
    due_day: int
    initial_route_edge_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("ShipmentReleaseEvent quantity must be positive")
        if not self.initial_route_edge_ids:
            raise ValueError("ShipmentReleaseEvent initial_route_edge_ids must not be empty")


@dataclass(frozen=True, slots=True)
class Shock:
    shock_id: str
    shock_type: ShockType
    target_type: TargetType
    target_id: str
    physical_start_day: int
    physical_end_day: int
    information_day: int
    capacity_multiplier: float = 1.0
    lead_time_multiplier: float = 1.0
    cost_multiplier: float = 1.0
    demand_multiplier: float = 1.0
    event_group_id: str | None = None

    def __post_init__(self) -> None:
        if self.physical_end_day < self.physical_start_day:
            raise ValueError(
                f"shock {self.shock_id}: physical_end_day must be >= physical_start_day"
            )


@dataclass(frozen=True, slots=True)
class DayEvents:
    day: int
    demand_events: tuple[DemandEvent, ...]
    shipment_release_events: tuple[ShipmentReleaseEvent, ...]
    edge_extra_delay_days: dict[str, int]
    newly_known_shock_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EventTape:
    scenario_id: str
    replication: int
    seed: int
    days: tuple[DayEvents, ...]
    shocks: tuple[Shock, ...]
