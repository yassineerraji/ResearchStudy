"""Mutable run state, per-day records, and the final result of one run.

Inside the domain package, this module defines everything that changes as a
simulation progresses: shipment lifecycle and position, per-day operational
overrides on top of the immutable network, inventory and backlog, running
cost and service counters, and the complete SimulationState that bundles them
for one day. It also defines the historical, read-only records a run
produces — DailyMetrics for one day and SimulationResult for a whole run. In
the full system, SimulationState is what the engine mutates and what an
observation is built from, but policies never receive it directly. This
module does not decide what happens to the state (that is
simulation/transition.py's job) and does not itself enforce day-to-day
invariants across mutations — those are asserted by the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from supply_chain_simulator.domain.models import NetworkDefinition


class ShipmentStatus(Enum):
    """Physical status of a shipment on a given day."""

    AT_NODE = "AT_NODE"
    IN_TRANSIT = "IN_TRANSIT"
    DELIVERED = "DELIVERED"


@dataclass(slots=True)
class OperationalNodeState:
    available: bool = True
    processing_capacity_multiplier: float = 1.0


@dataclass(slots=True)
class OperationalEdgeState:
    available: bool = True
    capacity_multiplier: float = 1.0
    lead_time_multiplier: float = 1.0
    cost_multiplier: float = 1.0


@dataclass(slots=True)
class Shipment:
    shipment_id: str
    product_id: str
    quantity: int
    origin_node_id: str
    destination_node_id: str
    release_day: int
    due_day: int
    planned_route_edge_ids: tuple[str, ...]
    next_edge_index: int
    status: ShipmentStatus
    current_node_id: str | None
    current_edge_id: str | None
    edge_entry_day: int | None
    edge_arrival_day: int | None
    reroute_count: int
    expedite_count: int
    capacity_wait_days: int
    delivered_day: int | None

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"shipment {self.shipment_id}: quantity must be positive")

        if self.status is ShipmentStatus.AT_NODE:
            if self.current_node_id is None or self.current_edge_id is not None:
                raise ValueError(
                    f"shipment {self.shipment_id}: AT_NODE requires a current_node_id "
                    f"and no current_edge_id"
                )
        elif self.status is ShipmentStatus.IN_TRANSIT:
            if (
                self.current_node_id is not None
                or self.current_edge_id is None
                or self.edge_arrival_day is None
            ):
                raise ValueError(
                    f"shipment {self.shipment_id}: IN_TRANSIT requires a current_edge_id "
                    f"and edge_arrival_day and no current_node_id"
                )
        elif self.status is ShipmentStatus.DELIVERED and (
            self.current_node_id != self.destination_node_id
            or self.current_edge_id is not None
            or self.delivered_day is None
        ):
            raise ValueError(
                f"shipment {self.shipment_id}: DELIVERED requires current_node_id to "
                f"be the destination, no current_edge_id, and a delivered_day"
            )


@dataclass(slots=True)
class CostCounters:
    transport: float = 0.0
    reroute: float = 0.0
    expedite: float = 0.0
    holding: float = 0.0
    backlog: float = 0.0
    late: float = 0.0
    terminal: float = 0.0


@dataclass(slots=True)
class ServiceCounters:
    total_demand_units: int = 0
    same_day_fulfilled_units: int = 0
    backlog_fulfilled_units: int = 0
    delivered_shipment_units: int = 0
    late_delivered_units: int = 0
    total_lateness_unit_days: int = 0
    decision_count: int = 0
    valid_action_count: int = 0
    invalid_action_count: int = 0
    abstention_count: int = 0
    fallback_count: int = 0
    wait_count: int = 0
    reroute_count: int = 0
    expedite_count: int = 0
    expedited_units: int = 0


@dataclass(slots=True)
class SimulationState:
    day: int
    network_definition: NetworkDefinition
    node_operational_state: dict[str, OperationalNodeState]
    edge_operational_state: dict[str, OperationalEdgeState]
    inventory: dict[str, dict[str, int]]
    backlog: dict[str, dict[str, int]]
    shipments: dict[str, Shipment]
    active_shock_ids: set[str] = field(default_factory=set)
    known_shock_ids: set[str] = field(default_factory=set)
    costs: CostCounters = field(default_factory=CostCounters)
    service: ServiceCounters = field(default_factory=ServiceCounters)
    daily_edge_used_capacity: dict[str, int] = field(default_factory=dict)
    daily_node_used_processing: dict[str, int] = field(default_factory=dict)
    pre_shock_inventory: dict[str, dict[str, int]] = field(default_factory=dict)
    pre_shock_backlog: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DailyMetrics:
    experiment_id: str
    scenario_id: str
    replication: int
    policy: str
    run_kind: str
    day: int
    inventory_units: int
    backlog_units: int
    shipments_at_node: int
    shipments_in_transit: int
    shipments_delivered: int
    daily_demand_units: int
    daily_same_day_fulfilled_units: int
    daily_backlog_fulfilled_units: int
    daily_transport_cost: float
    daily_reroute_cost: float
    daily_expedite_cost: float
    daily_holding_cost: float
    daily_backlog_cost: float
    daily_late_cost: float
    cumulative_total_cost: float
    active_shock_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SimulationResult:
    experiment_id: str
    scenario_id: str
    replication: int
    policy_name: str
    run_kind: str
    final_day: int
    final_state: SimulationState
    daily_metrics: tuple[DailyMetrics, ...]
    terminated_with_unresolved_state: bool
