"""Shared cost calculations, charged identically for every policy and branch.

Inside the simulation package, this module owns every rule that turns a
physical event — an edge entry, a reroute, an expedite, a day's held
inventory or backlog, a late delivery, an unresolved end-of-run state — into
a cost, and adds it to a SimulationState's running CostCounters. In the full
system, this is the one place the total cost of a run is defined, so that
the heuristic and the LLM agent are always scored by exactly the same
arithmetic. It does not decide when an edge entry, reroute, or delivery
happens — simulation/transition.py calls these functions at the right point
in the daily sequence — and it does not estimate future cost, only charges
cost that has already actually occurred.
"""

from __future__ import annotations

from supply_chain_simulator.domain.state import (
    Shipment,
    ShipmentStatus,
    SimulationState,
)


def charge_edge_entry_transport_cost(
    state: SimulationState, quantity: int, effective_unit_transport_cost: float
) -> float:
    cost = quantity * effective_unit_transport_cost
    state.costs.transport += cost
    return cost


def charge_reroute_cost(state: SimulationState, quantity: int, reroute_cost_per_unit: float) -> float:
    cost = quantity * reroute_cost_per_unit
    state.costs.reroute += cost
    return cost


def charge_expedite_cost(
    state: SimulationState, quantity: int, expedite_premium_per_unit: float
) -> float:
    cost = quantity * expedite_premium_per_unit
    state.costs.expedite += cost
    return cost


def charge_end_of_day_holding_cost(state: SimulationState) -> float:
    total = 0.0
    for product_quantities in state.inventory.values():
        for product_id, quantity in product_quantities.items():
            product = state.network_definition.get_product(product_id)
            total += quantity * product.holding_cost_per_unit_day
    state.costs.holding += total
    return total


def charge_end_of_day_backlog_cost(state: SimulationState) -> float:
    total = 0.0
    for product_quantities in state.backlog.values():
        for product_id, quantity in product_quantities.items():
            product = state.network_definition.get_product(product_id)
            total += quantity * product.backlog_cost_per_unit_day
    state.costs.backlog += total
    return total


def charge_delivery_late_penalty(state: SimulationState, shipment: Shipment) -> float:
    if shipment.delivered_day is None:
        raise ValueError(
            f"shipment {shipment.shipment_id} has no delivered_day; cannot charge late penalty"
        )
    lateness_days = max(0, shipment.delivered_day - shipment.due_day)
    product = state.network_definition.get_product(shipment.product_id)
    cost = shipment.quantity * lateness_days * product.late_penalty_per_unit_day
    state.costs.late += cost
    return cost


def charge_terminal_cost(state: SimulationState, terminal_penalty_days: int) -> float:
    total = 0.0

    for product_quantities in state.backlog.values():
        for product_id, quantity in product_quantities.items():
            product = state.network_definition.get_product(product_id)
            total += quantity * product.backlog_cost_per_unit_day * terminal_penalty_days

    for shipment in state.shipments.values():
        if shipment.status is ShipmentStatus.DELIVERED:
            continue
        product = state.network_definition.get_product(shipment.product_id)
        lateness_days = max(1, state.day - shipment.due_day)
        total += shipment.quantity * product.late_penalty_per_unit_day * lateness_days

    state.costs.terminal += total
    return total


def total_cost(state: SimulationState) -> float:
    costs = state.costs
    return (
        costs.transport
        + costs.reroute
        + costs.expedite
        + costs.holding
        + costs.backlog
        + costs.late
        + costs.terminal
    )
