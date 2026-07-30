"""Immutable business definitions: the network, its nodes, edges, and product.

Inside the domain package, this module defines the permanent, never-mutated
vocabulary of the supply chain — node and transport-mode enums, the frozen
Node/Edge/Product entities, the NetworkDefinition that bundles them into one
lookup-able network, and the Route descriptor used to describe a path through
it. In the full system, this is the shared ground truth that the physics, the
observation builder, and both policies all agree on: a node or an edge means
exactly the same thing everywhere. It does not hold any per-day operational
status (availability, multipliers, capacity usage) — that belongs to
domain/state.py — and it never changes once loaded for a run.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class NodeType(Enum):
    """Operational role of a network location."""

    SUPPLIER = "SUPPLIER"
    PORT = "PORT"
    HUB = "HUB"
    PLANT = "PLANT"
    CUSTOMER = "CUSTOMER"


class TransportMode(Enum):
    """Mode of transport carried by a transport lane."""

    ROAD = "ROAD"
    RAIL = "RAIL"
    SEA = "SEA"
    AIR = "AIR"


@dataclass(frozen=True, slots=True)
class Node:
    node_id: str
    name: str
    node_type: NodeType
    latitude: float | None
    longitude: float | None
    storage_capacity: int
    processing_capacity: int
    source_capacity: int

    def __post_init__(self) -> None:
        if self.storage_capacity < 0:
            raise ValueError(f"node {self.node_id}: storage_capacity must be non-negative")
        if self.processing_capacity < 0:
            raise ValueError(f"node {self.node_id}: processing_capacity must be non-negative")
        if self.source_capacity < 0:
            raise ValueError(f"node {self.node_id}: source_capacity must be non-negative")
        if self.node_type is not NodeType.SUPPLIER and self.source_capacity != 0:
            raise ValueError(
                f"node {self.node_id}: source_capacity must be 0 for non-SUPPLIER nodes"
            )


@dataclass(frozen=True, slots=True)
class Edge:
    edge_id: str
    origin_node_id: str
    destination_node_id: str
    mode: TransportMode
    distance_km: float
    base_lead_time_days: int
    daily_capacity: int
    unit_transport_cost: float
    reliability: float
    emergency: bool

    def __post_init__(self) -> None:
        if self.origin_node_id == self.destination_node_id:
            raise ValueError(f"edge {self.edge_id}: origin and destination must differ")
        if self.distance_km < 0:
            raise ValueError(f"edge {self.edge_id}: distance_km must be non-negative")
        if self.base_lead_time_days < 1:
            raise ValueError(f"edge {self.edge_id}: base_lead_time_days must be at least 1")
        if self.daily_capacity < 0:
            raise ValueError(f"edge {self.edge_id}: daily_capacity must be non-negative")
        if self.unit_transport_cost < 0:
            raise ValueError(f"edge {self.edge_id}: unit_transport_cost must be non-negative")
        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError(f"edge {self.edge_id}: reliability must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class Product:
    product_id: str
    name: str
    holding_cost_per_unit_day: float
    backlog_cost_per_unit_day: float
    late_penalty_per_unit_day: float

    def __post_init__(self) -> None:
        if self.holding_cost_per_unit_day < 0:
            raise ValueError(
                f"product {self.product_id}: holding_cost_per_unit_day must be non-negative"
            )
        if self.backlog_cost_per_unit_day < 0:
            raise ValueError(
                f"product {self.product_id}: backlog_cost_per_unit_day must be non-negative"
            )
        if self.late_penalty_per_unit_day < 0:
            raise ValueError(
                f"product {self.product_id}: late_penalty_per_unit_day must be non-negative"
            )


@dataclass(frozen=True, slots=True)
class NetworkDefinition:
    nodes: dict[str, Node]
    edges: dict[str, Edge]
    products: dict[str, Product]

    def __post_init__(self) -> None:
        for node_id, node in self.nodes.items():
            if node_id != node.node_id:
                raise ValueError(f"nodes key {node_id} does not match node_id {node.node_id}")
        for edge_id, edge in self.edges.items():
            if edge_id != edge.edge_id:
                raise ValueError(f"edges key {edge_id} does not match edge_id {edge.edge_id}")
            if edge.origin_node_id not in self.nodes:
                raise ValueError(
                    f"edge {edge.edge_id} references unknown origin_node_id "
                    f"{edge.origin_node_id}"
                )
            if edge.destination_node_id not in self.nodes:
                raise ValueError(
                    f"edge {edge.edge_id} references unknown destination_node_id "
                    f"{edge.destination_node_id}"
                )
        for product_id, product in self.products.items():
            if product_id != product.product_id:
                raise ValueError(
                    f"products key {product_id} does not match product_id {product.product_id}"
                )

    def get_node(self, node_id: str) -> Node:
        try:
            return self.nodes[node_id]
        except KeyError:
            raise KeyError(f"unknown node_id: {node_id}") from None

    def get_edge(self, edge_id: str) -> Edge:
        try:
            return self.edges[edge_id]
        except KeyError:
            raise KeyError(f"unknown edge_id: {edge_id}") from None

    def get_product(self, product_id: str) -> Product:
        try:
            return self.products[product_id]
        except KeyError:
            raise KeyError(f"unknown product_id: {product_id}") from None


@dataclass(frozen=True, slots=True)
class Route:
    route_id: str
    edge_ids: tuple[str, ...]
    node_ids: tuple[str, ...]
    contains_emergency_edge: bool
