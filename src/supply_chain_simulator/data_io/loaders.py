"""Loads YAML configuration and converts it into the immutable domain model.

Inside the data_io package, this module turns the network, scenario, policy,
and experiment YAML files under configs/ into typed, validated Pydantic
models, resolves the file paths an experiment references relative to that
experiment file, and combines everything into one ResolvedConfig. It then
converts a validated NetworkConfig into the frozen NetworkDefinition (Node,
Edge, Product) the simulation actually runs on, builds the day-0
SimulationState from it, and converts a ScenarioConfig's shocks into frozen
Shock domain objects. In the full system, this is the place a malformed,
incomplete, or internally inconsistent configuration is caught and reported
before any simulation code runs, and the only place configuration data turns
into domain objects. It does not perform the deeper graph-reachability
analysis that route enumeration needs — that belongs to simulation/routing.py
— and it does not implement any day-to-day simulation behavior itself.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml  # type: ignore[import-untyped]
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    StringConstraints,
    ValidationError,
    model_validator,
)

from supply_chain_simulator.domain.events import Shock, ShockType, TargetType
from supply_chain_simulator.domain.models import (
    Edge,
    NetworkDefinition,
    Node,
    NodeType,
    Product,
    TransportMode,
)
from supply_chain_simulator.domain.state import (
    CostCounters,
    OperationalEdgeState,
    OperationalNodeState,
    ServiceCounters,
    SimulationState,
)

IdentifierStr = Annotated[str, StringConstraints(pattern=r"^[a-z0-9_]+$", min_length=1)]
UnitReliability = Annotated[float, Field(ge=0.0, le=1.0)]


class ConfigurationError(Exception):
    """Raised when a configuration file is missing, malformed, or invalid."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


# --- configs/networks/*.yaml -------------------------------------------------


class UnitsConfig(_StrictModel):
    quantity: str
    distance: str
    time: str
    currency: str


class ProductConfig(_StrictModel):
    product_id: IdentifierStr
    name: str
    holding_cost_per_unit_day: NonNegativeFloat
    backlog_cost_per_unit_day: NonNegativeFloat
    late_penalty_per_unit_day: NonNegativeFloat


class NodeConfig(_StrictModel):
    node_id: IdentifierStr
    name: str
    node_type: Literal["SUPPLIER", "PORT", "HUB", "PLANT", "CUSTOMER"]
    latitude: float | None = None
    longitude: float | None = None
    storage_capacity: NonNegativeInt
    processing_capacity: NonNegativeInt
    source_capacity: NonNegativeInt

    @model_validator(mode="after")
    def _validate_source_capacity(self) -> NodeConfig:
        if self.node_type != "SUPPLIER" and self.source_capacity != 0:
            raise ValueError("source_capacity must be 0 for non-SUPPLIER nodes")
        return self


class EdgeConfig(_StrictModel):
    edge_id: IdentifierStr
    origin_node_id: IdentifierStr
    destination_node_id: IdentifierStr
    mode: Literal["ROAD", "RAIL", "SEA", "AIR"]
    distance_km: NonNegativeFloat
    base_lead_time_days: PositiveInt
    daily_capacity: NonNegativeInt
    unit_transport_cost: NonNegativeFloat
    reliability: UnitReliability
    emergency: bool

    @model_validator(mode="after")
    def _validate_distinct_endpoints(self) -> EdgeConfig:
        if self.origin_node_id == self.destination_node_id:
            raise ValueError("origin_node_id and destination_node_id must differ")
        return self


class InitialInventoryConfig(_StrictModel):
    node_id: IdentifierStr
    product_id: IdentifierStr
    quantity: NonNegativeInt


class DemandProcessConfig(_StrictModel):
    destination_node_id: IdentifierStr
    product_id: IdentifierStr
    distribution: Literal["TRUNCATED_NORMAL"]
    mean_daily_demand: NonNegativeFloat
    standard_deviation: NonNegativeFloat
    minimum_daily_demand: NonNegativeInt
    maximum_daily_demand: NonNegativeInt

    @model_validator(mode="after")
    def _validate_demand_bounds(self) -> DemandProcessConfig:
        if self.minimum_daily_demand > self.maximum_daily_demand:
            raise ValueError("minimum_daily_demand must not exceed maximum_daily_demand")
        return self


class ReplenishmentPlanConfig(_StrictModel):
    product_id: IdentifierStr
    origin_node_id: IdentifierStr
    destination_node_id: IdentifierStr
    first_release_day: PositiveInt
    release_every_days: PositiveInt
    shipment_quantity: PositiveInt
    due_offset_days: PositiveInt
    initial_route_edge_ids: Annotated[list[IdentifierStr], Field(min_length=1)]


class ActionCostsConfig(_StrictModel):
    reroute_cost_per_unit: NonNegativeFloat
    expedite_premium_per_unit: NonNegativeFloat


class NetworkConfig(_StrictModel):
    schema_version: Literal[1]
    network_id: IdentifierStr
    units: UnitsConfig
    products: Annotated[list[ProductConfig], Field(min_length=1)]
    nodes: Annotated[list[NodeConfig], Field(min_length=1)]
    edges: list[EdgeConfig]
    initial_inventory: list[InitialInventoryConfig]
    demand_process: DemandProcessConfig
    replenishment_plan: ReplenishmentPlanConfig
    action_costs: ActionCostsConfig

    @model_validator(mode="after")
    def _validate_cross_references(self) -> NetworkConfig:
        node_ids = {node.node_id for node in self.nodes}
        if len(node_ids) != len(self.nodes):
            raise ValueError("duplicate node_id values in nodes")

        edge_ids = {edge.edge_id for edge in self.edges}
        if len(edge_ids) != len(self.edges):
            raise ValueError("duplicate edge_id values in edges")

        product_ids = {product.product_id for product in self.products}
        if len(product_ids) != len(self.products):
            raise ValueError("duplicate product_id values in products")

        edges_by_id = {edge.edge_id: edge for edge in self.edges}
        for edge in self.edges:
            if edge.origin_node_id not in node_ids:
                raise ValueError(
                    f"edge {edge.edge_id} references unknown origin_node_id "
                    f"{edge.origin_node_id}"
                )
            if edge.destination_node_id not in node_ids:
                raise ValueError(
                    f"edge {edge.edge_id} references unknown destination_node_id "
                    f"{edge.destination_node_id}"
                )

        for inventory_line in self.initial_inventory:
            if inventory_line.node_id not in node_ids:
                raise ValueError(
                    f"initial_inventory references unknown node_id "
                    f"{inventory_line.node_id}"
                )
            if inventory_line.product_id not in product_ids:
                raise ValueError(
                    f"initial_inventory references unknown product_id "
                    f"{inventory_line.product_id}"
                )

        if self.demand_process.destination_node_id not in node_ids:
            raise ValueError("demand_process references unknown destination_node_id")
        if self.demand_process.product_id not in product_ids:
            raise ValueError("demand_process references unknown product_id")

        plan = self.replenishment_plan
        if plan.origin_node_id not in node_ids:
            raise ValueError("replenishment_plan references unknown origin_node_id")
        if plan.destination_node_id not in node_ids:
            raise ValueError("replenishment_plan references unknown destination_node_id")
        if plan.product_id not in product_ids:
            raise ValueError("replenishment_plan references unknown product_id")

        route_edges = []
        for edge_id in plan.initial_route_edge_ids:
            if edge_id not in edges_by_id:
                raise ValueError(
                    f"replenishment_plan references unknown edge_id {edge_id}"
                )
            route_edges.append(edges_by_id[edge_id])

        if route_edges[0].origin_node_id != plan.origin_node_id:
            raise ValueError(
                "replenishment_plan initial_route_edge_ids does not begin at "
                "origin_node_id"
            )
        if route_edges[-1].destination_node_id != plan.destination_node_id:
            raise ValueError(
                "replenishment_plan initial_route_edge_ids does not end at "
                "destination_node_id"
            )
        for previous_edge, next_edge in pairwise(route_edges):
            if previous_edge.destination_node_id != next_edge.origin_node_id:
                raise ValueError(
                    "replenishment_plan initial_route_edge_ids is not a continuous "
                    "path"
                )

        return self


# --- configs/scenarios/*.yaml ------------------------------------------------


class ShockConfig(_StrictModel):
    shock_id: IdentifierStr
    shock_type: Literal[
        "NODE_CLOSURE",
        "EDGE_CLOSURE",
        "NODE_CAPACITY_REDUCTION",
        "EDGE_CAPACITY_REDUCTION",
        "EDGE_LEAD_TIME_INCREASE",
        "EDGE_COST_INCREASE",
    ]
    target_type: Literal["NODE", "EDGE"]
    target_id: IdentifierStr
    physical_start_day: PositiveInt
    physical_end_day: PositiveInt
    information_day: PositiveInt
    capacity_multiplier: NonNegativeFloat = 1.0
    lead_time_multiplier: NonNegativeFloat = 1.0
    cost_multiplier: NonNegativeFloat = 1.0

    @model_validator(mode="after")
    def _validate_day_range(self) -> ShockConfig:
        if self.physical_end_day < self.physical_start_day:
            raise ValueError("physical_end_day must be >= physical_start_day")
        return self


class ScenarioConfig(_StrictModel):
    schema_version: Literal[1]
    scenario_id: IdentifierStr
    description: str
    shocks: Annotated[list[ShockConfig], Field(min_length=1)]

    @model_validator(mode="after")
    def _validate_unique_shock_ids(self) -> ScenarioConfig:
        shock_ids = [shock.shock_id for shock in self.shocks]
        if len(set(shock_ids)) != len(shock_ids):
            raise ValueError("duplicate shock_id values in shocks")
        return self


# --- configs/policies/*.yaml --------------------------------------------------


class HeuristicPolicyConfig(_StrictModel):
    schema_version: Literal[1]
    policy_id: IdentifierStr
    policy_type: Literal["HEURISTIC"]
    expedite_trigger_lateness_days: NonNegativeInt
    cost_tolerance: PositiveFloat


class LLMPolicyConfig(_StrictModel):
    schema_version: Literal[1]
    policy_id: IdentifierStr
    policy_type: Literal["LLM"]
    provider: Literal["OPENAI"]
    api_key_environment_variable: str
    model_environment_variable: str
    execution_mode: Literal["LIVE", "REPLAY"]
    temperature: NonNegativeFloat
    max_tool_calls: PositiveInt
    max_output_tokens: PositiveInt
    request_timeout_seconds: PositiveInt
    max_retries: NonNegativeInt
    store_provider_response: bool
    fallback_policy: Literal["HEURISTIC", "WAIT"]
    replay_trace_path: str | None = None

    @model_validator(mode="after")
    def _validate_replay_requirements(self) -> LLMPolicyConfig:
        if self.execution_mode == "REPLAY" and not self.replay_trace_path:
            raise ValueError(
                "replay_trace_path is required when execution_mode is REPLAY"
            )
        return self


# --- configs/experiments/*.yaml -----------------------------------------------


class PolicyConfigPathsConfig(_StrictModel):
    heuristic: str
    llm_agent: str


class ExperimentConfig(_StrictModel):
    schema_version: Literal[1]
    experiment_id: IdentifierStr
    network_config: str
    scenario_config: str
    policy_configs: PolicyConfigPathsConfig
    warmup_days: PositiveInt
    horizon_days: PositiveInt
    drain_days: NonNegativeInt
    terminal_penalty_days: NonNegativeInt
    replications: PositiveInt
    base_seed: int
    counterfactual_mode: Literal["POLICY_SPECIFIC"]
    fail_fast: bool
    output_root: str
    write_event_tapes: bool
    write_daily_metrics: bool
    write_decision_traces: bool
    write_llm_interactions: bool

    @model_validator(mode="after")
    def _validate_horizon(self) -> ExperimentConfig:
        if self.warmup_days >= self.horizon_days:
            raise ValueError("warmup_days must be less than horizon_days")
        return self


# --- combined, path-resolved configuration ------------------------------------


class ResolvedConfig(_StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True, arbitrary_types_allowed=True)

    experiment: ExperimentConfig
    network: NetworkConfig
    scenario: ScenarioConfig
    heuristic_policy: HeuristicPolicyConfig
    llm_policy: LLMPolicyConfig

    experiment_config_path: Path
    network_config_path: Path
    scenario_config_path: Path
    heuristic_config_path: Path
    llm_config_path: Path
    output_root: Path

    @model_validator(mode="after")
    def _validate_shocks_against_network_and_warmup(self) -> ResolvedConfig:
        node_ids = {node.node_id for node in self.network.nodes}
        edge_ids = {edge.edge_id for edge in self.network.edges}
        for shock in self.scenario.shocks:
            if shock.physical_start_day <= self.experiment.warmup_days:
                raise ValueError(
                    f"shock {shock.shock_id} starts on day "
                    f"{shock.physical_start_day}, which is not after "
                    f"warmup_days={self.experiment.warmup_days}"
                )
            if shock.target_type == "NODE" and shock.target_id not in node_ids:
                raise ValueError(
                    f"shock {shock.shock_id} targets unknown node {shock.target_id}"
                )
            if shock.target_type == "EDGE" and shock.target_id not in edge_ids:
                raise ValueError(
                    f"shock {shock.shock_id} targets unknown edge {shock.target_id}"
                )
        return self


# --- loading and path resolution ----------------------------------------------


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(f"cannot read configuration file {path}: {exc}") from exc
    try:
        data = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigurationError(
            f"configuration file {path} must contain a mapping at the top level"
        )
    return data


def _load_model[ConfigT: BaseModel](path: Path, model: type[ConfigT]) -> ConfigT:
    data = _load_yaml_mapping(path)
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise ConfigurationError(f"invalid configuration in {path}:\n{exc}") from exc


def load_network_config(path: Path) -> NetworkConfig:
    return _load_model(path, NetworkConfig)


def load_scenario_config(path: Path) -> ScenarioConfig:
    return _load_model(path, ScenarioConfig)


def load_heuristic_policy_config(path: Path) -> HeuristicPolicyConfig:
    return _load_model(path, HeuristicPolicyConfig)


def load_llm_policy_config(path: Path) -> LLMPolicyConfig:
    return _load_model(path, LLMPolicyConfig)


def load_experiment_config(path: Path) -> ExperimentConfig:
    return _load_model(path, ExperimentConfig)


def _resolve_within_repo(path: Path, repo_root: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(repo_root):
        raise ConfigurationError(
            f"path {resolved} escapes the repository root {repo_root}"
        )
    return resolved


def _resolve_existing_config_file(path: Path, repo_root: Path) -> Path:
    resolved = _resolve_within_repo(path, repo_root)
    if not resolved.is_file():
        raise ConfigurationError(f"referenced configuration file does not exist: {resolved}")
    return resolved


def resolve_config(experiment_config_path: Path, repo_root: Path) -> ResolvedConfig:
    experiment_config_path = _resolve_existing_config_file(experiment_config_path, repo_root)
    experiment = load_experiment_config(experiment_config_path)
    experiment_dir = experiment_config_path.parent

    network_path = _resolve_existing_config_file(
        experiment_dir / experiment.network_config, repo_root
    )
    scenario_path = _resolve_existing_config_file(
        experiment_dir / experiment.scenario_config, repo_root
    )
    heuristic_path = _resolve_existing_config_file(
        experiment_dir / experiment.policy_configs.heuristic, repo_root
    )
    llm_path = _resolve_existing_config_file(
        experiment_dir / experiment.policy_configs.llm_agent, repo_root
    )
    output_root = _resolve_within_repo(experiment_dir / experiment.output_root, repo_root)

    return ResolvedConfig(
        experiment=experiment,
        network=load_network_config(network_path),
        scenario=load_scenario_config(scenario_path),
        heuristic_policy=load_heuristic_policy_config(heuristic_path),
        llm_policy=load_llm_policy_config(llm_path),
        experiment_config_path=experiment_config_path,
        network_config_path=network_path,
        scenario_config_path=scenario_path,
        heuristic_config_path=heuristic_path,
        llm_config_path=llm_path,
        output_root=output_root,
    )


# --- config-to-domain conversion and day-0 state construction -----------------


def build_network_definition(network_config: NetworkConfig) -> NetworkDefinition:
    nodes = {
        node.node_id: Node(
            node_id=node.node_id,
            name=node.name,
            node_type=NodeType(node.node_type),
            latitude=node.latitude,
            longitude=node.longitude,
            storage_capacity=node.storage_capacity,
            processing_capacity=node.processing_capacity,
            source_capacity=node.source_capacity,
        )
        for node in network_config.nodes
    }
    edges = {
        edge.edge_id: Edge(
            edge_id=edge.edge_id,
            origin_node_id=edge.origin_node_id,
            destination_node_id=edge.destination_node_id,
            mode=TransportMode(edge.mode),
            distance_km=edge.distance_km,
            base_lead_time_days=edge.base_lead_time_days,
            daily_capacity=edge.daily_capacity,
            unit_transport_cost=edge.unit_transport_cost,
            reliability=edge.reliability,
            emergency=edge.emergency,
        )
        for edge in network_config.edges
    }
    products = {
        product.product_id: Product(
            product_id=product.product_id,
            name=product.name,
            holding_cost_per_unit_day=product.holding_cost_per_unit_day,
            backlog_cost_per_unit_day=product.backlog_cost_per_unit_day,
            late_penalty_per_unit_day=product.late_penalty_per_unit_day,
        )
        for product in network_config.products
    }
    return NetworkDefinition(nodes=nodes, edges=edges, products=products)


def build_initial_state(
    network_definition: NetworkDefinition, network_config: NetworkConfig
) -> SimulationState:
    """Builds the day-0 state: normal network, configured inventory, nothing else.

    Matches CLAUDE.md section 13.1 exactly: the base network is normal, all
    operational states use defaults, backlog is zero, there are no shipments,
    costs and service counters are zero, and no shock is active or known.
    """
    inventory: dict[str, dict[str, int]] = {
        node_id: dict.fromkeys(network_definition.products, 0)
        for node_id in network_definition.nodes
    }
    for line in network_config.initial_inventory:
        inventory[line.node_id][line.product_id] += line.quantity

    backlog: dict[str, dict[str, int]] = {
        node_id: dict.fromkeys(network_definition.products, 0)
        for node_id in network_definition.nodes
    }

    return SimulationState(
        day=0,
        network_definition=network_definition,
        node_operational_state={
            node_id: OperationalNodeState() for node_id in network_definition.nodes
        },
        edge_operational_state={
            edge_id: OperationalEdgeState() for edge_id in network_definition.edges
        },
        inventory=inventory,
        backlog=backlog,
        shipments={},
        costs=CostCounters(),
        service=ServiceCounters(),
        daily_edge_used_capacity=dict.fromkeys(network_definition.edges, 0),
        daily_node_used_processing=dict.fromkeys(network_definition.nodes, 0),
    )


def build_shocks(scenario_config: ScenarioConfig) -> tuple[Shock, ...]:
    return tuple(
        Shock(
            shock_id=shock.shock_id,
            shock_type=ShockType(shock.shock_type),
            target_type=TargetType(shock.target_type),
            target_id=shock.target_id,
            physical_start_day=shock.physical_start_day,
            physical_end_day=shock.physical_end_day,
            information_day=shock.information_day,
            capacity_multiplier=shock.capacity_multiplier,
            lead_time_multiplier=shock.lead_time_multiplier,
            cost_multiplier=shock.cost_multiplier,
        )
        for shock in scenario_config.shocks
    )
