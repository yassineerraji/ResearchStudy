// Types for the backend's /api/v1 responses (webapp/backend/app/schemas).
//
// Nested manifest/summary/network/decision fields are typed loosely
// (Record<string, unknown> or narrow local shapes only where a component
// actually reads specific fields) because the backend deliberately
// passes through supply_chain_simulator's own already-serialized output
// files rather than re-typing their schema — see
// webapp/backend/app/schemas/gallery.py's module docstring. Re-typing
// everything field-for-field here would duplicate a schema that already
// lives in the research package and would drift as it evolves.

export interface Manifest {
  experiment_id: string
  created_at_utc: string
  replications: number
  llm_execution_mode: string
  llm_model: string | null
  base_seed: number
  [key: string]: unknown
}

export interface ExperimentSummary {
  replication_count?: number
  mean_delta?: number
  median_delta?: number
  standard_deviation_delta?: number
  mean_delta_ci_95_lower?: number
  mean_delta_ci_95_upper?: number
  llm_win_rate?: number
  heuristic_win_rate?: number
  tie_rate?: number
  best_llm_delta?: number
  worst_llm_delta?: number
  [key: string]: unknown
}

export interface ExperimentListItem {
  directory: string
  manifest: Manifest
  experiment_summary: ExperimentSummary | null
}

export interface ExperimentListResponse {
  experiments: ExperimentListItem[]
}

export interface NetworkNode {
  node_id: string
  name: string
  node_type: string
  latitude: number | null
  longitude: number | null
  storage_capacity: number
  processing_capacity: number
  source_capacity: number
}

export interface NetworkEdge {
  edge_id: string
  origin_node_id: string
  destination_node_id: string
  mode: string
  distance_km: number
  base_lead_time_days: number
  daily_capacity: number
  unit_transport_cost: number
  reliability: number
  emergency: boolean
}

export interface NetworkConfigContent {
  nodes: NetworkNode[]
  edges: NetworkEdge[]
  [key: string]: unknown
}

export interface Shock {
  shock_id: string
  shock_type: string
  target_type: string
  target_id: string
  physical_start_day: number
  physical_end_day: number
  information_day: number
  [key: string]: unknown
}

export interface ScenarioConfigContent {
  scenario_id?: string
  description?: string
  shocks: Shock[]
  [key: string]: unknown
}

export interface ReplicationRow {
  replication: string
  seed: string
  heuristic_undisrupted_cost: string
  heuristic_disrupted_cost: string
  heuristic_tcd: string
  llm_undisrupted_cost: string
  llm_disrupted_cost: string
  llm_tcd: string
  delta: string
  winner: string
  [key: string]: string
}

export interface RunMetricsRow {
  replication: string
  policy: string
  run_kind: string
  total_cost: string
  [key: string]: string
}

export interface ExperimentDetail {
  directory: string
  manifest: Manifest
  summary: {
    experiment_summary?: ExperimentSummary
    cost_component_means?: Record<string, Record<string, number>>
    service_metric_means?: Record<string, Record<string, number>>
    [key: string]: unknown
  } | null
  replications: ReplicationRow[]
  run_metrics: RunMetricsRow[]
  network: NetworkConfigContent
  scenario: ScenarioConfigContent
}

export interface DailyMetricsRow {
  experiment_id: string
  scenario_id: string
  replication: number
  policy: string
  run_kind: string
  day: number
  inventory_units: number
  backlog_units: number
  shipments_at_node: number
  shipments_in_transit: number
  shipments_delivered: number
  daily_demand_units: number
  daily_same_day_fulfilled_units: number
  daily_backlog_fulfilled_units: number
  daily_transport_cost: number
  daily_reroute_cost: number
  daily_expedite_cost: number
  daily_holding_cost: number
  daily_backlog_cost: number
  daily_late_cost: number
  cumulative_total_cost: number
  active_shock_ids: string[]
}

export interface DecisionAction {
  shipment_id: string
  action_type: string
  route_id: string | null
  reason_code: string
  rationale: string
}

export interface ValidationResultPayload {
  code: string
  detail: string
  is_valid: boolean
}

export interface DecisionTraceEntry {
  decision_key: Record<string, unknown>
  day: number
  shipment_id: string
  observation: Record<string, unknown>
  proposed_action: DecisionAction | null
  proposal_validation: ValidationResultPayload | null
  fallback_invoked: boolean
  fallback_action: DecisionAction | null
  fallback_validation: ValidationResultPayload | null
  executed_action: DecisionAction | null
  decision_latency_ms: number
  [key: string]: unknown
}

export interface ReplaySlice {
  directory: string
  replication: number
  policy: string
  run_kind: string
  daily_metrics: DailyMetricsRow[]
  decisions: DecisionTraceEntry[]
}

export interface ConfigSchemaResponse {
  config_type: string
  json_schema: Record<string, unknown>
  note: string
}

export interface ConfigDefaultsResponse {
  config_type: string
  content: Record<string, unknown>
}

export interface RunLimits {
  max_sandbox_replications: number
  max_concurrent_runs: number
}

export type RunLifecycleStatus = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'

export interface RunStatus {
  run_id: string
  status: RunLifecycleStatus
  experiment_id: string | null
  total_replications: number
  completed_replications: number
  error: string | null
  created_at: string
}

export interface RunSubmitRequest {
  network: Record<string, unknown>
  scenario: Record<string, unknown>
  heuristic_policy: Record<string, unknown>
  llm_policy: Record<string, unknown>
  experiment: Record<string, unknown>
  api_key: string
}

export interface GridCell {
  topology: string
  severity: string
  directory: string | null
  manifest: Manifest | null
  experiment_summary: ExperimentSummary | null
}

export interface GridResponse {
  topologies: string[]
  severities: string[]
  cells: GridCell[]
}
