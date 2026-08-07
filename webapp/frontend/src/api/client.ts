// Thin typed fetch wrapper around the backend's /api/v1 routes.
//
// Every function here is a direct, uncached call to one backend endpoint —
// no client-side business logic lives in this file. Requests use relative
// paths (`/api/v1/...`) so the same build works against Vite's dev proxy
// (vite.config.ts) and a same-origin production deployment without an
// environment-specific base URL.

import type {
  ConfigDefaultsResponse,
  ConfigSchemaResponse,
  ExperimentDetail,
  ExperimentListResponse,
  ReplaySlice,
  RunLimits,
  RunStatus,
  RunSubmitRequest,
} from './types'

const API_BASE = '/api/v1'

async function unwrap<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.json().catch(() => null)
    const detail = body && typeof body === 'object' && 'detail' in body ? body.detail : response.statusText
    throw new Error(`${response.status} ${detail}`)
  }
  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}

function getJson<T>(path: string): Promise<T> {
  return fetch(`${API_BASE}${path}`).then(unwrap<T>)
}

function postJson<T>(path: string, body: unknown): Promise<T> {
  return fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then(unwrap<T>)
}

function deleteRequest(path: string): Promise<void> {
  return fetch(`${API_BASE}${path}`, { method: 'DELETE' }).then(unwrap<void>)
}

export function listExperiments(): Promise<ExperimentListResponse> {
  return getJson<ExperimentListResponse>('/gallery')
}

export function getExperimentDetail(directory: string): Promise<ExperimentDetail> {
  return getJson<ExperimentDetail>(`/gallery/${encodeURIComponent(directory)}`)
}

export function getReplaySlice(
  directory: string,
  replication: number,
  policy: string,
  runKind: string,
): Promise<ReplaySlice> {
  const params = new URLSearchParams({
    replication: String(replication),
    policy,
    run_kind: runKind,
  })
  return getJson<ReplaySlice>(`/gallery/${encodeURIComponent(directory)}/replay?${params}`)
}

export function getConfigSchema(configType: string): Promise<ConfigSchemaResponse> {
  return getJson<ConfigSchemaResponse>(`/configs/schema/${configType}`)
}

export function getConfigDefaults(configType: string): Promise<ConfigDefaultsResponse> {
  return getJson<ConfigDefaultsResponse>(`/configs/defaults/${configType}`)
}

export function getRunLimits(): Promise<RunLimits> {
  return getJson<RunLimits>('/runs/limits')
}

export function submitRun(request: RunSubmitRequest): Promise<RunStatus> {
  return postJson<RunStatus>('/runs', request)
}

export function getRunStatus(runId: string): Promise<RunStatus> {
  return getJson<RunStatus>(`/runs/${encodeURIComponent(runId)}`)
}

export function cancelRun(runId: string): Promise<void> {
  return deleteRequest(`/runs/${encodeURIComponent(runId)}`)
}

export function getRunDetail(runId: string): Promise<ExperimentDetail> {
  return getJson<ExperimentDetail>(`/runs/${encodeURIComponent(runId)}/detail`)
}

export function getRunReplay(
  runId: string,
  replication: number,
  policy: string,
  runKind: string,
): Promise<ReplaySlice> {
  const params = new URLSearchParams({
    replication: String(replication),
    policy,
    run_kind: runKind,
  })
  return getJson<ReplaySlice>(`/runs/${encodeURIComponent(runId)}/replay?${params}`)
}
