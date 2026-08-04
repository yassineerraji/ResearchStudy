// Small, shared number/label formatters so every page renders costs,
// percentages, and policy names the same way instead of each component
// inventing its own `toFixed` call.

export function formatCurrency(value: number | string | undefined | null): string {
  if (value === undefined || value === null || value === '') return '—'
  const n = typeof value === 'string' ? Number(value) : value
  if (Number.isNaN(n)) return '—'
  return n.toLocaleString(undefined, { maximumFractionDigits: 0 })
}

export function formatPercent(value: number | undefined | null): string {
  if (value === undefined || value === null || Number.isNaN(value)) return '—'
  return `${(value * 100).toFixed(1)}%`
}

export function formatPolicyLabel(policy: string): string {
  return policy === 'llm_agent' ? 'LLM agent' : policy === 'heuristic' ? 'Heuristic' : policy
}

export function policySeriesVar(policy: string): string {
  return policy === 'llm_agent' ? 'var(--series-llm)' : 'var(--series-heuristic)'
}
