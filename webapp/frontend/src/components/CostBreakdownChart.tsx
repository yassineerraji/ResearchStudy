// Grouped bar chart of mean cost-component values, heuristic vs. LLM agent,
// for one run_kind (a segmented filter above the chart, not a second
// series — see dataviz skill's interaction.md: filters that change what's
// shown belong beside the chart, not encoded as another color).
//
// One axis (currency), color assigned by policy identity in the app's fixed
// categorical order (heuristic=slot 1 blue, llm_agent=slot 2 orange), a
// legend because there are two series, and a native tooltip per bar.

import { useMemo, useState } from 'react'
import { formatCurrency, formatPolicyLabel } from '../lib/format'
import styles from './CostBreakdownChart.module.css'

const COMPONENTS = [
  'transport_cost',
  'reroute_cost',
  'expedite_cost',
  'holding_cost',
  'backlog_cost',
  'late_cost',
  'terminal_cost',
] as const

const COMPONENT_LABELS: Record<string, string> = {
  transport_cost: 'Transport',
  reroute_cost: 'Reroute',
  expedite_cost: 'Expedite',
  holding_cost: 'Holding',
  backlog_cost: 'Backlog',
  late_cost: 'Late',
  terminal_cost: 'Terminal',
}

const POLICIES = ['heuristic', 'llm_agent'] as const

interface CostBreakdownChartProps {
  costComponentMeans: Record<string, Record<string, number>>
  runKind: 'DISRUPTED' | 'UNDISRUPTED'
  onRunKindChange: (runKind: 'DISRUPTED' | 'UNDISRUPTED') => void
}

const WIDTH = 640
const HEIGHT = 260
const PADDING = { top: 16, right: 16, bottom: 46, left: 56 }

export default function CostBreakdownChart({
  costComponentMeans,
  runKind,
  onRunKindChange,
}: CostBreakdownChartProps) {
  const [hover, setHover] = useState<string | null>(null)

  const series = useMemo(
    () =>
      POLICIES.map((policy) => ({
        policy,
        values: costComponentMeans[`${policy}:${runKind}`] ?? {},
      })),
    [costComponentMeans, runKind],
  )

  const maxValue = useMemo(() => {
    let max = 0
    for (const { values } of series) {
      for (const component of COMPONENTS) {
        max = Math.max(max, values[component] ?? 0)
      }
    }
    return max || 1
  }, [series])

  const plotWidth = WIDTH - PADDING.left - PADDING.right
  const plotHeight = HEIGHT - PADDING.top - PADDING.bottom
  const groupWidth = plotWidth / COMPONENTS.length
  const barWidth = (groupWidth - 16) / 2

  return (
    <div>
      <div className={styles.controls}>
        <div className={styles.segmented} role="tablist" aria-label="Disruption state">
          {(['DISRUPTED', 'UNDISRUPTED'] as const).map((option) => (
            <button
              key={option}
              type="button"
              role="tab"
              aria-selected={runKind === option}
              className={runKind === option ? styles.segmentActive : styles.segment}
              onClick={() => onRunKindChange(option)}
            >
              {option === 'DISRUPTED' ? 'Disrupted' : 'Undisrupted'}
            </button>
          ))}
        </div>
        <div className={styles.legend}>
          {POLICIES.map((policy) => (
            <span key={policy} className={styles.legendItem}>
              <span className={styles.legendDot} style={{ background: `var(--series-${policy === 'llm_agent' ? 'llm' : 'heuristic'})` }} />
              {formatPolicyLabel(policy)}
            </span>
          ))}
        </div>
      </div>

      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className={styles.chart} role="img" aria-label="Cost breakdown by component">
        {[0, 0.25, 0.5, 0.75, 1].map((fraction) => {
          const y = PADDING.top + plotHeight * (1 - fraction)
          return (
            <g key={fraction}>
              <line x1={PADDING.left} x2={WIDTH - PADDING.right} y1={y} y2={y} stroke="var(--gridline)" strokeWidth={1} />
              <text x={PADDING.left - 8} y={y} textAnchor="end" dy="0.32em" className={styles.axisLabel}>
                {formatCurrency(maxValue * fraction)}
              </text>
            </g>
          )
        })}

        {COMPONENTS.map((component, index) => {
          const groupX = PADDING.left + index * groupWidth
          return (
            <g key={component}>
              <text
                x={groupX + groupWidth / 2}
                y={HEIGHT - PADDING.bottom + 18}
                textAnchor="middle"
                className={styles.axisLabel}
              >
                {COMPONENT_LABELS[component]}
              </text>
              {series.map((s, seriesIndex) => {
                const value = s.values[component] ?? 0
                const barHeight = (value / maxValue) * plotHeight
                const x = groupX + 4 + seriesIndex * (barWidth + 8)
                const y = PADDING.top + plotHeight - barHeight
                const key = `${component}:${s.policy}`
                return (
                  <rect
                    key={key}
                    x={x}
                    y={y}
                    width={barWidth}
                    height={Math.max(barHeight, 0)}
                    rx={3}
                    fill={`var(--series-${s.policy === 'llm_agent' ? 'llm' : 'heuristic'})`}
                    opacity={hover === null || hover === key ? 1 : 0.45}
                    onMouseEnter={() => setHover(key)}
                    onMouseLeave={() => setHover(null)}
                  >
                    <title>
                      {formatPolicyLabel(s.policy)} · {COMPONENT_LABELS[component]}: {formatCurrency(value)}
                    </title>
                  </rect>
                )
              })}
            </g>
          )
        })}
      </svg>
    </div>
  )
}
