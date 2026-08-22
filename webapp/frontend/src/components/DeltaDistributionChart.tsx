// One dot per replication's delta (LLM TCD − heuristic TCD), on a single
// diverging strip along a shared axis centered on zero — the direct visual
// answer to "how much do replications actually vary," which the summary
// stat tiles (mean, CI) alone don't show. Diverging by the same sign
// convention as the rest of the app (--series-heuristic / --series-llm), a
// shaded band for the 95% CI on the mean, and a small vertical jitter per
// dot (deterministic, index-based — not real randomness) only to keep
// overlapping points visually distinguishable.

import type { ReplicationRow } from '../api/types'
import { formatCurrency } from '../lib/format'
import styles from './DeltaDistributionChart.module.css'

interface DeltaDistributionChartProps {
  replications: ReplicationRow[]
  meanDelta?: number
  ciLower?: number
  ciUpper?: number
}

const WIDTH = 640
const HEIGHT = 130
const PADDING = { top: 16, right: 24, bottom: 28, left: 24 }
const ROW_Y = 60
const JITTER_STEPS = [0, -8, 8, -14, 14]

export default function DeltaDistributionChart({
  replications,
  meanDelta,
  ciLower,
  ciUpper,
}: DeltaDistributionChartProps) {
  const points = replications
    .map((row) => ({
      replication: row.replication,
      winner: row.winner,
      delta: Number(row.delta),
    }))
    .filter((p) => !Number.isNaN(p.delta))

  if (points.length === 0) {
    return <p className={styles.empty}>No per-replication data available.</p>
  }

  const values = points.map((p) => p.delta)
  const rawMin = Math.min(...values, ciLower ?? Infinity, 0)
  const rawMax = Math.max(...values, ciUpper ?? -Infinity, 0)
  const span = rawMax - rawMin || 1
  const domainMin = rawMin - span * 0.08
  const domainMax = rawMax + span * 0.08

  const plotLeft = PADDING.left
  const plotRight = WIDTH - PADDING.right
  const plotWidth = plotRight - plotLeft

  function xFor(value: number): number {
    return plotLeft + ((value - domainMin) / (domainMax - domainMin)) * plotWidth
  }

  const zeroX = xFor(0)
  const meanX = meanDelta !== undefined ? xFor(meanDelta) : null
  const ciBand = ciLower !== undefined && ciUpper !== undefined ? [xFor(ciLower), xFor(ciUpper)] : null

  return (
    <div>
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className={styles.chart}
        role="img"
        aria-label="Per-replication delta distribution"
      >
        {ciBand && (
          <rect
            x={ciBand[0]}
            y={PADDING.top}
            width={Math.max(ciBand[1] - ciBand[0], 1)}
            height={ROW_Y + 24 - PADDING.top}
            className={styles.ciBand}
          />
        )}

        <line x1={zeroX} x2={zeroX} y1={PADDING.top} y2={ROW_Y + 24} className={styles.zeroLine} />
        <text x={zeroX} y={ROW_Y + 38} textAnchor="middle" className={styles.axisLabel}>
          0
        </text>

        {meanX !== null && (
          <line x1={meanX} x2={meanX} y1={ROW_Y - 16} y2={ROW_Y + 16} className={styles.meanTick} />
        )}

        {points.map((point, index) => {
          const cx = xFor(point.delta)
          const cy = ROW_Y + JITTER_STEPS[index % JITTER_STEPS.length]
          const seriesVar = point.delta < 0 ? 'var(--series-llm)' : point.delta > 0 ? 'var(--series-heuristic)' : 'var(--text-muted)'
          return (
            <circle key={point.replication} cx={cx} cy={cy} r={4} fill={seriesVar} className={styles.dot}>
              <title>
                Replication {point.replication}: delta {formatCurrency(point.delta)} ({point.winner.toLowerCase()} wins)
              </title>
            </circle>
          )
        })}

        <text x={plotLeft} y={HEIGHT - 8} textAnchor="start" className={styles.axisLabel}>
          {formatCurrency(domainMin)}
        </text>
        <text x={plotRight} y={HEIGHT - 8} textAnchor="end" className={styles.axisLabel}>
          {formatCurrency(domainMax)}
        </text>
      </svg>

      <div className={styles.legend}>
        <span className={styles.legendItem}>
          <span className={styles.legendDot} style={{ background: 'var(--series-heuristic)' }} />
          Heuristic cheaper
        </span>
        <span className={styles.legendItem}>
          <span className={styles.legendDot} style={{ background: 'var(--series-llm)' }} />
          LLM agent cheaper
        </span>
        {ciBand && (
          <span className={styles.legendItem}>
            <span className={styles.legendBand} />
            95% CI on mean delta
          </span>
        )}
      </div>
    </div>
  )
}
