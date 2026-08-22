// Topology x severity grid heatmap: one cell per real completed experiment,
// diverging by the sign of mean delta (--series-heuristic when the
// heuristic is cheaper, --series-llm when the LLM agent is cheaper — the
// same fixed sign convention Gallery.tsx's winner badge already uses),
// magnitude encoded as fill opacity. A cell whose 95% CI on mean delta
// includes zero (not distinguishable from a tie at that replication count)
// gets a diagonal hatch overlay — a secondary, non-color channel, and the
// same fact is also stated in the cell's tooltip text, never color/texture
// alone. A missing cell (that combination hasn't been run yet) renders as a
// labeled blank, never omitted or treated as zero.

import type { GridCell } from '../api/types'
import { formatCurrency, formatPercent } from '../lib/format'
import styles from './GridHeatmap.module.css'

interface GridHeatmapProps {
  topologies: string[]
  severities: string[]
  cells: GridCell[]
  onSelectCell?: (cell: GridCell) => void
}

const CELL_SIZE = 96
const GAP = 6
const LABEL_COL_WIDTH = 90
const LABEL_ROW_HEIGHT = 28
const MIN_OPACITY = 0.18

function ciIncludesZero(summary: GridCell['experiment_summary'] | undefined): boolean {
  const lower = summary?.mean_delta_ci_95_lower
  const upper = summary?.mean_delta_ci_95_upper
  if (lower === undefined || upper === undefined) return false
  return lower <= 0 && upper >= 0
}

export default function GridHeatmap({ topologies, severities, cells, onSelectCell }: GridHeatmapProps) {
  const maxAbsDelta = cells.reduce((max, cell) => {
    const delta = cell.experiment_summary?.mean_delta
    return delta === undefined ? max : Math.max(max, Math.abs(delta))
  }, 0)

  const width = LABEL_COL_WIDTH + severities.length * (CELL_SIZE + GAP) - GAP
  const height = LABEL_ROW_HEIGHT + topologies.length * (CELL_SIZE + GAP) - GAP

  function cellAt(topology: string, severity: string): GridCell | undefined {
    return cells.find((c) => c.topology === topology && c.severity === severity)
  }

  return (
    <div className={styles.wrapper}>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className={styles.chart}
        role="img"
        aria-label="Mean cost delta by network topology and disruption severity"
      >
        <defs>
          <pattern id="tie-hatch" width="6" height="6" patternTransform="rotate(45)" patternUnits="userSpaceOnUse">
            <rect width="6" height="6" fill="transparent" />
            <line x1="0" y1="0" x2="0" y2="6" stroke="var(--text-muted)" strokeWidth="1.5" />
          </pattern>
        </defs>

        {severities.map((severity, col) => (
          <text
            key={severity}
            x={LABEL_COL_WIDTH + col * (CELL_SIZE + GAP) + CELL_SIZE / 2}
            y={LABEL_ROW_HEIGHT - 10}
            textAnchor="middle"
            className={styles.axisLabel}
          >
            {severity}
          </text>
        ))}

        {topologies.map((topology, row) => {
          const y = LABEL_ROW_HEIGHT + row * (CELL_SIZE + GAP)
          return (
            <g key={topology}>
              <text x={LABEL_COL_WIDTH - 12} y={y + CELL_SIZE / 2} textAnchor="end" dy="0.32em" className={styles.axisLabel}>
                {topology}
              </text>
              {severities.map((severity, col) => {
                const x = LABEL_COL_WIDTH + col * (CELL_SIZE + GAP)
                const cell = cellAt(topology, severity)
                const summary = cell?.experiment_summary
                const delta = summary?.mean_delta
                const clickable = Boolean(cell?.directory && onSelectCell)

                if (!cell || !cell.directory || delta === undefined) {
                  return (
                    <g key={severity}>
                      <rect x={x} y={y} width={CELL_SIZE} height={CELL_SIZE} rx={6} className={styles.blankCell} />
                      <text x={x + CELL_SIZE / 2} y={y + CELL_SIZE / 2} textAnchor="middle" dy="0.32em" className={styles.blankLabel}>
                        not run
                      </text>
                    </g>
                  )
                }

                const seriesVar = delta > 0 ? 'var(--series-heuristic)' : 'var(--series-llm)'
                const magnitude = maxAbsDelta > 0 ? Math.abs(delta) / maxAbsDelta : 0
                const opacity = MIN_OPACITY + magnitude * (1 - MIN_OPACITY)
                const tie = ciIncludesZero(summary)
                const winner = delta < -0.01 ? 'LLM agent cheaper' : delta > 0.01 ? 'Heuristic cheaper' : 'tie'

                return (
                  <g
                    key={severity}
                    className={clickable ? styles.clickable : undefined}
                    onClick={clickable ? () => onSelectCell?.(cell) : undefined}
                  >
                    <rect x={x} y={y} width={CELL_SIZE} height={CELL_SIZE} rx={6} fill={seriesVar} opacity={opacity} />
                    {tie && (
                      <rect x={x} y={y} width={CELL_SIZE} height={CELL_SIZE} rx={6} fill="url(#tie-hatch)" />
                    )}
                    <rect x={x} y={y} width={CELL_SIZE} height={CELL_SIZE} rx={6} className={styles.cellBorder} />
                    <text x={x + CELL_SIZE / 2} y={y + CELL_SIZE / 2 - 6} textAnchor="middle" className={styles.cellValue}>
                      {formatCurrency(delta)}
                    </text>
                    <text x={x + CELL_SIZE / 2} y={y + CELL_SIZE / 2 + 14} textAnchor="middle" className={styles.cellSub}>
                      {formatPercent(summary?.llm_win_rate)} LLM wins
                    </text>
                    <title>
                      {topology} x {severity}: mean delta {formatCurrency(delta)} ({winner}
                      {tie ? ', 95% CI includes zero — not distinguishable from a tie' : ''}). LLM win rate{' '}
                      {formatPercent(summary?.llm_win_rate)}, heuristic win rate{' '}
                      {formatPercent(summary?.heuristic_win_rate)}.
                    </title>
                  </g>
                )
              })}
            </g>
          )
        })}
      </svg>

      <div className={styles.legend}>
        <span className={styles.legendItem}>
          <span className={styles.legendSwatch} style={{ background: 'var(--series-heuristic)' }} />
          Heuristic cheaper
        </span>
        <span className={styles.legendItem}>
          <span className={styles.legendSwatch} style={{ background: 'var(--series-llm)' }} />
          LLM agent cheaper
        </span>
        <span className={styles.legendItem}>
          <span className={styles.legendSwatchHatch} />
          95% CI includes zero (not distinguishable from a tie)
        </span>
        <span className={styles.legendItem}>
          <span className={styles.legendSwatchBlank} />
          Not yet run
        </span>
      </div>
    </div>
  )
}
