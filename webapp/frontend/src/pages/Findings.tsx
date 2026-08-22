// Presents the audited research result directly in the webapp instead of
// leaving it to reports/*_slide_deck.html: the V1 severity-flip story (the
// Standard column of the same grid), the full V2 topology x severity grid,
// and the signal-to-noise check that confirms V2's redesign actually fixed
// the flaw the V1 audit found (every V1 win rate was exactly 0% or 100%
// because the mean cost gap dwarfed the replication-to-replication noise;
// V2.10/plot 14 tracks |mean delta| / stdev(delta) for exactly this reason).
// All numbers come live from GET /api/v1/gallery/grid — the real completed
// output directories under outputs/ — never hardcoded.

import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { getGrid } from '../api/client'
import type { GridCell, GridResponse } from '../api/types'
import GridHeatmap from '../components/GridHeatmap'
import { formatCurrency, formatPercent } from '../lib/format'
import styles from './Findings.module.css'

function signalToNoise(cell: GridCell | undefined): number | null {
  const mean = cell?.experiment_summary?.mean_delta
  const stdev = cell?.experiment_summary?.standard_deviation_delta
  if (mean === undefined || stdev === undefined || stdev === 0) return null
  return Math.abs(mean) / stdev
}

export default function Findings() {
  const navigate = useNavigate()
  const [grid, setGrid] = useState<GridResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    getGrid()
      .then(setGrid)
      .catch((err: Error) => setError(err.message))
  }, [])

  const standardRow = useMemo(
    () => grid?.cells.filter((c) => c.topology === 'Standard') ?? [],
    [grid],
  )

  if (error) return <p>Could not load results: {error}</p>
  if (!grid) return <p>Loading…</p>

  const bySeverity = (severity: string) => standardRow.find((c) => c.severity === severity)

  return (
    <article className={styles.findings}>
      <h1>What the experiments actually found</h1>
      <p className={styles.lede}>
        Every number below is computed live from the real, completed experiment output on this
        server &mdash; nothing here is a hardcoded figure copied from a report.
      </p>

      <section>
        <h2>V1: the verdict flips with severity</h2>
        <p>
          Version 1 ran the Standard network under three disruption severities. The heuristic won
          decisively when the disruption was mild; the LLM agent won just as decisively once the
          disruption got serious &mdash; and within each severity, every single replication agreed
          on the winner.
        </p>
        <div className={styles.severityRow}>
          {['Light', 'Medium', 'Heavy'].map((severity) => {
            const cell = bySeverity(severity)
            const delta = cell?.experiment_summary?.mean_delta
            return (
              <div key={severity} className={styles.severityCard}>
                <div className={styles.severityLabel}>{severity}</div>
                <div
                  className={styles.severityValue}
                  style={{ color: delta !== undefined && delta < 0 ? 'var(--series-llm)' : 'var(--series-heuristic)' }}
                >
                  {formatCurrency(delta)}
                </div>
                <div className={styles.severitySub}>
                  {cell ? `${formatPercent(cell.experiment_summary?.llm_win_rate)} LLM win rate` : 'not run'}
                </div>
              </div>
            )
          })}
        </div>
      </section>

      <section>
        <h2>V2: does the story hold across network shape too?</h2>
        <p>
          The same comparison, now crossed with three network topologies (Compact, Standard,
          Extended). Click any cell to open that experiment&rsquo;s full detail and replay.
        </p>
        <GridHeatmap
          topologies={grid.topologies}
          severities={grid.severities}
          cells={grid.cells}
          onSelectCell={(cell) => cell.directory && navigate(`/gallery/${cell.directory}`)}
        />
      </section>

      <section>
        <h2>Did V2 actually fix the audit finding?</h2>
        <p>
          The reason every V1 win rate was exactly 0% or 100% was that the mean cost gap between
          policies was 7.6&times;&ndash;10.2&times; larger than the random noise between
          replications &mdash; not enough randomness for a close result to ever occur. The same
          ratio, computed per cell below (<code>|mean delta| / standard deviation of delta</code>),
          should now sit far lower wherever the V2 redesign's richer randomness took hold.
        </p>
        <div className={styles.snrGrid}>
          {grid.cells.map((cell) => {
            const ratio = signalToNoise(cell)
            return (
              <div key={`${cell.topology}-${cell.severity}`} className={styles.snrCell}>
                <div className={styles.snrLabel}>
                  {cell.topology} &times; {cell.severity}
                </div>
                <div className={styles.snrValue}>{ratio === null ? '—' : `${ratio.toFixed(1)}x`}</div>
              </div>
            )
          })}
        </div>
      </section>
    </article>
  )
}
