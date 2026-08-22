// Lists every completed experiment the backend can see under outputs/.
// Read-only: no run is triggered from this page. Each card links to
// RunDetail for the full breakdown.

import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { getGrid, listExperiments } from '../api/client'
import type { ExperimentListItem, GridResponse } from '../api/types'
import GridHeatmap from '../components/GridHeatmap'
import { formatCurrency, formatPercent } from '../lib/format'
import styles from './Gallery.module.css'

function winnerLabel(meanDelta: number | undefined): { text: string; color: string } {
  if (meanDelta === undefined) return { text: 'no data', color: 'var(--text-muted)' }
  if (meanDelta < -0.01) return { text: 'LLM agent lower disruption cost', color: 'var(--series-llm)' }
  if (meanDelta > 0.01) return { text: 'Heuristic lower disruption cost', color: 'var(--series-heuristic)' }
  return { text: 'tie', color: 'var(--text-muted)' }
}

type ViewMode = 'list' | 'grid'

export default function Gallery() {
  const navigate = useNavigate()
  const [view, setView] = useState<ViewMode>('list')
  const [experiments, setExperiments] = useState<ExperimentListItem[] | null>(null)
  const [grid, setGrid] = useState<GridResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    listExperiments()
      .then((res) => setExperiments(res.experiments))
      .catch((err: Error) => setError(err.message))
  }, [])

  useEffect(() => {
    if (view === 'grid' && !grid) {
      getGrid()
        .then(setGrid)
        .catch((err: Error) => setError(err.message))
    }
  }, [view, grid])

  if (error) return <p className={styles.error}>Could not load experiments: {error}</p>
  if (!experiments) return <p className={styles.loading}>Loading experiments…</p>

  const viewToggle = (
    <div className={styles.viewToggle} role="tablist" aria-label="Gallery view">
      <button
        type="button"
        role="tab"
        aria-selected={view === 'list'}
        className={view === 'list' ? styles.viewButtonActive : styles.viewButton}
        onClick={() => setView('list')}
      >
        List
      </button>
      <button
        type="button"
        role="tab"
        aria-selected={view === 'grid'}
        className={view === 'grid' ? styles.viewButtonActive : styles.viewButton}
        onClick={() => setView('grid')}
      >
        Topology x severity grid
      </button>
    </div>
  )

  if (view === 'grid') {
    return (
      <div>
        <h1>Results Gallery</h1>
        <p>The V2 topology x severity grid, one cell per real completed experiment.</p>
        {viewToggle}
        {!grid ? (
          <p className={styles.loading}>Loading grid…</p>
        ) : (
          <GridHeatmap
            topologies={grid.topologies}
            severities={grid.severities}
            cells={grid.cells}
            onSelectCell={(cell) => cell.directory && navigate(`/gallery/${cell.directory}`)}
          />
        )}
      </div>
    )
  }

  if (experiments.length === 0) {
    return (
      <p className={styles.empty}>
        No completed experiments found under <code>outputs/</code> yet.
      </p>
    )
  }

  return (
    <div>
      <h1>Results Gallery</h1>
      <p>Completed heuristic-vs-LLM disruption-response experiments, newest first.</p>
      {viewToggle}
      <div className={styles.grid}>
        {experiments.map((item) => {
          const winner = winnerLabel(item.experiment_summary?.mean_delta)
          return (
            <Link key={item.directory} to={`/gallery/${item.directory}`} className={styles.card}>
              <div className={styles.cardTitle}>{item.manifest.experiment_id}</div>
              <div className={styles.cardMeta}>
                {item.manifest.replications} replications · {item.manifest.llm_model ?? 'unknown model'} ·{' '}
                {item.manifest.llm_execution_mode}
              </div>
              <div className={styles.cardStats}>
                <span>
                  mean Δ <strong>{formatCurrency(item.experiment_summary?.mean_delta)}</strong>
                </span>
                <span>
                  LLM win rate <strong>{formatPercent(item.experiment_summary?.llm_win_rate)}</strong>
                </span>
              </div>
              <div className={styles.winnerBadge} style={{ color: winner.color }}>
                <span className={styles.dot} style={{ background: winner.color }} />
                {winner.text}
              </div>
            </Link>
          )
        })}
      </div>
    </div>
  )
}
