// Lists every completed experiment the backend can see under outputs/.
// Read-only: no run is triggered from this page. Each card links to
// RunDetail for the full breakdown.

import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { listExperiments } from '../api/client'
import type { ExperimentListItem } from '../api/types'
import { formatCurrency, formatPercent } from '../lib/format'
import styles from './Gallery.module.css'

function winnerLabel(meanDelta: number | undefined): { text: string; color: string } {
  if (meanDelta === undefined) return { text: 'no data', color: 'var(--text-muted)' }
  if (meanDelta < -0.01) return { text: 'LLM agent lower disruption cost', color: 'var(--series-llm)' }
  if (meanDelta > 0.01) return { text: 'Heuristic lower disruption cost', color: 'var(--series-heuristic)' }
  return { text: 'tie', color: 'var(--text-muted)' }
}

export default function Gallery() {
  const [experiments, setExperiments] = useState<ExperimentListItem[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    listExperiments()
      .then((res) => setExperiments(res.experiments))
      .catch((err: Error) => setError(err.message))
  }, [])

  if (error) return <p className={styles.error}>Could not load experiments: {error}</p>
  if (!experiments) return <p className={styles.loading}>Loading experiments…</p>
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
