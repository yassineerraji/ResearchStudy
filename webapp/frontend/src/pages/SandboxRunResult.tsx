// Route wrapper for a visitor's own submitted sandbox run: polls
// GET /api/v1/runs/{id} every second while queued/running, shows progress
// and a cancel button, then — once the run reaches a terminal status —
// either an error (failed/cancelled) or the exact same `ExperimentDetailView`
// the Results Gallery uses (completed), backed by `/api/v1/runs/{id}/detail`
// and `.../replay` instead of the gallery's endpoints.

import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { cancelRun, getRunDetail, getRunReplay, getRunStatus } from '../api/client'
import type { RunStatus } from '../api/types'
import ExperimentDetailView from '../components/ExperimentDetailView'
import styles from './SandboxRunResult.module.css'

const POLL_INTERVAL_MS = 1000

function statusDotClass(status: RunStatus['status']): string {
  if (status === 'failed') return styles.dotFailed
  if (status === 'cancelled') return styles.dotCancelled
  return styles.dotRunning
}

export default function SandboxRunResult() {
  const { runId } = useParams<{ runId: string }>()
  const [status, setStatus] = useState<RunStatus | null>(null)
  const [fetchError, setFetchError] = useState<string | null>(null)

  useEffect(() => {
    if (!runId) return
    let cancelled = false

    async function poll() {
      try {
        const next = await getRunStatus(runId!)
        if (cancelled) return
        setStatus(next)
        if (next.status === 'queued' || next.status === 'running') {
          setTimeout(poll, POLL_INTERVAL_MS)
        }
      } catch (err) {
        if (!cancelled) setFetchError((err as Error).message)
      }
    }
    poll()

    return () => {
      cancelled = true
    }
  }, [runId])

  if (!runId) return null
  if (fetchError) return <p>Could not load run: {fetchError}</p>
  if (!status) return <p>Loading…</p>

  if (status.status === 'completed') {
    return (
      <ExperimentDetailView
        key={runId}
        fetchDetail={() => getRunDetail(runId)}
        fetchReplay={(replication, policy, runKind) => getRunReplay(runId, replication, policy, runKind)}
        backTo="/gallery"
        backLabel="← Results Gallery"
      />
    )
  }

  const progressPercent =
    status.total_replications > 0
      ? Math.round((status.completed_replications / status.total_replications) * 100)
      : 0

  return (
    <div className={styles.card}>
      <div className={styles.statusLine}>
        <span className={statusDotClass(status.status)} />
        {status.status === 'queued' && 'Queued — waiting for a free slot'}
        {status.status === 'running' && 'Running'}
        {status.status === 'failed' && 'Failed'}
        {status.status === 'cancelled' && 'Cancelled'}
      </div>

      {(status.status === 'queued' || status.status === 'running') && (
        <>
          <div className={styles.progressTrack}>
            <div className={styles.progressFill} style={{ width: `${progressPercent}%` }} />
          </div>
          <div className={styles.progressLabel}>
            Replication {status.completed_replications} / {status.total_replications || '?'}
          </div>
          <div className={styles.actions}>
            <button
              type="button"
              className={styles.button}
              onClick={() => cancelRun(runId).catch((err: Error) => setFetchError(err.message))}
            >
              Cancel
            </button>
          </div>
        </>
      )}

      {status.status === 'failed' && (
        <>
          <div className={styles.errorBox}>{status.error ?? 'Unknown error.'}</div>
          <div className={styles.actions}>
            <Link to="/runs/new" className={styles.button}>
              Try again
            </Link>
          </div>
        </>
      )}

      {status.status === 'cancelled' && (
        <div className={styles.actions}>
          <Link to="/runs/new" className={styles.button}>
            Start another run
          </Link>
        </div>
      )}
    </div>
  )
}
