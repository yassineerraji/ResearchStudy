// Controls for picking a (replication, policy, run_kind) branch and
// scrubbing through its days. Purely presentational/controlled — RunDetail
// owns the actual data fetch and current selection; this component only
// renders the selectors, the day slider, and an optional auto-play step
// timer, per the dataviz skill's guidance to keep filters in one row above
// the visualization they drive.

import { useEffect, useRef, useState } from 'react'
import { formatPolicyLabel } from '../lib/format'
import styles from './ReplayScrubber.module.css'

interface ReplayScrubberProps {
  replication: number
  replicationCount: number
  onReplicationChange: (value: number) => void
  policy: string
  onPolicyChange: (value: string) => void
  runKind: 'DISRUPTED' | 'UNDISRUPTED'
  onRunKindChange: (value: 'DISRUPTED' | 'UNDISRUPTED') => void
  days: number[]
  day: number | null
  onDayChange: (value: number) => void
}

export default function ReplayScrubber({
  replication,
  replicationCount,
  onReplicationChange,
  policy,
  onPolicyChange,
  runKind,
  onRunKindChange,
  days,
  day,
  onDayChange,
}: ReplayScrubberProps) {
  const [playing, setPlaying] = useState(false)
  const timerRef = useRef<number | null>(null)

  useEffect(() => {
    if (!playing || days.length === 0) return
    timerRef.current = window.setInterval(() => {
      const currentIndex = day === null ? -1 : days.indexOf(day)
      const nextIndex = currentIndex + 1
      if (nextIndex >= days.length) {
        setPlaying(false)
        return
      }
      onDayChange(days[nextIndex])
    }, 350)
    return () => {
      if (timerRef.current !== null) window.clearInterval(timerRef.current)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, day, days])

  const dayIndex = day === null ? 0 : Math.max(0, days.indexOf(day))

  return (
    <div className={styles.controls}>
      <div className={styles.row}>
        <label className={styles.field}>
          Replication
          <select
            value={replication}
            onChange={(event) => onReplicationChange(Number(event.target.value))}
          >
            {Array.from({ length: replicationCount }, (_, i) => i + 1).map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </label>

        <label className={styles.field}>
          Policy
          <select value={policy} onChange={(event) => onPolicyChange(event.target.value)}>
            <option value="heuristic">{formatPolicyLabel('heuristic')}</option>
            <option value="llm_agent">{formatPolicyLabel('llm_agent')}</option>
          </select>
        </label>

        <label className={styles.field}>
          World
          <select
            value={runKind}
            onChange={(event) => onRunKindChange(event.target.value as 'DISRUPTED' | 'UNDISRUPTED')}
          >
            <option value="DISRUPTED">Disrupted</option>
            <option value="UNDISRUPTED">Undisrupted</option>
          </select>
        </label>
      </div>

      <div className={styles.scrubRow}>
        <button
          type="button"
          className={styles.playButton}
          onClick={() => setPlaying((p) => !p)}
          disabled={days.length === 0}
          aria-label={playing ? 'Pause' : 'Play'}
        >
          {playing ? '❚❚' : '►'}
        </button>
        <input
          type="range"
          min={0}
          max={Math.max(0, days.length - 1)}
          value={dayIndex}
          disabled={days.length === 0}
          onChange={(event) => onDayChange(days[Number(event.target.value)])}
          className={styles.slider}
        />
        <span className={styles.dayLabel}>{day !== null ? `Day ${day}` : '—'}</span>
      </div>
    </div>
  )
}
