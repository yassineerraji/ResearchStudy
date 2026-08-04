// One experiment's full detail: summary stats, cost breakdown, the network
// diagram, and a day-by-day replay scrubber with the decisions made on the
// selected day. All data comes from a small number of backend calls — the
// heavy daily_metrics/decision_traces files are never fetched in full (see
// webapp/backend's byte-offset gallery index); only one branch's slice at a
// time.

import { useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { getExperimentDetail, getReplaySlice } from '../api/client'
import type { DailyMetricsRow, ExperimentDetail, ReplaySlice } from '../api/types'
import StatTile from '../components/StatTile'
import NetworkDiagram from '../components/NetworkDiagram'
import CostBreakdownChart from '../components/CostBreakdownChart'
import ReplayScrubber from '../components/ReplayScrubber'
import DecisionExplorer from '../components/DecisionExplorer'
import { formatCurrency, formatPercent } from '../lib/format'
import styles from './RunDetail.module.css'

export default function RunDetail() {
  const { directory } = useParams<{ directory: string }>()

  const [detail, setDetail] = useState<ExperimentDetail | null>(null)
  const [detailError, setDetailError] = useState<string | null>(null)
  const [costRunKind, setCostRunKind] = useState<'DISRUPTED' | 'UNDISRUPTED'>('DISRUPTED')

  const [replication, setReplication] = useState(1)
  const [policy, setPolicy] = useState('heuristic')
  const [replayRunKind, setReplayRunKind] = useState<'DISRUPTED' | 'UNDISRUPTED'>('DISRUPTED')
  const [replaySlice, setReplaySlice] = useState<ReplaySlice | null>(null)
  const [replayError, setReplayError] = useState<string | null>(null)
  const [day, setDay] = useState<number | null>(null)

  useEffect(() => {
    if (!directory) return
    getExperimentDetail(directory)
      .then(setDetail)
      .catch((err: Error) => setDetailError(err.message))
  }, [directory])

  useEffect(() => {
    if (!directory) return
    setReplayError(null)
    getReplaySlice(directory, replication, policy, replayRunKind)
      .then((slice) => {
        setReplaySlice(slice)
        const days = slice.daily_metrics.map((row) => row.day)
        const shockStartDay = days.find((d) =>
          slice.daily_metrics.find((row) => row.day === d)?.active_shock_ids.length,
        )
        setDay(shockStartDay ?? days[0] ?? null)
      })
      .catch((err: Error) => setReplayError(err.message))
  }, [directory, replication, policy, replayRunKind])

  const days = useMemo(() => replaySlice?.daily_metrics.map((row) => row.day) ?? [], [replaySlice])
  const dayRow: DailyMetricsRow | undefined = useMemo(
    () => replaySlice?.daily_metrics.find((row) => row.day === day),
    [replaySlice, day],
  )

  if (detailError) return <p>Could not load experiment: {detailError}</p>
  if (!detail) return <p>Loading…</p>

  const summary = detail.summary?.experiment_summary
  const costComponentMeans = detail.summary?.cost_component_means ?? {}

  return (
    <article>
      <Link to="/gallery" className={styles.backLink}>
        ← Results Gallery
      </Link>

      <header className={styles.header}>
        <h1>{detail.manifest.experiment_id}</h1>
        <div className={styles.meta}>
          {detail.manifest.replications} replications · model {detail.manifest.llm_model ?? 'unknown'} (
          {detail.manifest.llm_execution_mode}) · seed {detail.manifest.base_seed} · created{' '}
          {detail.manifest.created_at_utc}
        </div>
      </header>

      <div className={styles.statRow}>
        <StatTile label="Mean Δ (LLM − heuristic)" value={formatCurrency(summary?.mean_delta)} />
        <StatTile label="LLM win rate" value={formatPercent(summary?.llm_win_rate)} />
        <StatTile label="Heuristic win rate" value={formatPercent(summary?.heuristic_win_rate)} />
        <StatTile label="Tie rate" value={formatPercent(summary?.tie_rate)} />
      </div>

      <section className={styles.section}>
        <h2>Cost breakdown</h2>
        <CostBreakdownChart
          costComponentMeans={costComponentMeans}
          runKind={costRunKind}
          onRunKindChange={setCostRunKind}
        />
      </section>

      <section className={styles.section}>
        <h2>Network and disruption replay</h2>
        <ReplayScrubber
          replication={replication}
          replicationCount={detail.manifest.replications}
          onReplicationChange={setReplication}
          policy={policy}
          onPolicyChange={setPolicy}
          runKind={replayRunKind}
          onRunKindChange={setReplayRunKind}
          days={days}
          day={day}
          onDayChange={setDay}
        />

        {replayError && <p>Could not load replay data: {replayError}</p>}

        {dayRow && (
          <div className={styles.dayStats}>
            <StatTile label="Inventory" value={String(dayRow.inventory_units)} />
            <StatTile label="Backlog" value={String(dayRow.backlog_units)} />
            <StatTile label="In transit" value={String(dayRow.shipments_in_transit)} />
            <StatTile label="Cumulative cost" value={formatCurrency(dayRow.cumulative_total_cost)} />
          </div>
        )}

        <div className={styles.replayGrid}>
          <NetworkDiagram
            network={detail.network}
            shocks={detail.scenario.shocks}
            activeShockIds={dayRow?.active_shock_ids ?? []}
          />
          <div>
            <DecisionExplorer decisions={replaySlice?.decisions ?? []} day={day} />
          </div>
        </div>
      </section>
    </article>
  )
}
