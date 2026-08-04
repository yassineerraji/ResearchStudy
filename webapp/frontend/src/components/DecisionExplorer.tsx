// Lists every shipment-level decision made on the selected day for the
// selected (replication, policy, run_kind) branch: the executed action,
// its rationale, and whether a fallback was invoked or the proposal was
// invalid — using the app's fixed status colors (warning/critical), never
// color alone (each carries an icon-equivalent text label).

import type { DecisionTraceEntry } from '../api/types'
import styles from './DecisionExplorer.module.css'

interface DecisionExplorerProps {
  decisions: DecisionTraceEntry[]
  day: number | null
}

export default function DecisionExplorer({ decisions, day }: DecisionExplorerProps) {
  const dayDecisions = day === null ? [] : decisions.filter((d) => d.day === day)

  if (dayDecisions.length === 0) {
    return <p className={styles.empty}>No shipment decisions were required on this day.</p>
  }

  return (
    <div className={styles.list}>
      {dayDecisions.map((decision) => {
        const executed = decision.executed_action
        const proposalInvalid = decision.proposal_validation?.is_valid === false
        return (
          <div key={`${decision.shipment_id}:${decision.day}`} className={styles.card}>
            <div className={styles.header}>
              <span className={styles.shipmentId}>{decision.shipment_id}</span>
              <span className={styles.actionType}>{executed?.action_type ?? '—'}</span>
            </div>
            {(decision.fallback_invoked || proposalInvalid) && (
              <div className={styles.badges}>
                {proposalInvalid && (
                  <span className={styles.badgeCritical}>⚠ invalid proposal</span>
                )}
                {decision.fallback_invoked && (
                  <span className={styles.badgeWarning}>↺ fallback invoked</span>
                )}
              </div>
            )}
            {executed?.rationale && <p className={styles.rationale}>&ldquo;{executed.rationale}&rdquo;</p>}
            <div className={styles.meta}>
              reason: {executed?.reason_code ?? '—'}
              {executed?.route_id ? ` · route: ${executed.route_id}` : ''}
              {' · '}
              {decision.decision_latency_ms.toFixed(1)} ms
            </div>
          </div>
        )
      })}
    </div>
  )
}
