// Lists every shipment-level decision made on the selected day for the
// selected (replication, policy, run_kind) branch: the executed action,
// its rationale, and whether a fallback was invoked or the proposal was
// invalid — using the app's fixed status colors (warning/critical), never
// color alone (each carries an icon-equivalent text label). When the
// decision came from the LLM agent, an "Agent reasoning" disclosure below
// it shows the actual tool-call trace (llm_interactions.jsonl) behind that
// decision — never fabricated or summarized, exactly the calls and results
// the agent saw.

import type { DecisionTraceEntry, LlmInteraction, ToolOutput } from '../api/types'
import styles from './DecisionExplorer.module.css'

interface DecisionExplorerProps {
  decisions: DecisionTraceEntry[]
  llmInteractions: LlmInteraction[]
  day: number | null
}

function findOutput(outputs: ToolOutput[], toolCallId: string): ToolOutput | undefined {
  return outputs.find((o) => o.tool_call_id === toolCallId)
}

function formatJson(value: unknown): string {
  if (value === null || value === undefined) return '—'
  if (typeof value === 'object' && Object.keys(value as object).length === 0) return '(no arguments)'
  return JSON.stringify(value, null, 2)
}

export default function DecisionExplorer({ decisions, llmInteractions, day }: DecisionExplorerProps) {
  const dayDecisions = day === null ? [] : decisions.filter((d) => d.day === day)

  if (dayDecisions.length === 0) {
    return <p className={styles.empty}>No shipment decisions were required on this day.</p>
  }

  function interactionFor(decision: DecisionTraceEntry): LlmInteraction | undefined {
    return llmInteractions.find(
      (i) => i.decision_key.day === decision.day && i.decision_key.shipment_id === decision.shipment_id,
    )
  }

  return (
    <div className={styles.list}>
      {dayDecisions.map((decision) => {
        const executed = decision.executed_action
        const proposalInvalid = decision.proposal_validation?.is_valid === false
        const interaction = interactionFor(decision)
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

            {interaction && (
              <details className={styles.reasoning}>
                <summary className={styles.reasoningSummary}>
                  Agent reasoning ({interaction.tool_calls.length} tool call
                  {interaction.tool_calls.length === 1 ? '' : 's'}, {interaction.model},{' '}
                  {interaction.latency_ms.toFixed(0)} ms
                  {interaction.attempt_count > 1 ? `, ${interaction.attempt_count} attempts` : ''})
                </summary>
                <ol className={styles.toolCallList}>
                  {interaction.tool_calls.map((call) => {
                    const output = findOutput(interaction.tool_outputs, call.tool_call_id)
                    return (
                      <li key={call.tool_call_id} className={styles.toolCall}>
                        <div className={styles.toolCallName}>{call.name}</div>
                        <pre className={styles.toolCallBlock}>{formatJson(call.arguments)}</pre>
                        {output && (
                          <>
                            <div className={styles.toolCallLabel}>result</div>
                            <pre className={styles.toolCallBlock}>{formatJson(output.output)}</pre>
                          </>
                        )}
                      </li>
                    )
                  })}
                </ol>
                <div className={styles.tokenUsage}>
                  tokens: {interaction.token_usage.input_tokens ?? '—'} in /{' '}
                  {interaction.token_usage.output_tokens ?? '—'} out
                </div>
              </details>
            )}
          </div>
        )
      })}
    </div>
  )
}
