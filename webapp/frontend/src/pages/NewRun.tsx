// Lets a visitor launch their own small live experiment against a curated
// subset of parameters — disruption timing, replication count/seed,
// heuristic aggressiveness, and LLM agent behavior — rather than a full
// generic config editor. Network topology and the harder-to-reason-about
// experiment fields (warmup/horizon/drain days) stay fixed at the baseline
// defaults for this first pass; see the M4 design notes on why.
//
// Submission goes straight to POST /api/v1/runs, which re-validates the
// whole bundle server-side through the same `resolve_config` path
// `/configs/validate` uses (see webapp/backend/app/services/run_launcher.py)
// — this form does not duplicate that validation, only basic input bounds.

import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { getConfigDefaults, getRunLimits, submitRun } from '../api/client'
import type { RunLimits } from '../api/types'
import styles from './NewRun.module.css'

type Draft = Record<string, unknown>

export default function NewRun() {
  const navigate = useNavigate()

  const [network, setNetwork] = useState<Draft | null>(null)
  const [scenario, setScenario] = useState<Draft | null>(null)
  const [heuristicPolicy, setHeuristicPolicy] = useState<Draft | null>(null)
  const [llmPolicy, setLlmPolicy] = useState<Draft | null>(null)
  const [experiment, setExperiment] = useState<Draft | null>(null)
  const [limits, setLimits] = useState<RunLimits | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)

  const [apiKey, setApiKey] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState<string | null>(null)

  useEffect(() => {
    Promise.all([
      getConfigDefaults('network'),
      getConfigDefaults('scenario'),
      getConfigDefaults('heuristic_policy'),
      getConfigDefaults('llm_policy'),
      getConfigDefaults('experiment'),
      getRunLimits(),
    ])
      .then(([n, s, h, l, e, lim]) => {
        setNetwork(n.content)
        setScenario(s.content)
        setHeuristicPolicy(h.content)
        setLlmPolicy(l.content)
        setExperiment({
          ...e.content,
          replications: Math.min(Number(e.content.replications ?? 1), lim.max_sandbox_replications),
        })
        setLimits(lim)
      })
      .catch((err: Error) => setLoadError(err.message))
  }, [])

  if (loadError) return <p className={styles.loadError}>Could not load defaults: {loadError}</p>
  if (!network || !scenario || !heuristicPolicy || !llmPolicy || !experiment || !limits) {
    return <p className={styles.loading}>Loading defaults…</p>
  }

  const shocks = (scenario.shocks as Draft[]) ?? []
  const shock = shocks[0] ?? {}
  const warmupDays = Number(experiment.warmup_days ?? 0)
  const horizonDays = Number(experiment.horizon_days ?? 0)

  function updateShock(field: string, value: number) {
    setScenario((prev) => {
      if (!prev) return prev
      const nextShocks = [...((prev.shocks as Draft[]) ?? [])]
      nextShocks[0] = { ...nextShocks[0], [field]: value }
      return { ...prev, shocks: nextShocks }
    })
  }

  function updateExperimentField(field: string, value: unknown) {
    setExperiment((prev) => (prev ? { ...prev, [field]: value } : prev))
  }

  function updateHeuristicField(field: string, value: unknown) {
    setHeuristicPolicy((prev) => (prev ? { ...prev, [field]: value } : prev))
  }

  function updateLlmField(field: string, value: unknown) {
    setLlmPolicy((prev) => (prev ? { ...prev, [field]: value } : prev))
  }

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault()
    if (!apiKey.trim()) {
      setSubmitError('An OpenAI API key is required to run the LLM agent branch.')
      return
    }
    setSubmitting(true)
    setSubmitError(null)
    try {
      const result = await submitRun({
        network: network!,
        scenario: scenario!,
        heuristic_policy: heuristicPolicy!,
        llm_policy: llmPolicy!,
        experiment: experiment!,
        api_key: apiKey,
      })
      navigate(`/runs/${result.run_id}`)
    } catch (err) {
      setSubmitError((err as Error).message)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div>
      <h1>Run your own experiment</h1>
      <p className={styles.intro}>
        Launches a small, real, live comparison — the heuristic and an LLM agent (using your own
        OpenAI key) respond to the same disruption, paired against an undisrupted control. Capped
        at {limits.max_sandbox_replications} replication{limits.max_sandbox_replications === 1 ? '' : 's'} to
        keep runtime and cost bounded; the network topology stays fixed at the baseline.
      </p>

      <form className={styles.form} onSubmit={handleSubmit}>
        <section className={styles.section}>
          <h2>Disruption timing</h2>
          <p className={styles.sectionHint}>
            The primary port closes for this window. Warm-up runs through day {warmupDays}; normal
            demand and shipments stop after day {horizonDays}.
          </p>
          <div className={styles.fieldRow}>
            <label className={styles.field}>
              Closure starts (day)
              <input
                type="number"
                min={warmupDays + 1}
                max={horizonDays}
                value={Number(shock.physical_start_day ?? 0)}
                onChange={(e) => updateShock('physical_start_day', Number(e.target.value))}
              />
            </label>
            <label className={styles.field}>
              Closure ends (day)
              <input
                type="number"
                min={Number(shock.physical_start_day ?? warmupDays + 1)}
                max={horizonDays}
                value={Number(shock.physical_end_day ?? 0)}
                onChange={(e) => updateShock('physical_end_day', Number(e.target.value))}
              />
            </label>
          </div>
        </section>

        <section className={styles.section}>
          <h2>Experiment scale</h2>
          <p className={styles.sectionHint}>Each replication runs all four paired branches once.</p>
          <div className={styles.fieldRow}>
            <label className={styles.field}>
              Replications
              <input
                type="number"
                min={1}
                max={limits.max_sandbox_replications}
                value={Number(experiment.replications ?? 1)}
                onChange={(e) => updateExperimentField('replications', Number(e.target.value))}
              />
              <span className={styles.fieldHint}>max {limits.max_sandbox_replications}</span>
            </label>
            <label className={styles.field}>
              Random seed
              <input
                type="number"
                value={Number(experiment.base_seed ?? 0)}
                onChange={(e) => updateExperimentField('base_seed', Number(e.target.value))}
              />
            </label>
          </div>
        </section>

        <section className={styles.section}>
          <h2>Heuristic policy</h2>
          <div className={styles.fieldRow}>
            <label className={styles.field}>
              Expedite trigger (days late)
              <input
                type="number"
                min={0}
                value={Number(heuristicPolicy.expedite_trigger_lateness_days ?? 0)}
                onChange={(e) =>
                  updateHeuristicField('expedite_trigger_lateness_days', Number(e.target.value))
                }
              />
              <span className={styles.fieldHint}>
                how many days of predicted lateness before it pays for a rush shipment
              </span>
            </label>
          </div>
        </section>

        <section className={styles.section}>
          <h2>LLM agent policy</h2>
          <div className={styles.fieldRow}>
            <label className={styles.field}>
              Max tool calls per decision
              <input
                type="number"
                min={1}
                max={20}
                value={Number(llmPolicy.max_tool_calls ?? 8)}
                onChange={(e) => updateLlmField('max_tool_calls', Number(e.target.value))}
              />
            </label>
            <label className={styles.field}>
              Fallback policy
              <select
                value={String(llmPolicy.fallback_policy ?? 'HEURISTIC')}
                onChange={(e) => updateLlmField('fallback_policy', e.target.value)}
              >
                <option value="HEURISTIC">Heuristic</option>
                <option value="WAIT">Wait</option>
              </select>
              <span className={styles.fieldHint}>used when the agent abstains or is invalid</span>
            </label>
          </div>
        </section>

        <section className={`${styles.section} ${styles.apiKeySection}`}>
          <h2>Your OpenAI API key</h2>
          <label className={styles.field}>
            API key
            <input
              type="password"
              autoComplete="off"
              placeholder="sk-..."
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
            />
          </label>
          <p className={styles.apiKeyNote}>
            Held only in this page's memory for this submission — never stored, logged, or sent
            anywhere except as this one run's OpenAI credential. Lost on refresh.
          </p>
        </section>

        <div className={styles.submitRow}>
          <button type="submit" className={styles.submitButton} disabled={submitting}>
            {submitting ? 'Submitting…' : 'Run experiment'}
          </button>
          {submitError && <span className={styles.submitError}>{submitError}</span>}
        </div>
      </form>
    </div>
  )
}
