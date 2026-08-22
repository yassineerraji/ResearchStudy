// Lets a visitor launch their own small live experiment: pick one of the
// nine real topology x severity presets from the actual research grid
// (a preset picker only — no free-form shock editor, by design), then
// fine-tune the disruption's timing/uncertainty parameters, replication
// count/seed, heuristic aggressiveness, and LLM agent behavior, and submit
// against their own OpenAI API key and model.
//
// Submission goes straight to POST /api/v1/runs, which re-validates the
// whole bundle server-side through the same `resolve_config` path
// `/configs/validate` uses (see webapp/backend/app/services/run_launcher.py)
// — this form does not duplicate that validation, only basic input bounds.

import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { getConfigDefaults, getPreset, getRunLimits, listPresets, submitRun } from '../api/client'
import type { Preset, RunLimits } from '../api/types'
import styles from './NewRun.module.css'

type Draft = Record<string, unknown>

const DEFAULT_PRESET_ID = 'standard_medium'

export default function NewRun() {
  const navigate = useNavigate()

  const [presets, setPresets] = useState<Preset[] | null>(null)
  const [selectedPresetId, setSelectedPresetId] = useState(DEFAULT_PRESET_ID)
  const [network, setNetwork] = useState<Draft | null>(null)
  const [scenario, setScenario] = useState<Draft | null>(null)
  const [heuristicPolicy, setHeuristicPolicy] = useState<Draft | null>(null)
  const [llmPolicy, setLlmPolicy] = useState<Draft | null>(null)
  const [experiment, setExperiment] = useState<Draft | null>(null)
  const [limits, setLimits] = useState<RunLimits | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [presetLoading, setPresetLoading] = useState(false)

  const [apiKey, setApiKey] = useState('')
  const [model, setModel] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState<string | null>(null)

  useEffect(() => {
    Promise.all([
      listPresets(),
      getConfigDefaults('network'),
      getConfigDefaults('scenario'),
      getConfigDefaults('heuristic_policy'),
      getConfigDefaults('llm_policy'),
      getConfigDefaults('experiment'),
      getRunLimits(),
    ])
      .then(([p, n, s, h, l, e, lim]) => {
        setPresets(p)
        // The defaults endpoints already point at Standard x Medium's real
        // files, so they match DEFAULT_PRESET_ID's content exactly — no
        // second fetch needed just to populate the initial form state.
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

  const topologyOrder = useMemo(() => {
    const seen: string[] = []
    for (const preset of presets ?? []) {
      if (!seen.includes(preset.topology)) seen.push(preset.topology)
    }
    return seen
  }, [presets])

  async function handlePresetChange(presetId: string) {
    setSelectedPresetId(presetId)
    setPresetLoading(true)
    setLoadError(null)
    try {
      const preset = await getPreset(presetId)
      setNetwork(preset.network)
      setScenario(preset.scenario)
      setExperiment((prev) => {
        if (!prev) return prev
        const next: Draft = { ...prev }
        if (preset.base_seed !== null) next.base_seed = preset.base_seed
        if (preset.warmup_days !== null) next.warmup_days = preset.warmup_days
        if (preset.horizon_days !== null) next.horizon_days = preset.horizon_days
        if (preset.drain_days !== null) next.drain_days = preset.drain_days
        if (preset.terminal_penalty_days !== null) next.terminal_penalty_days = preset.terminal_penalty_days
        return next
      })
    } catch (err) {
      setLoadError((err as Error).message)
    } finally {
      setPresetLoading(false)
    }
  }

  if (loadError) return <p className={styles.loadError}>Could not load defaults: {loadError}</p>
  if (!presets || !network || !scenario || !heuristicPolicy || !llmPolicy || !experiment || !limits) {
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
    if (!model.trim()) {
      setSubmitError('A model name is required — you pay for your own run, so you choose the model.')
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
        model,
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
        OpenAI key and model) respond to the same disruption, paired against an undisrupted
        control. Capped at {limits.max_sandbox_replications} replication
        {limits.max_sandbox_replications === 1 ? '' : 's'} to keep runtime and cost bounded.
      </p>

      <form className={styles.form} onSubmit={handleSubmit}>
        <section className={styles.section}>
          <h2>Network &amp; severity</h2>
          <p className={styles.sectionHint}>
            One of the nine real topology x severity combinations from the actual research grid —
            loads that combination's exact validated network and scenario files.
          </p>
          <label className={styles.field}>
            Preset
            <select
              value={selectedPresetId}
              disabled={presetLoading}
              onChange={(e) => handlePresetChange(e.target.value)}
            >
              {topologyOrder.map((topology) => (
                <optgroup key={topology} label={topology}>
                  {presets
                    .filter((p) => p.topology === topology)
                    .map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.severity}
                      </option>
                    ))}
                </optgroup>
              ))}
            </select>
          </label>
        </section>

        <section className={styles.section}>
          <h2>Disruption timing</h2>
          <p className={styles.sectionHint}>
            The disruption's real start day, duration, and disclosure delay are each sampled once
            per replication around these parameters, not fixed in advance. Warm-up runs through
            day {warmupDays}; normal demand and shipments stop after day {horizonDays}.
          </p>
          <div className={styles.fieldRow}>
            <label className={styles.field}>
              Planned start day
              <input
                type="number"
                min={warmupDays + 1}
                max={horizonDays}
                value={Number(shock.planned_start_day ?? 0)}
                onChange={(e) => updateShock('planned_start_day', Number(e.target.value))}
              />
            </label>
            <label className={styles.field}>
              Start day jitter (± days)
              <input
                type="number"
                min={0}
                value={Number(shock.start_day_jitter_days ?? 0)}
                onChange={(e) => updateShock('start_day_jitter_days', Number(e.target.value))}
              />
            </label>
            <label className={styles.field}>
              Max information delay (days)
              <input
                type="number"
                min={0}
                value={Number(shock.max_information_delay_days ?? 0)}
                onChange={(e) => updateShock('max_information_delay_days', Number(e.target.value))}
              />
              <span className={styles.fieldHint}>how late either policy might learn about it</span>
            </label>
          </div>
          <div className={styles.fieldRow}>
            <label className={styles.field}>
              Duration — minimum (days)
              <input
                type="number"
                min={1}
                value={Number(shock.minimum_duration_days ?? 1)}
                onChange={(e) => updateShock('minimum_duration_days', Number(e.target.value))}
              />
            </label>
            <label className={styles.field}>
              Duration — mean (days)
              <input
                type="number"
                min={1}
                value={Number(shock.duration_mean_days ?? 1)}
                onChange={(e) => updateShock('duration_mean_days', Number(e.target.value))}
              />
            </label>
            <label className={styles.field}>
              Duration — std dev (days)
              <input
                type="number"
                min={0}
                value={Number(shock.duration_std_days ?? 0)}
                onChange={(e) => updateShock('duration_std_days', Number(e.target.value))}
              />
            </label>
            <label className={styles.field}>
              Duration — maximum (days)
              <input
                type="number"
                min={1}
                value={Number(shock.maximum_duration_days ?? 1)}
                onChange={(e) => updateShock('maximum_duration_days', Number(e.target.value))}
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
          <h2>Your OpenAI API key &amp; model</h2>
          <div className={styles.fieldRow}>
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
            <label className={styles.field}>
              Model
              <input
                type="text"
                autoComplete="off"
                placeholder="gpt-4.1"
                value={model}
                onChange={(e) => setModel(e.target.value)}
              />
              <span className={styles.fieldHint}>any model name your key has access to</span>
            </label>
          </div>
          <p className={styles.apiKeyNote}>
            Both are held only in this page's memory for this submission — never stored, logged,
            or sent anywhere except as this one run's own OpenAI credential and model choice. Lost
            on refresh. You are billed directly by OpenAI for your own run.
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
