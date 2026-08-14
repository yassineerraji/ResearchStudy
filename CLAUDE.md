# Supply-Chain Agent Evaluation — Build Contract

> **Project:** `supply-chain-agent-evaluation`
> **Primary language:** Python
> **Implementation style:** small modular monolith
> **Core principle:** build the smallest correct system that can answer the research question fairly and reproducibly

## Document status: two parts

This file is in **two parts**, kept in one document deliberately so V2 is always read against what it extends.

- **Part 1 — Version 1 Build Contract** (immediately below, §1–§10). **Complete, built, tested, and frozen.** 323 automated tests pass; the real 100-replication-per-profile experiment (Light/Medium/Heavy) has been run against the live OpenAI API and the results audited. Part 1 is a **condensed reference** — it states what already exists and must keep working, at the level of detail a coding agent needs to extend the system safely, not the original line-by-line build contract that produced it (that exhaustive version remains available in git history if ever needed). Part 1 is reference, not a to-do list.
- **Part 2 — Version 2 Build Contract** (appended after Part 1's §10, its own V2.1–V2.12). This is the new, active contract: it extends Part 1's already-built architecture with the three axes of improvement identified from auditing V1's results — richer randomness, a topology × severity experimental grid, and an expanded, more realistic disruption taxonomy. Part 2 assumes Part 1 exists and works; it does not restate what does not change.

**Cross-reference convention:** anywhere either part needs to point at the other, it says so explicitly — "V1 §6" or "V2 §3" — never a bare section number. Part 2's own sections are numbered V2.1–V2.12; they are not a continuation of Part 1's 1–10.

**Instruction hierarchy (applies to both parts, per V1 §1):** Yassine's current explicit instruction for the active task still outranks either part's own invariants. When working on V2, V2's invariants are the relevant ones to check against; Part 1's invariants remain authoritative for anything V2 does not explicitly change. When working on a V1 bug fix, Part 1 remains authoritative and V2's not-yet-built mechanics are irrelevant. A lower-level source must never silently override a higher-level one, and a newly discovered ambiguity in either part must be raised, not guessed at.

---

# Part 1 — Version 1 Build Contract (Reference)

Version 1 is **complete, built, tested, and frozen** — 323 automated tests pass, and the real 100-replication-per-profile experiment (Light/Medium/Heavy) has been run against the live OpenAI API and audited. This part is a **compact reference** for what already exists and must keep working, not a build checklist. Part 2 extends this architecture; nothing here is still open, and nothing here should be re-derived from first principles — if a genuine ambiguity shows up, it is a documentation gap or a proposal for a future revision, not license to invent behavior.

## V1 §1. Mission, scope, and instruction hierarchy

**Research question:** under identical disrupted supply-chain conditions, can a bounded LLM-enabled policy produce valid shipment-level decisions that reduce the incremental cost of disruption more effectively than a classical heuristic — without the comparison favoring either policy?

**What V1 is:** a discrete-time (whole days), single-product, transportation-and-inventory simulator over a directed logistics network (suppliers → ports → hubs → one plant). Shipments release on a fixed daily schedule, face stochastic demand, ordinary transport delays, and a temporary designed disruption. A policy may intervene on a shipment waiting at a node with exactly one action:

- `WAIT`, `REROUTE`, `EXPEDITE`, `ABSTAIN`

Policies never create supply, change demand, modify the network, split shipments, or bypass validation — they only choose among these four actions for one shipment at a time, from a read-only observation.

**Explicit non-goals (V1):** multi-product handling, supplier selection/procurement, shipment splitting, joint multi-shipment optimization, dynamic pricing, fleet routing, reinforcement learning, a database, a web UI, distributed/real-time execution, uncertain disruption duration, and delayed/asymmetric information. (V2 §2 lifts the last two of these — see there.)

**Instruction hierarchy:** (1) Yassine's current explicit instruction for the active task, (2) the scientific invariants in this file, (3) repository-wide workflow rules, (4) existing tests/interfaces, (5) existing implementation. A lower-level source never silently overrides a higher one; a genuinely new ambiguity gets raised, not guessed at.

## V1 §2. Formal model essentials

The network is a directed graph $\bar G=(V,E)$ of `Node`s (`SUPPLIER`/`PORT`/`HUB`/`PLANT`/`CUSTOMER`, `PLANT` is the V1 destination) and `Edge`s (`ROAD`/`RAIL`/`SEA`/`AIR`), each with capacity, cost, lead-time, and reliability fields. The base graph is immutable; each day's **operational** graph $G_t=\Phi(\bar G, Z_t)$ applies temporary multipliers (`capacity_multiplier`, `lead_time_multiplier`, `cost_multiplier`, `available`) from active shocks — always computed fresh from the immutable base values each day, never compounded onto yesterday's already-modified value.

Daily state $S_t=(G_t, I_t, B_t, H_t, D_t, Z_t, C_t)$ (network, inventory, backlog, shipments, demand, active disruptions, costs) evolves via a shared, deterministic transition $S_{t+1}=T(S_t,A_t,w_t)$ — identical inputs always produce identical outputs. Policies never see `SimulationState` directly, only a built `DecisionObservation`.

**Cost objective** for one run:

$$J = C_{transport}+C_{reroute}+C_{expedite}+C_{holding}+C_{backlog}+C_{late}+C_{terminal}$$

**Paired disruption metric**, per policy $p$, scenario $s$, replication $r$:

$$TCD_{p,s,r}=J^{disrupted}_{p,s,r}-J^{undisrupted}_{p,s,r} \qquad \Delta_{s,r}=TCD_{LLM,s,r}-TCD_{heuristic,s,r}$$

$\Delta<0$ favors the LLM, $\Delta>0$ favors the heuristic. This pairing — one seed, one demand/delay draw, four cloned branches (heuristic×undisrupted, heuristic×disrupted, LLM×undisrupted, LLM×disrupted) sharing one warm-up snapshot — is the core fairness mechanism the whole system exists to protect, and V2 does not change it.

## V1 §3. Global conventions

- Time is integer days; day 0 is the loaded initial state.
- Quantities (product, inventory, demand, shipment, capacity) are integers; **shipment splitting is forbidden**; costs are floats, rounded only for display (6 decimals in output).
- Shipment IDs: `shipment_<release_day:03d>_<sequence:03d>`; route IDs are edge IDs joined by `__`.
- Deterministic ordering is used wherever order could affect results: shipment allocation by (earliest due day, earliest release day, lexicographic ID); route tie-breaking by (lowest estimated cost, earliest arrival, fewest edges, lexicographic route ID). Dictionary iteration order is never relied on.
- `COST_TOLERANCE = 1e-9` for algorithmic equality; `OUTPUT_TIE_TOLERANCE = 0.01` (one currency cent) for policy-result ties.

## V1 §4. Architecture

Runtime dependencies are deliberately minimal: `networkx` (routing), `pydantic` (strict config validation), `PyYAML`, `openai` (Responses API). Standard library covers CSV/JSON/hashing/random/logging/paths. No database, web framework, or async runtime.

Package dependency direction is one-way:

```text
domain → simulation/decisions → policies → experiments → cli
integrations is used only by the LLM policy; data_io constructs/serializes but owns no simulation logic.
```

`domain` imports nothing else in the project; `simulation` never imports a concrete policy class (the engine never branches on policy type); circular imports are forbidden.

Repository structure:

```text
supply-chain-agent-evaluation/
├── CLAUDE.md, README.md, pyproject.toml, .gitignore, .env.example
├── configs/{networks,scenarios,policies,experiments}/*.yaml
├── data/, outputs/
├── src/supply_chain_simulator/
│   ├── domain/        {models,state,actions,events}.py
│   ├── simulation/     {engine,transition,routing,costs}.py
│   ├── decisions/       {observation,validator}.py
│   ├── policies/         {base,heuristic,llm_agent,fallback}.py
│   ├── experiments/       {event_tape,runner,metrics}.py
│   ├── integrations/       llm_client.py
│   └── data_io/              {loaders,writers}.py
├── analysis/            plot_results.py   (standalone; outside mypy/pytest scope)
└── tests/{unit,integration,fixtures}/
```

No additional architectural folder exists; `outputs/<experiment_id>__<UTC_TIMESTAMP>/` subfolders are generated at runtime, not source.

## V1 §5. Package responsibilities

Each entry below is a contract, not just documentation — "unchanged" (as V2 uses it throughout Part 2) means unchanged in public behavior, not merely untouched by an editor.

- **`domain/models.py`** — immutable `Node`, `Edge`, `Product`, `NetworkDefinition`, `Route` (frozen dataclasses; base network definitions, never mutated).
- **`domain/state.py`** — mutable `SimulationState`, `Shipment` (statuses `AT_NODE`/`IN_TRANSIT`/`DELIVERED`), operational node/edge state, cost/service counters, `SimulationResult`.
- **`domain/actions.py`** — `ActionType` (`WAIT`/`REROUTE`/`EXPEDITE`/`ABSTAIN`), `ReasonCode`, `DecisionAction`, `ValidationCode`/`ValidationResult`.
- **`domain/events.py`** — `ShockType` (`NODE_CLOSURE`/`EDGE_CLOSURE`/`NODE_CAPACITY_REDUCTION`/`EDGE_CAPACITY_REDUCTION`/`EDGE_LEAD_TIME_INCREASE`/`EDGE_COST_INCREASE`), `TargetType` (`NODE`/`EDGE`), `Shock`, `DemandEvent`, `ShipmentReleaseEvent`, `EventTape`. Unused multiplier fields on a `Shock` default to, and stay, `1.0`.
- **`simulation/routing.py`** — all route enumeration/estimation over a `networkx.DiGraph`; candidates are simple paths ≤6 edges, deterministically sorted, capped at 5 options; never reserves capacity; has no side effects.
- **`simulation/costs.py`** — the shared cost formulas: transport charged once per edge entry (`qty × unit_cost`); reroute/expedite charged once per executed action (`qty × per-unit rate`); holding/backlog charged end-of-day on standing quantities; late penalty charged once on delivery (`qty × max(0, delivered_day − due_day) × rate`); terminal cost applied only if unresolved state remains at the max drain day. These are the formulas V2 §2 keeps unchanged.
- **`simulation/transition.py`** — pure daily-state-change functions (release, arrivals, demand/backlog fulfilment, decision triggers, validated-action application, departure allocation, end-of-day costs). Calls no LLM and chooses no actions itself.
- **`simulation/engine.py`** — the daily orchestration loop; deep-clones the initial state, never mutates the caller's snapshot, never branches on concrete policy type, fails on violated invariants.
- **`decisions/observation.py`** — builds the read-only `DecisionObservation` (shipment/destination/shock context, current plan, route options) both policies receive identically; never exposes future randomness or mutable state.
- **`decisions/validator.py`** — `validate_action(...)`: schema → shipment identity/state → route requirement/membership/continuity/availability/capacity → emergency-semantics match. Never silently repairs an invalid action.
- **`policies/heuristic.py`** — deterministic classical benchmark: build a `WAIT` candidate from the current plan, a `REROUTE` candidate per non-emergency route, an `EXPEDITE` candidate per emergency route (kept only if `WAIT` lateness ≥ `expedite_trigger_lateness_days` or no reroute exists); pick lowest `estimated_total_cost` (transport + action cost + estimated late penalty only — it does not estimate future holding/backlog cost, by design); ties break `WAIT` > `REROUTE` > `EXPEDITE` > lexicographic route ID.
- **`policies/llm_agent.py`** — bounded tool-using agent. Approved tools, exactly: `get_shipment_context`, `get_destination_context`, `list_route_options`, `inspect_route`, `submit_action` — all deterministic, read-only views over one immutable observation; none can mutate state, create a route, or see future event-tape values. Max 8 tool calls per decision; hitting the limit without a submission produces `ABSTAIN`.
- **`policies/fallback.py`** — `ABSTAIN`/invalid → configured fallback (default: heuristic) → if still invalid, terminal safe `WAIT`; every stage logged separately.
- **`integrations/llm_client.py`** — `OpenAIResponsesClient` (Responses API, function tools, `store=False`, temperature 0, 60s timeout, 3 retries on transient errors only — infrastructure failure **fails the experiment**, it is never silently downgraded to a fallback), `ReplayLLMClient` (exact rerun from a recorded trace, matched by `decision_key`), `FakeLLMClient` (tests only).
- **`experiments/event_tape.py`** — all policy-independent randomness. `replication_seed = base_seed + replication`; per-stream seeds via `sha256(f"{replication_seed}:{stream_name}")`, first 8 bytes as an unsigned int. V1's two required streams: `demand`, `edge_delays`. Draw order is always deterministic (days → destinations → products/edges ascending). Demand: `round(gauss(mean, std))` clamped to config bounds. Ordinary edge delay: 0 days if a draw ≤ reliability, else 1 — drawn for every edge on every day regardless of use, so policy choices never alter the random sequence. **All randomness for a replication is drawn upfront, before any policy makes a decision** — this is the fairness mechanism V2's new randomness sources (V2 §3.6) must also honor. The undisrupted tape is the disrupted tape with only the designed shock(s) removed; demand, releases, and ordinary delays stay identical.
- **`experiments/runner.py`** — orchestrates one warm-up + four cloned branches per replication, in a fixed order; fails the whole experiment on any branch failure (`fail_fast`).
- **`experiments/metrics.py`** — run-level metrics (fill rates, cost components, decision-quality rates) and the experiment-level summary (mean/median TCD and delta, 95% CI, win/loss/tie rates via a ±0.01 currency-cent tie band).
- **`data_io/loaders.py`** — strict (`extra="forbid"`) Pydantic config models; validates cross-references, route continuity, path containment; redacts secrets in any written-out resolved config.
- **`data_io/writers.py`** — one output directory per run; manifest, resolved config, event tapes, metrics, decision traces, LLM interactions, summary; atomic write-then-rename; flushes after each replication.
- **`cli.py`** — `argparse`-based `validate-config` and `run` commands only; no scientific-parameter CLI overrides (change the config file instead); exit codes 0/1/2/3/4 for success/unexpected/config/invariant/LLM-integration errors.

## V1 §6. Configuration files

Four config kinds compose one experiment (`configs/networks/`, `configs/scenarios/`, `configs/policies/`, `configs/experiments/`), each schema-versioned and cross-validated by `data_io/loaders.py`. Exact current field values live in the files themselves, not duplicated here; in brief:

- **`networks/baseline_network.yaml`** — the one network V1 experiments used: 5 nodes (`supplier_1`, `port_primary`, `port_alternative`, `hub_1`, `plant_1`), 6 edges (including one `AIR` emergency edge), one product (`component_a`), 200 units initial plant inventory, truncated-normal demand (mean 40, std 5, clamped [25,55]), and a replenishment plan releasing a fixed 40-unit shipment every day with a 15-day due offset. This file is the **Standard** topology tier in V2 (V2 §3.1) and stays byte-identical.
- **`scenarios/port_closure.yaml`** — the baseline designed disruption: `port_primary` fully closed days 21–27, disclosed immediately.
- **`policies/heuristic.yaml`** / **`policies/llm_agent.yaml`** — heuristic tuning (`expedite_trigger_lateness_days`, `cost_tolerance`) and LLM provider/execution settings (model, temperature 0, tool-call/token/timeout/retry limits, fallback policy, live-vs-replay mode).
- **`experiments/baseline_comparison.yaml`** — ties one network + one scenario + both policy configs together with `warmup_days: 20`, `horizon_days: 60`, `drain_days: 30`, `terminal_penalty_days: 30`, `replications: 100`, `base_seed: 1042`.

## V1 §7. Daily simulation sequence

Day 0 is the loaded initial state (200 units plant inventory, no shipments, no active shocks). Days 1–20 are warm-up: the undisrupted tape, decisions disabled, no designed shock — its purpose is to build a realistic in-transit pipeline. At the end of day 20 the physical state (inventory, backlog, shipments, positions) is cloned forward but cost/service counters reset to zero, so **evaluation costs start on day 21**.

Every day thereafter follows this fixed order — the one sequence every fairness and reproducibility guarantee in this system rests on:

1. Begin day — reset daily capacity usage.
2. Reveal information — add shocks whose `information_day` matches today to known shocks.
3. Apply physical shocks — rebuild operational multipliers/availability fresh from the immutable base values.
4. Process arrivals — deliver or advance in-transit shipments due today; postpone on unavailable/full destination.
5. Release scheduled shipments — validate source availability/capacity/storage (V1 guarantees this always succeeds; V2 §3.7 changes this step to *defer* instead of raise).
6. Realize and fulfil demand — inventory clears backlog first, then same-day demand; shortfall becomes backlog.
7. Build the decision set — one observation per non-delivered at-node shipment, all from the same pre-action state.
8. Request policy decisions, in deterministic shipment order.
9. Validate and resolve fallback for each proposed action.
10. Apply valid route changes (`REROUTE`/`EXPEDITE`); charge action cost.
11. Allocate departures under node/edge capacity, in deterministic order; no partial dispatch.
12. Charge end-of-day holding and backlog costs.
13. Record daily metrics; assert all state invariants (non-negativity, capacity limits, shipment-state consistency, product-balance — e.g. released quantity always reconciles against inventory, backlog, and in-system shipments) — fail immediately on violation.

A shipment requires a decision only when it's `AT_NODE`, not at its destination, and at least one trigger holds: an active shock affects its remaining route, its next edge/node is unavailable, its estimated arrival is already late, `capacity_wait_days ≥ 2`, or its route is malformed. Otherwise it continues automatically.

**Shock semantics:** node closure blocks arrivals/departures and excludes the node from new route candidates; edge closure blocks new entries only; capacity reductions multiply throughput; lead-time/cost increases apply only to new edge entries while active. Overlapping shocks combine (availability AND, multipliers multiplied), always recomputed from base values — an "unused multiplier stays `1.0`" convention (V1 §5) applies to every shock type.

**Drain and termination:** after day 60, no new demand/shipments; the sim keeps running until all shipments are delivered and backlog is zero, or day 90 (max drain), whichever comes first; unresolved state at day 90 is charged a terminal cost and flagged `terminated_with_unresolved_state`.

## V1 §8. Outputs

One directory per run, `outputs/<experiment_id>__<UTC_TIMESTAMP>/`, containing: `manifest.json` (config/prompt hashes, git commit, dependency versions), `resolved_config.yaml` (secrets redacted), `run.log`, `event_tapes.jsonl`, `run_metrics.csv` (one row per branch), `daily_metrics.csv`, `decision_traces.jsonl`, `llm_interactions.jsonl` (LLM decisions only), `replications.csv` (one row per replication: both TCDs, delta, winner), `summary.json` (aggregate stats, win/loss/tie rates, 95% CI). Written atomically, flushed after every replication so a `fail_fast` abort never loses completed work.

## V1 §9. Testing and quality bar

323 automated tests (`tests/unit/`, `tests/integration/`) cover every domain invariant, config-loading failure mode, event-tape reproducibility property, transition/routing/cost calculation, validator code, heuristic decision, and full paired-experiment/reproducibility guarantee — with a manually-checkable tiny fixture network for exact end-to-end assertions. No test ever calls a real API. Ruff and Mypy (strict, `src/` only) pass; coverage stays ≥90%.

V1 was built across nine milestones (bootstrap → domain/config → event tape → routing/costs → core simulation → observation/validation → heuristic policy → experiment runner/outputs → LLM integration), then run end-to-end against the live API — all now complete. No milestone was started before its required predecessor was verified passing; V2's own milestone sequence (Part 2) follows the same discipline.

## V1 §10. Final locked decisions for Version 1

The following are settled:

| Topic | Version 1 decision |
|---|---|
| Time resolution | Daily |
| Product count | One |
| Decision unit | Complete shipment |
| Shipment splitting | No |
| Policy action timing | Shipment at node only |
| Action space | WAIT, REROUTE, EXPEDITE, ABSTAIN |
| Baseline disruption | Primary-port closure, days 21–27 |
| Shock information | Complete and immediate |
| Demand | Truncated normal, pre-generated |
| Ordinary delays | Pre-generated 0/1 edge-day delay |
| Randomness | Shared paired event tape |
| Warm-up | 20 days, decisions disabled |
| Evaluation horizon | Day 60 |
| Drain | Up to 30 days |
| Counterfactual | Policy-specific |
| Heuristic | Lowest estimated candidate cost |
| LLM tools | Five approved read-only/submission tools |
| LLM fallback | Heuristic, then terminal WAIT |
| Live provider failure | Fail experiment |
| Exact LLM rerun | Replay recorded trace |
| Output storage | Files only |
| Runtime architecture | Modular monolith |
| CLI | argparse, validate-config and run |
| Tests | No live API |
| Primary metric | Paired TCD delta |
| Success condition | Valid fair comparison, not an LLM win |

No coding agent may change these decisions without explicit approval from Yassine.

---

# Part 2 — Version 2 Build Contract

## Why V2 exists

V1 answered its research question cleanly: across 300 real replications (100 each for a Light/Medium/Heavy disruption), the LLM agent beat the classical heuristic overwhelmingly under Medium and Heavy disruptions and lost consistently under Light ones. But auditing those results surfaced something the V1 design didn't anticipate: **within every single profile, all 100 replications agreed on the winner — 100%/0%, never a mixed result.** A full audit (see `REPORT.md` and the conversation history around it) traced this to three compounding, fixable properties of V1's design, not to a flaw in the comparison itself:

1. The mean cost gap between policies is 7.6×–10.2× larger than the random noise between replications, in every profile — the two policies are separated by a gap the environment's randomness was never designed to be able to bridge.
2. The shipment release schedule is **fully deterministic** (exactly 40 units, every single day) and the disruption's timing, duration, and disclosure are all **fully known in advance** — so the same handful of shipments hit the same disruption the same way in every replication, regardless of the random demand/delay draws.
3. The network has only 5 nodes, 6 edges, and effectively 3 end-to-end routes — not enough structural diversity for decisions to ever land close to a genuine tie.

V2 exists to fix exactly these three things — and only these three, plus the richer disruption taxonomy that falls naturally out of fixing them — without touching anything about V1 that already works. **V1's simulation physics, cost model, action space, policy protocol, LLM integration, CLI, and output schemas are not being redesigned.** V2 extends the *inputs* those mechanisms already handle correctly: more varied topologies, more varied and realistic disruptions, and more sources of randomness feeding the same paired-replication machinery V1 already validated.

## V2.1 Mission delta

V1's research question was:

> Under identical disrupted supply-chain conditions and operational constraints, can a bounded LLM-enabled policy produce valid shipment-level decisions that reduce the incremental cost of disruption more effectively and robustly than a classical heuristic?

V2 extends it to explicitly make topology and disruption character first-class variables, not fixed backdrop:

> **Across a range of network topologies and a realistic range of disruption types, timings, and severities — each with genuine, unresolved-until-it-happens uncertainty — does a bounded LLM-enabled policy produce valid shipment-level decisions that reduce the incremental cost of disruption more effectively and more robustly than a classical heuristic, and how does that advantage vary with network complexity and disruption character?**

The system must still compare policies, not give one a more favorable world — that invariant from V1 §1 is unchanged and, if anything, more load-bearing now that topology and disruption realization both vary: every new source of variation introduced below must still be drawn once, before either policy sees it, and shared identically between the heuristic and LLM branches for a given replication and topology cell.

## V2.2 Scope changes from Version 1

### V1 non-goals now explicitly in scope

Two items V1 §1 explicitly excluded are now, deliberately, in scope:

- **"Uncertain disruption duration"** — a disruption's realized duration is now drawn from a distribution per replication, not fixed in the scenario file.
- **"Delayed or asymmetric information"** — a disruption's `information_day` may now fall after its `physical_start_day`, drawn from a distribution, so policies can genuinely be caught by surprise.

Everything else in V1 §1's non-goals list **stays excluded** — V2 does not introduce multi-product handling, supplier-selection decisions, procurement, shipment splitting, joint multi-shipment optimization, dynamic pricing, fleet-level routing, reinforcement learning, a database, a web UI, or distributed/real-time execution. Supplier-side shocks (V2.3.2) make a supplier node disruptable; they do not make supplier *choice* a policy action — the policy still only ever chooses among WAIT / REROUTE / EXPEDITE / ABSTAIN for an existing shipment, exactly as in V1 §1.

### New in scope

- **Network topology as an experimental variable.** Three topology tiers (V2.3.1), crossed with the three severity tiers V1 already built, forming a 3×3 experimental grid (V2.8).
- **A richer shock taxonomy**: demand-side shocks, supplier-side shocks, and compound/cascading multi-shock events (V2.3.2–V2.3.4), alongside V1's existing six network-side shock types, which are unchanged.
- **Shock realization from distributions** rather than fixed scenario values (V2.3.3): start-day jitter, duration uncertainty, and information delay are all sampled once per replication from a documented distribution, using the same deterministic, auditable seeding approach V1 already uses for demand and ordinary delays.
- **Shipment quantity randomness** in the release schedule (V2.3.5): release timing stays on V1's fixed daily cadence (this is a deliberate, scoped choice — see V2.3.5), but the quantity released each day is now drawn from a distribution instead of always being exactly 40 units.

### What must not change

- The four-action policy interface (WAIT / REROUTE / EXPEDITE / ABSTAIN), the shared validator, the fallback chain, and the LLM's five approved tools — all unchanged.
- The daily event order (V1 §7) — unchanged in sequence; V2.3.7 lists the two narrow, additive exceptions required to keep it correct under the new randomness.
- The cost model (V1 §5, §2) and every cost formula in V1 §5 — unchanged. New shock types produce new *inputs* to existing formulas (e.g., a demand shock changes `D_t`; it does not add a new cost term).
- The paired TCD/delta metric (V1 §2) — unchanged in definition. It is now computed per (topology tier, severity tier) cell instead of once globally; V2.8 defines exactly how the aggregate reporting extends.
- Every already-built package (`domain`, `simulation`, `decisions`, `policies`, `experiments`, `integrations`, `data_io`) keeps its existing responsibility boundaries (V1 §4). V2's new mechanics are additive to these packages, not a new architectural layer.

## V2.3 Formal model extensions

### V2.3.1 Topology tiers

Three fixed network topologies, independent of and crossed with the three severity tiers (V2.8). **Naming is deliberately distinct from the severity tiers** ("Light/Medium/Heavy") to prevent the two axes from ever being confused in code, config, output files, or conversation: topology tiers are named **Compact / Standard / Extended**.

- **Standard** is V1's existing, unmodified `configs/networks/baseline_network.yaml` (5 nodes, 6 edges) — kept byte-identical so every V1 result remains a valid, comparable reference point inside the V2 grid (it is exactly the "Standard × Medium" cell).
- **Compact** removes the alternate port entirely, so a primary-port disruption leaves REROUTE structurally impossible — only WAIT and EXPEDITE remain, a deliberately more constrained decision environment than Standard.
- **Extended** adds a third port and a second hub, giving genuine mesh redundancy — two viable multi-hop alternatives to the primary route instead of Standard's one.

All three tiers share the exact same product, initial inventory, demand process, and (template) replenishment plan (V2.3.5) — **only the network's nodes and edges differ between tiers.** This isolates topology's effect on routing flexibility from any confound with shipment volume or demand.

#### Compact (`configs/networks/topology_compact.yaml`)

**Nodes (4):**

| node_id | node_type | storage_capacity | processing_capacity | source_capacity |
|---|---|---:|---:|---:|
| `supplier_1` | SUPPLIER | 2000 | 200 | 100 |
| `port_primary` | PORT | 2000 | 200 | 0 |
| `hub_1` | HUB | 2000 | 200 | 0 |
| `plant_1` | PLANT | 1000 | 200 | 0 |

**Edges (4):**

| edge_id | origin → destination | mode | distance_km | base_lead_time_days | daily_capacity | unit_transport_cost | reliability | emergency |
|---|---|---|---:|---:|---:|---:|---:|---|
| `supplier_to_primary_port` | supplier_1 → port_primary | ROAD | 400 | 2 | 100 | 4.00 | 0.97 | false |
| `primary_port_to_hub` | port_primary → hub_1 | SEA | 6000 | 10 | 200 | 8.00 | 0.94 | false |
| `hub_to_plant` | hub_1 → plant_1 | ROAD | 300 | 2 | 120 | 3.00 | 0.98 | false |
| `supplier_to_plant_air` | supplier_1 → plant_1 | AIR | 6200 | 2 | 40 | 40.00 | 0.99 | **true** |

Every value is copied unchanged from Standard's corresponding edge — Compact is Standard minus the alternate port and its two edges, nothing more, so any behavioral difference between Compact and Standard is attributable to route *availability*, not to different underlying per-edge economics.

#### Standard (`configs/networks/baseline_network.yaml`)

Unchanged from V1 §6 — 5 nodes, 6 edges. Not reproduced here; see V1 §6 for the exact file content.

#### Extended (`configs/networks/topology_extended.yaml`)

**Nodes (7):**

| node_id | node_type | storage_capacity | processing_capacity | source_capacity |
|---|---|---:|---:|---:|
| `supplier_1` | SUPPLIER | 2000 | 200 | 100 |
| `port_primary` | PORT | 2000 | 200 | 0 |
| `port_alternative` | PORT | 1200 | 80 | 0 |
| `port_tertiary` | PORT | 1200 | 90 | 0 |
| `hub_1` | HUB | 2000 | 200 | 0 |
| `hub_2` | HUB | 1500 | 150 | 0 |
| `plant_1` | PLANT | 1000 | 200 | 0 |

`supplier_1` and `plant_1` are deliberately identical to Standard/Compact (same source and demand-side economics across every tier — richness is added only in the middle of the network).

**Edges (10):**

| edge_id | origin → destination | mode | distance_km | base_lead_time_days | daily_capacity | unit_transport_cost | reliability | emergency |
|---|---|---|---:|---:|---:|---:|---:|---|
| `supplier_to_primary_port` | supplier_1 → port_primary | ROAD | 400 | 2 | 100 | 4.00 | 0.97 | false |
| `supplier_to_alternative_port` | supplier_1 → port_alternative | ROAD | 650 | 4 | 60 | 7.00 | 0.96 | false |
| `supplier_to_tertiary_port` | supplier_1 → port_tertiary | ROAD | 800 | 5 | 60 | 8.50 | 0.95 | false |
| `primary_port_to_hub_1` | port_primary → hub_1 | SEA | 6000 | 10 | 200 | 8.00 | 0.94 | false |
| `alternative_port_to_hub_1` | port_alternative → hub_1 | SEA | 6700 | 12 | 60 | 11.00 | 0.93 | false |
| `alternative_port_to_hub_2` | port_alternative → hub_2 | SEA | 5800 | 9 | 50 | 10.00 | 0.94 | false |
| `tertiary_port_to_hub_2` | port_tertiary → hub_2 | SEA | 6200 | 11 | 55 | 10.50 | 0.93 | false |
| `hub_1_to_plant` | hub_1 → plant_1 | ROAD | 300 | 2 | 120 | 3.00 | 0.98 | false |
| `hub_2_to_plant` | hub_2 → plant_1 | ROAD | 350 | 3 | 100 | 3.50 | 0.97 | false |
| `supplier_to_plant_air` | supplier_1 → plant_1 | AIR | 6200 | 2 | 40 | 40.00 | 0.99 | **true** |

Every edge shared with Standard (`supplier_to_primary_port`, `supplier_to_alternative_port`, the primary/alternative-to-hub legs, `supplier_to_plant_air`) keeps Standard's exact values, renamed only where a second hub required disambiguating the edge id (`primary_port_to_hub_1`, `alternative_port_to_hub_1`, `hub_1_to_plant` are Standard's `primary_port_to_hub`, `alternative_port_to_hub`, `hub_to_plant` respectively, under new ids since Extended has more than one hub). `port_alternative` now reaches **both** hubs, giving two genuinely distinct multi-hop alternatives to the primary route (`port_primary → hub_1` vs. `port_tertiary → hub_2`), plus a third partial alternative through the shared `port_alternative` node.

**Default route:** in every tier, the replenishment plan's `initial_route_edge_ids` is the "primary" path — `supplier_to_primary_port → primary_port_to_hub(_1) → hub_to_plant` (`hub_1_to_plant` in Extended). The additional infrastructure in Standard and Extended only matters once that primary path is disrupted and candidate-route enumeration (V1 §5) considers alternatives — which is exactly the mechanism V1 already built and does not change.

### V2.3.2 Extended shock taxonomy

V1's six shock types (`domain/events.py`'s `ShockType` enum) are unchanged and keep working exactly as built. V2 adds three new shock types and one new target kind:

```text
ShockType additions:
    DEMAND_SPIKE
    DEMAND_DROP
    SUPPLIER_CAPACITY_REDUCTION

TargetType addition:
    DEMAND
```

("Supplier closure" needs no new shock type — it is `NODE_CLOSURE` targeting a `SUPPLIER`-type node, which V1's enum already supports. What changes is that `release_shipments` must now handle a closed or under-capacity supplier gracefully instead of raising an invariant error — see V2.3.7.)

**`DEMAND_SPIKE` / `DEMAND_DROP`** (`target_type: DEMAND`, `target_id` = the demand process's `destination_node_id`, e.g. `plant_1`): while active, multiplies the demand process's mean and its clamp bounds by a new `demand_multiplier` field (`NonNegativeFloat`, default `1.0`) — `DEMAND_SPIKE` scenarios use a multiplier `> 1.0`, `DEMAND_DROP` scenarios use one `< 1.0`. This is a **generation-time** effect, not a runtime one (V2.3.6): demand events are pre-generated once per replication exactly as in V1, so a demand shock's effect is baked into the generated event tape when it is built, not re-evaluated day by day during simulation. No new field is reused for this — `capacity_multiplier` already has an established meaning (departure/processing throughput) and reusing it for demand would be a silent, confusing overload, so `demand_multiplier` is deliberately a new, separate field.

**`SUPPLIER_CAPACITY_REDUCTION`** (`target_type: NODE`, `target_id` must be a `SUPPLIER`-type node): reduces the supplier's *release* capacity, using the existing `capacity_multiplier` field (this reuse is deliberate and safe: V1's `NODE_CAPACITY_REDUCTION` only ever touches `processing_capacity_multiplier`, a different operational dimension, so there is no collision). This requires one new field on `OperationalNodeState` (`source_capacity_multiplier: float = 1.0`, V2.4) and one new branch in `apply_shock_operational_state` — otherwise identical in shape to how `NODE_CAPACITY_REDUCTION` already works.

### V2.3.3 Shock realization: templates and distributions

This is the mechanism that answers the audit's core finding: in V1, a shock's start day, end day, and information day are fixed values written directly into the scenario YAML, identical in every replication. In V2, a scenario defines **shock templates** — the same fields as V1's `Shock`, plus distribution parameters — and one concrete `Shock` per template is **sampled once per replication**, before either policy makes a single decision, using the same deterministic, auditable seeding V1 already uses for demand and ordinary delays (V1 §5).

**New template fields** (replacing V1's fixed `physical_start_day` / `physical_end_day` / `information_day` in the scenario schema — the realized `Shock` domain object keeps those exact three fields unchanged, per V2.3.6):

| Field | Type | Meaning |
|---|---:|---|
| `planned_start_day` | `int` | Nominal/anchor start day — same role V1's `physical_start_day` played |
| `start_day_jitter_days` | `NonNegativeInt`, default `0` | Realized start day = `planned_start_day + Uniform{-J, ..., +J}` (discrete, inclusive). `0` reproduces V1's fixed-start-day behavior exactly. |
| `minimum_duration_days` | `PositiveInt` | Hard floor on realized duration |
| `duration_mean_days` | `PositiveFloat` | Mean of the duration distribution |
| `duration_std_days` | `NonNegativeFloat`, default `0` | Std. dev. of the duration distribution. `0` reproduces V1's fixed-duration behavior exactly. |
| `maximum_duration_days` | `PositiveInt` | Hard ceiling on realized duration |
| `max_information_delay_days` | `NonNegativeInt`, default `0` | Realized delay = discrete `Uniform{0, ..., max_information_delay_days}`. `0` reproduces V1's "complete and immediate" information behavior exactly. |

**Realization formulas**, computed once per replication per template, in this fixed order (matching V1 §5's "draw order within a stream is deterministic" convention):

1. `start_day_jitter = Uniform{-start_day_jitter_days, ..., +start_day_jitter_days}` → `physical_start_day = planned_start_day + start_day_jitter`
2. `duration_days = round(TruncatedNormal(duration_mean_days, duration_std_days, min=minimum_duration_days, max=maximum_duration_days))` (identical distribution family and clamping convention to V1's demand generation, V1 §5) → `physical_end_day = physical_start_day + duration_days - 1`
3. `information_delay = Uniform{0, ..., max_information_delay_days}` → `information_day = min(physical_start_day + information_delay, physical_end_day)` — clamped so information can never arrive *after* the disruption it describes has already ended; a policy may still learn about a disruption on the day it ends, but never learn about it only in retrospect.

The three draws for one template are always made in that order (start day, then duration, then information delay), and templates within one scenario are realized in ascending `shock_id` order — both fixed, so a given `(base_seed, replication)` pair always reproduces the identical realized shocks, exactly as V1's existing reproducibility guarantee (V1 §9) requires.

**Where this happens, and why the simulation engine does not change:** realization happens entirely inside `experiments/event_tape.py`, before the event tape is handed to `simulation/engine.py`. The output of realization is a tuple of ordinary `domain.events.Shock` objects — the exact same frozen dataclass V1 already has, with concrete `physical_start_day` / `physical_end_day` / `information_day` values. `simulation/transition.py`'s `apply_shock_operational_state` and every other consumer of `Shock` objects need **zero changes** for node/edge shock types: they already only ever look at the realized, concrete values. Only `DEMAND_SPIKE`/`DEMAND_DROP` need a second, narrow change to `generate_demand_events` (V2.3.6), since demand is pre-generated rather than evaluated at runtime.

**A degenerate window is a configuration error, not a runtime concern:** if `start_day_jitter_days` and `duration_std_days` are both `0` and `max_information_delay_days` is `0`, realization always reproduces the exact fixed values a V1-style scenario would have used. V1's existing `port_closure.yaml`, `port_partial_capacity.yaml`, and `port_extended_closure.yaml` remain valid V2 templates unchanged, simply with all-zero uncertainty fields — nothing forces existing V1 scenarios to be rewritten.

### V2.3.4 Compound and cascading disruption events

A scenario may group multiple shock templates under a shared `event_group_id` (`str | None`, default `None`). Templates sharing a group id represent effects of **one underlying event** (a regional storm, a geopolitical closure, a single supplier's plant fire cascading into a capacity shortfall) and share **one** realized `start_day_jitter` draw — all members start their jitter from the same random offset — while each member still draws its **own** `duration_days` and `information_delay` independently, representing that correlated events start together but each affected asset recovers, and is disclosed, on its own timeline. Templates with `event_group_id: null` are realized fully independently, exactly as V2.3.3 describes for a single template — this is the default, so ungrouped scenarios (including all of V1's existing ones, ported forward) need no changes.

Realization order for a scenario with groups: process `event_group_id`s in ascending lexicographic order (treating `null` as sorting last, as one group per ungrouped template); for a real group, draw the one shared `start_day_jitter` first, then realize each member template's `duration_days` and `information_delay` in ascending `shock_id` order within the group. This keeps the whole scenario's realization fully deterministic and auditable, consistent with V2.3.3's single-template ordering rule.

### V2.3.5 Release schedule realization

V1's replenishment plan releases exactly `shipment_quantity` units every `release_every_days` days, with zero variation, ever — the audit identified this as the single largest contributor to the 0%/100% win rates, since the set of shipments any disruption catches is otherwise fully predictable. V2 makes the **quantity** stochastic; it deliberately leaves the **timing** fixed.

**Why timing stays fixed and quantity does not:** randomizing release timing would ripple into due-date computation, shipment-id generation (V1 §3's `shipment_<release_day>_<sequence>` scheme assumes one release per calendar day), and the deterministic-ordering rules used throughout decision-triggering and departure allocation (V1 §3) — a much larger, riskier change for a benefit ("which day" varies) that timing jitter on the *disruption* (V2.3.3) already delivers more directly and more safely. Quantity variation is additive, narrow, and touches only the release-event generation formula and one new feasibility clamp.

**New replenishment plan template fields** (config-level, alongside V1's existing `shipment_quantity`, which becomes the distribution's mean):

| Field | Type | Meaning |
|---|---:|---|
| `shipment_quantity_mean` | `PositiveFloat` | Replaces V1's fixed `shipment_quantity` as the distribution's mean |
| `shipment_quantity_std` | `NonNegativeFloat`, default `0` | Std. dev. `0` reproduces V1's fixed-quantity behavior exactly |
| `minimum_shipment_quantity` | `PositiveInt` | Hard floor |
| `maximum_shipment_quantity` | `PositiveInt` | Hard ceiling |

**Realized quantity**, drawn once per scheduled release event, in release-day order: `quantity = round(TruncatedNormal(shipment_quantity_mean, shipment_quantity_std, min=minimum_shipment_quantity, max=maximum_shipment_quantity))` — identical distribution family and clamping convention to V1's demand generation (V1 §5) and V2.3.3's duration realization, for consistency.

**Feasibility clamp:** V1 §3 requires "every shipment quantity must fit the static capacity of every edge in its configured initial route" — a config-load-time guarantee in V1, since quantity was fixed. With quantity now random, this becomes a **generation-time** clamp instead: after drawing a raw quantity, it is further clamped to `min(raw_quantity, minimum static capacity across every edge and every node's processing_capacity on the initial route)`, deterministically, before the release event is finalized. This guarantees every generated `ShipmentReleaseEvent` remains structurally feasible on its initial route, exactly preserving V1's existing invariant, without requiring a config-time check that quantity can no longer statically satisfy.

### V2.3.6 Updated randomness streams

V1 §5 requires exactly two deterministic streams per replication: `demand` and `edge_delays`. V2 adds two more, derived with the identical mechanism (`sha256(f"{replication_seed}:{stream_name}")`, first eight bytes as an unsigned integer, per V1 §5):

```text
Required streams (V2):
    demand              (V1, unchanged)
    edge_delays         (V1, unchanged)
    shock_realization   (V2.3.3, V2.3.4 -- start-day jitter, duration, information delay)
    release_quantity    (V2.3.5 -- shipment quantity)
```

Draw order within each new stream follows the same style of rule V1 already uses for its two streams: `shock_realization` draws proceed in the scenario's fixed template/group order (V2.3.3–V2.3.4); `release_quantity` draws proceed in ascending scheduled-release-day order. As in V1, every draw happens once, up front, before either policy makes a single decision — nothing about V2's new randomness is drawn lazily or on demand during the day loop.

**Demand shocks and the demand stream:** `DEMAND_SPIKE`/`DEMAND_DROP` do not add a new draw to the `demand` stream — they change the *parameters* (`generate_demand_events` reads the realized shocks and, for any day inside an active demand shock's realized window, uses `mean_daily_demand * demand_multiplier` and equivalently scaled clamp bounds in place of the base values) but the random number consumed from the `demand` stream's `rng.gauss(...)` call is drawn exactly once per day exactly as in V1 — only what it is drawn *around* changes. This keeps the `demand` stream's draw count and draw order byte-identical to V1 regardless of whether a demand shock is present, which matters for V2.3.7's undisrupted-counterfactual rule below.

**Undisrupted counterfactual, extended:** V1's undisrupted tape is built by stripping only the scenario's shocks from an otherwise-identical disrupted tape (V1 §5, "the undisrupted tape is a copy with the designed shocks removed"). This is unchanged in V2 and now also applies uniformly across every new shock type: removing a realized `DEMAND_SPIKE`/`DEMAND_DROP`/`SUPPLIER_CAPACITY_REDUCTION`/compound-group shock from the disrupted tape and regenerating demand for that tape's days without it produces the fair "what if this hadn't happened" counterfactual, exactly as removing a `NODE_CLOSURE` already does today.

### V2.3.7 Daily event order: the one required change

V1 §7's thirteen-step daily order (reveal information → apply shocks → process arrivals → release shipments → fulfil demand → build/resolve decisions → allocate departures → charge costs → record metrics) is **unchanged in sequence.** Demand-shock effects are already fully baked into step 6 by generation time (V2.3.6), so step 6 itself needs no runtime awareness of shocks, exactly as before. Supplier capacity shocks are handled by step 3 (`apply_shock_operational_state` gains one new branch, V2.3.2) exactly as every other node/edge shock already is.

The one required change is to **step 5, shipment release**, and it exists specifically to make `SUPPLIER_CLOSURE` (`NODE_CLOSURE` targeting a supplier) and `SUPPLIER_CAPACITY_REDUCTION` usable at all: V1's `release_shipments` raises `SimulationInvariantError` if a scheduled release's origin is unavailable, over capacity, or would overflow storage, because V1's baseline configuration is designed to guarantee this never happens (V1 §7 step 5). V2 introduces supplier-side shocks specifically to make it happen on purpose, so this can no longer be treated as an invariant violation.

**New behavior:** `SimulationState` gains one new field, `pending_releases: list[ShipmentReleaseEvent]` (default empty). Step 5 becomes:

1. Attempt every event currently in `pending_releases`, in ascending `shipment_id` order (i.e., oldest-scheduled first — a first-committed-first-served rule), removing each from `pending_releases` on success.
2. Attempt every event newly scheduled for today, in the existing V1 §3 order (ascending `shipment_id`).
3. For either group, if the origin node is unavailable, over its effective `source_capacity` (V2.3.2's `source_capacity_multiplier`, floored exactly as every other effective-capacity calculation in V1 §2), or would overflow storage: **append the event to `pending_releases` and continue** — this is no longer a `SimulationInvariantError`.

`due_day` is unaffected by deferral: it was already computed once, at event-tape generation time, from the shipment's *originally scheduled* release day (V1's existing `ShipmentReleaseEvent.due_day = day + due_offset_days`), not from whenever `release_shipments` actually succeeds — so a supplier closure that delays a release automatically and correctly produces a later, costlier delivery relative to a due date the shipment was always going to be held to. No change to due-date computation is needed; this behavior already falls out of V1's existing design.

If a pending release still hasn't succeeded by the final drain day, it simply never entered the system — `total_released` (V1's product-balance invariant, V1 §7) only ever counts shipments that actually released, so the balance assertion remains correct without modification.

## V2.4 Domain model deltas

Every V2 mechanic above reduces to four small, additive changes to already-built dataclasses — nothing existing is renamed, removed, or given new required fields that would break a V1 caller:

```python
# domain/state.py
@dataclass(slots=True)
class OperationalNodeState:
    available: bool = True
    processing_capacity_multiplier: float = 1.0
    source_capacity_multiplier: float = 1.0          # NEW (V2.3.2)

@dataclass(slots=True)
class SimulationState:
    ...                                                 # every V1 field, unchanged
    pending_releases: list[ShipmentReleaseEvent] = field(default_factory=list)  # NEW (V2.3.7)
```

```python
# domain/events.py
class ShockType(Enum):
    ...                                                 # every V1 value, unchanged
    DEMAND_SPIKE = "DEMAND_SPIKE"                        # NEW (V2.3.2)
    DEMAND_DROP = "DEMAND_DROP"                          # NEW (V2.3.2)
    SUPPLIER_CAPACITY_REDUCTION = "SUPPLIER_CAPACITY_REDUCTION"  # NEW (V2.3.2)

class TargetType(Enum):
    NODE = "NODE"                                        # V1, unchanged
    EDGE = "EDGE"                                         # V1, unchanged
    DEMAND = "DEMAND"                                     # NEW (V2.3.2)

@dataclass(frozen=True, slots=True)
class Shock:
    ...                                                 # every V1 field, unchanged
    demand_multiplier: float = 1.0                       # NEW (V2.3.2); unused (stays 1.0) for every shock type except DEMAND_SPIKE/DEMAND_DROP, per V1 §7's existing "unused multiplier fields remain 1.0" convention
```

No change to `Node`, `Edge`, `Product`, `Route`, `DecisionAction`, `ValidationResult`, `ShipmentStatus`, `CostCounters`, `ServiceCounters`, `DailyMetrics`, or `SimulationResult` — every one of V1's cost formulas, validation codes, and output shapes is untouched by V2.

## V2.5 Configuration contracts

### `configs/scenarios/*.yaml`: shock templates

`ShockConfig` (`data_io/loaders.py`) gains V2.3.3's distribution fields and V2.3.4's grouping field, replacing the three V1 fixed-day fields. V1's `physical_start_day`/`physical_end_day`/`information_day` are removed from the *config* schema (they remain exactly as-is on the realized, domain-level `Shock` — V2.4 — since that object is built by realization, not loaded from YAML):

```yaml
schema_version: 1
scenario_id: demand_spike_before_peak_season
description: >
  A realistic demand-side shock: a temporary 40% demand spike with an
  uncertain, jittered start day and a duration that isn't known in advance.

shocks:
  - shock_id: seasonal_demand_spike
    shock_type: DEMAND_SPIKE
    target_type: DEMAND
    target_id: plant_1
    planned_start_day: 25
    start_day_jitter_days: 3
    minimum_duration_days: 5
    duration_mean_days: 10
    duration_std_days: 3
    maximum_duration_days: 18
    max_information_delay_days: 2
    demand_multiplier: 1.4
    event_group_id: null
```

A **supplier-side** example, showing `SUPPLIER_CAPACITY_REDUCTION` and the reuse of `capacity_multiplier`:

```yaml
schema_version: 1
scenario_id: supplier_capacity_shortfall
description: Supplier's own release capacity is cut by 60% for roughly two weeks, with genuine timing and duration uncertainty.

shocks:
  - shock_id: supplier_output_shortfall
    shock_type: SUPPLIER_CAPACITY_REDUCTION
    target_type: NODE
    target_id: supplier_1
    planned_start_day: 21
    start_day_jitter_days: 4
    minimum_duration_days: 7
    duration_mean_days: 14
    duration_std_days: 4
    maximum_duration_days: 25
    max_information_delay_days: 3
    capacity_multiplier: 0.4
    event_group_id: null
```

A **compound/cascading** example, showing `event_group_id` correlating two shocks of different types under one root cause:

```yaml
schema_version: 1
scenario_id: regional_disruption_event
description: >
  One regional event with two correlated effects: the alternate port closes,
  and the supplier's output is simultaneously reduced by transport
  disruption in the same region. Both start together; each recovers and is
  disclosed on its own timeline.

shocks:
  - shock_id: regional_port_closure
    shock_type: EDGE_CLOSURE
    target_type: EDGE
    target_id: alternative_port_to_hub
    planned_start_day: 22
    start_day_jitter_days: 2
    minimum_duration_days: 5
    duration_mean_days: 9
    duration_std_days: 2
    maximum_duration_days: 15
    max_information_delay_days: 1
    event_group_id: regional_event_2026_a

  - shock_id: regional_supplier_shortfall
    shock_type: SUPPLIER_CAPACITY_REDUCTION
    target_type: NODE
    target_id: supplier_1
    planned_start_day: 22
    start_day_jitter_days: 2
    minimum_duration_days: 4
    duration_mean_days: 7
    duration_std_days: 2
    maximum_duration_days: 12
    max_information_delay_days: 3
    capacity_multiplier: 0.65
    event_group_id: regional_event_2026_a
```

(`regional_port_closure` and `regional_supplier_shortfall` share one `start_day_jitter` draw per V2.3.4, but realize independent durations and information delays.)

**Updated validation** (`ResolvedConfig`, `data_io/loaders.py`): V1's rule "every shock starts after warm-up" (V1 §6) becomes a worst-case check against jitter: `planned_start_day - start_day_jitter_days > warmup_days` must hold for every template, so that **no possible realization** of the shock can start during or before warm-up. Every other V1 cross-reference validation (target exists in the network, no duplicate `shock_id`s, etc.) is unchanged in kind, just applied to templates instead of fixed shocks.

### `configs/networks/*.yaml`: replenishment plan

`ReplenishmentPlanConfig` gains V2.3.5's quantity-distribution fields in place of the fixed `shipment_quantity`:

```yaml
replenishment_plan:
  product_id: component_a
  origin_node_id: supplier_1
  destination_node_id: plant_1
  first_release_day: 1
  release_every_days: 1
  shipment_quantity_mean: 40
  shipment_quantity_std: 6
  minimum_shipment_quantity: 20
  maximum_shipment_quantity: 55
  due_offset_days: 15
  initial_route_edge_ids:
    - supplier_to_primary_port
    - primary_port_to_hub
    - hub_to_plant
```

Setting `shipment_quantity_std: 0` reproduces V1's fixed-quantity behavior exactly (`shipment_quantity_mean` takes the role of V1's `shipment_quantity`).

### `configs/experiments/*.yaml`: the grid is config, not a new mechanism

**V2 introduces no new experiment-orchestration abstraction.** The 3×3 (topology × severity) grid (V2.8) is realized as **nine separate experiment config files**, each in exactly V1's existing `ExperimentConfig` shape (V1 §6) — only `network_config` and `scenario_config` differ between them — each run independently via the already-built `python -m supply_chain_simulator.cli run --config ...`. This is the exact operational pattern already validated for V1's own three-profile run (`baseline_comparison.yaml` / `light_disruption_comparison.yaml` / `heavy_disruption_comparison.yaml`): nine files instead of three, nothing else different. `cli.py`, `experiments/runner.py`, and `data_io/writers.py` require **no changes** to support the grid.

## V2.6 Repository structure additions

Purely additive to V1 §4's tree — no existing file moves, and V1's `baseline_network.yaml` / `port_closure.yaml` / `port_partial_capacity.yaml` / `port_extended_closure.yaml` / `baseline_comparison.yaml` / `light_disruption_comparison.yaml` / `heavy_disruption_comparison.yaml` stay exactly where they are, exactly as they are:

```text
configs/
├── networks/
│   ├── baseline_network.yaml          (V1, unchanged — the Standard topology tier)
│   ├── topology_compact.yaml          (NEW — V2.3.1)
│   └── topology_extended.yaml         (NEW — V2.3.1)
├── scenarios/
│   ├── port_closure.yaml              (V1, unchanged)
│   ├── port_partial_capacity.yaml     (V1, unchanged)
│   ├── port_extended_closure.yaml     (V1, unchanged)
│   ├── demand_spike_before_peak_season.yaml     (NEW — example, V2.5)
│   ├── supplier_capacity_shortfall.yaml         (NEW — example, V2.5)
│   └── regional_disruption_event.yaml           (NEW — example, V2.5)
└── experiments/
    ├── baseline_comparison.yaml       (V1, unchanged — this is the Standard x Medium grid cell)
    ├── light_disruption_comparison.yaml    (V1, unchanged — Standard x Light)
    ├── heavy_disruption_comparison.yaml    (V1, unchanged — Standard x Heavy)
    └── (six more, one per remaining grid cell — V2.8)
```

No new top-level architectural folder is introduced (`src/supply_chain_simulator`'s package boundaries are unchanged, per V1 §4's "do not create additional architectural folders" and V2.2's "what must not change"). `analysis/` (built during V1's results-visualization work, outside the core package) gains new capability, not a new folder — V2.7.

## V2.7 File-by-file implementation contract deltas

This section lists every existing V1 file that changes for V2, and exactly what changes in it. Any file not listed here is unchanged. As in V1 §5, "unchanged" means unchanged in public contract and behavior, not merely untouched by an editor.

### `domain/state.py`

- `OperationalNodeState` gains `source_capacity_multiplier: float = 1.0` (V2.4).
- `SimulationState` gains `pending_releases: list[ShipmentReleaseEvent]` (V2.4), initialized empty at day-0 construction (V1 §7).

### `domain/events.py`

- `ShockType` gains `DEMAND_SPIKE`, `DEMAND_DROP`, `SUPPLIER_CAPACITY_REDUCTION` (V2.3.2).
- `TargetType` gains `DEMAND` (V2.3.2).
- `Shock` gains `demand_multiplier: float = 1.0` and `event_group_id: str | None = None` (V2.3.2, V2.3.4). All other `Shock` fields are unchanged — a realized `Shock` still carries concrete `physical_start_day`/`physical_end_day`/`information_day` integers; only the *config* that produces it (V2.5) now expresses these as distributions.

### `experiments/event_tape.py`

The only file that gains genuinely new logic, since V1 §5's "all randomness drawn upfront, before any policy decision" is preserved by doing every new draw here, before `simulation/engine.py` ever runs.

New required functions, added alongside V1 §5's existing ones:

```python
realize_shock(template, stream_seed, shipment...) -> Shock
realize_shock_group(templates, stream_seed) -> list[Shock]
realize_release_quantity(mean, std, minimum, maximum, stream_seed, day, sequence) -> int
```

- `realize_shock` draws `start_day_jitter`, `duration`, and `information_delay` per V2.3.3's exact formulas and fixed order (jitter, then duration, then information delay), from the new `shock_realization` stream (V2.3.6).
- `realize_shock_group` implements V2.3.4: draws one shared jitter value for all templates sharing an `event_group_id`, then calls the per-member duration/information-delay draws independently, in `shock_id` ascending order.
- `realize_release_quantity` implements V2.3.5's TruncatedNormal-and-clamp pattern, called once per scheduled release in `generate_shipment_release_events`, from the new `release_quantity` stream, in day-ascending / sequence-ascending order (matching V1 §5's existing draw-order discipline).
- `generate_demand_events` gains one additional argument: the realized list of `DEMAND`-type shocks active on each day, used only to select which multiplier (if any) scales that day's `mean_daily_demand` before the existing `round(rng.gauss(...))` clamp step. This changes demand's *parameters* on shocked days; it adds no new draw and does not change the `demand` stream's draw count or order (V2.3.6) — the undisrupted tape still reuses the exact same `demand` stream draws with the multiplier removed.
- `build_disrupted_event_tape` and `build_undisrupted_event_tape` both gain a shock-realization step before demand generation (since realized `DEMAND`-type shocks feed into demand generation) and otherwise keep V1 §5's structure; the undisrupted tape's existing rule ("remove only the designed shock, keep every other draw identical") extends unchanged to every new shock type, per V2.3.6.

### `simulation/transition.py`

- `release_shipments` (V1 §5): where V1 raises `SimulationInvariantError` when a scheduled release exceeds source availability/capacity/storage, V2 instead appends the release event to `state.pending_releases` and continues (V2.3.7). `SimulationInvariantError` is still raised for genuine invariant violations elsewhere in the file; this is the one narrow, deliberate exception, justified by SUPPLIER_CLOSURE/SUPPLIER_CAPACITY_REDUCTION making an unreleasable day a normal, expected occurrence rather than a bug.
- `apply_shock_operational_state` gains one new branch: `SUPPLIER_CAPACITY_REDUCTION` sets `source_capacity_multiplier` on the target node's `OperationalNodeState` (mirroring the existing `NODE_CAPACITY_REDUCTION` → `processing_capacity_multiplier` branch exactly). `DEMAND_SPIKE`/`DEMAND_DROP` shocks are **not** handled here at all — per V2.3.3, demand shocks are realized into event-tape parameters at build time, never into runtime operational state, so this function's demand-related surface area is unchanged.
- Step 5 of the daily loop (`release_shipments`'s caller) changes per V2.3.7: attempt every entry in `state.pending_releases` (in shipment-ID order) before today's newly scheduled releases, each subject to the same availability/capacity/storage check; a still-infeasible entry stays in `pending_releases` for the next day.

### `data_io/loaders.py`

- `ScenarioConfig`'s nested shock schema replaces V1's three fixed-day fields with V2.5's distribution fields and `event_group_id`.
- `NetworkConfig`'s replenishment-plan schema replaces V1's fixed `shipment_quantity` with V2.5's mean/std/min/max fields.
- The warm-up-vs-shock-start validator changes from checking `physical_start_day > warmup_days` to checking `planned_start_day - start_day_jitter_days > warmup_days` (V2.5).
- No change to path resolution, secret handling, or any other V1 §5 rule.

### `analysis/plot_results.py`

Gains the ability to accept a 2D (topology × severity) set of experiment output directories — e.g. `--experiment <dir> --topology Compact --severity Light` repeated for each grid cell — and produce the "disruption robustness heatmap" plot type that V1's `VISUAL_REPORTING_PLAN_ESSENTIAL.md` specified but that V1's flat 3-profile grid couldn't populate (only one topology existed). The existing per-experiment and cross-severity plots (built for V1) are unchanged; the heatmap is additive. Because `analysis/` sits outside `src/supply_chain_simulator/` and outside mypy/pytest scope (V1 §4, confirmed unchanged by V2.2), this addition carries no new testing or type-checking obligation beyond what V1 already established for this script.

### Everything else

`simulation/routing.py`, `simulation/costs.py`, `simulation/engine.py`, `decisions/observation.py`, `decisions/validator.py`, `policies/*.py`, `integrations/llm_client.py`, `data_io/writers.py`, `cli.py` — **no changes**. V2's new randomness and new shock types resolve entirely to ordinary, already-supported domain objects (a realized `Shock`, a `ShipmentReleaseEvent` with a different quantity, a node's `source_capacity_multiplier`) before reaching any of these files; none of them can tell a V2-realized `Shock` apart from a V1 fixed one.

## V2.8 Experimental design: the topology × severity grid

### V2.8.1 The grid

V2 crosses two independent axes, per Yassine's confirmed direction (`Independent, crossed axes`):

|  | Light | Medium | Heavy |
|---|---|---|---|
| **Compact** | Compact × Light | Compact × Medium | Compact × Heavy |
| **Standard** | Standard × Light | Standard × Medium (= V1's `baseline_comparison.yaml`, unmodified) | Standard × Heavy |
| **Extended** | Extended × Light | Extended × Medium | Extended × Heavy |

Nine cells. Topology (Compact/Standard/Extended, V2.3.1) and severity (Light/Medium/Heavy, already built and validated in V1 as `port_partial_capacity.yaml`/`port_closure.yaml`/`port_extended_closure.yaml`) are deliberately named with disjoint vocabulary so a cell label is unambiguous without a legend.

Each cell is one `configs/experiments/*.yaml` file (V2.5) — nine files total, following V1's existing naming convention, e.g. `compact_light_comparison.yaml`, `extended_heavy_comparison.yaml`. `Standard × Medium` is exactly V1's existing `baseline_comparison.yaml` and is not duplicated.

Severity scenario files (`port_partial_capacity.yaml` etc.) are reused across topology tiers unchanged where their `target_id` exists in every tier (`port_primary` and `supplier_1` exist in Compact, Standard, and Extended by construction, V2.3.1); Extended's richer topology additionally supports the demand-side and supplier-side example scenarios from V2.5, which may be substituted into any tier's severity column at Yassine's discretion when the grid is populated — the grid's structure does not require every cell to use an identical shock *type*, only a comparable shock *severity*, consistent with V2.1's mission delta.

### V2.8.2 Replication count

V1 used 100 replications per single profile (300 total) within a $16.71 budget, calibrated empirically (V1's real 3-replication-per-profile calibration run) at roughly $0.05–$0.06 per replication at full horizon/drain. Nine cells at the same per-replication cost and the same total budget implies roughly 33 replications per cell — a real reduction in per-cell statistical power versus V1's single-profile runs.

This is a cost/power tradeoff, not a scientific rule, so it is **not locked** the way V1's fixed values were. The recommended default is:

```text
replications: 50   # per grid cell, 450 total
```

chosen as a middle point that keeps total spend near V1's original budget scale (450 × ~$0.055 ≈ $25) while giving each cell meaningfully more power than a bare 33. Yassine must explicitly confirm the actual `replications` value and run the same calibration procedure V1 used (a 3-replication real smoke at full horizon/drain per cell, or one representative cell if cells are judged cost-homogeneous) before committing to the full grid — this mirrors V1 §10's own "no coding agent may silently pick a scientific parameter" discipline, extended to V2's new grid.

## V2.9 Testing contract additions

All V1 tests (V1 §9) continue to pass unmodified — V2 adds tests, it does not rewrite V1's. New tests, following V1's existing per-module file convention:

### `tests/unit/test_event_tape.py` (additions)

- `realize_shock` is deterministic for a fixed stream seed and produces `start_day_jitter`/`duration`/`information_delay` draws in the fixed order specified by V2.3.3;
- a template with every uncertainty field zeroed (`start_day_jitter_days=0`, `duration_std_days=0`, `max_information_delay_days=0`) reproduces V1's exact fixed-shock behavior bit-for-bit, proving V2 is a strict superset of V1;
- `realize_shock_group` draws one shared jitter value across all members of an `event_group_id` and independent duration/information-delay per member;
- `realize_release_quantity` is deterministic, clamped to `[minimum_shipment_quantity, maximum_shipment_quantity]`, and reproduces V1's fixed quantity when `shipment_quantity_std=0`;
- the new `shock_realization` and `release_quantity` streams are distinct from `demand` and `edge_delays` and from each other (V2.3.6);
- demand-shock realization changes `generate_demand_events`'s *parameters* on shocked days without changing the `demand` stream's draw count or order versus an otherwise-identical unshocked run.

### `tests/unit/test_transition.py` (additions)

- a release that fails the source availability/capacity/storage check is appended to `pending_releases` instead of raising `SimulationInvariantError`;
- a pending release is retried, in shipment-ID order, before the current day's new releases, and succeeds once the source recovers;
- `SUPPLIER_CAPACITY_REDUCTION` reduces `source_capacity_multiplier` and correctly throttles `release_shipments`, mirroring the existing `NODE_CAPACITY_REDUCTION` test for `processing_capacity_multiplier`;
- `DEMAND_SPIKE`/`DEMAND_DROP` shocks produce **no** operational-state change (asserting the negative, since V2.3.2 deliberately keeps demand shocks out of runtime state).

### `tests/unit/test_loaders.py` (additions)

- `topology_compact.yaml` and `topology_extended.yaml` load into the exact node/edge counts and values specified in V2.3.1;
- a shock template config with `planned_start_day - start_day_jitter_days <= warmup_days` fails loudly (the updated V2.5 validator);
- a shock template with every V1-style fixed field but no new distribution fields is rejected (config schema is not silently backward-compatible — V1's scenario files must be migrated to the new schema, per V2.2's explicit non-goal of a compatibility shim);
- the three example scenario configs from V2.5 (demand spike, supplier shortfall, compound event) load into the exact expected `ShockConfig` objects, including correct `event_group_id` grouping.

### New: `tests/unit/test_topology.py`

- each topology tier's `NetworkDefinition` has the exact node/edge count and connectivity from V2.3.1;
- every tier's default `initial_route_edge_ids` begins at the source and ends at the destination (reusing V1's existing route-continuity check, applied to all three tiers);
- Compact has no structurally valid non-emergency reroute path around a `port_primary` closure (proving the intended REROUTE-impossible property);
- Extended has at least two structurally distinct non-emergency reroute paths around a `port_primary` closure.

### `tests/integration/test_paired_experiment.py` (additions)

- a compound event's two shocks realize a shared start day but independent durations, and the undisrupted tape removes both while preserving every other draw, extending V1's existing "undisrupted tape differs only by shocks" assertion to grouped shocks;
- a full paired run on each topology tier completes and produces valid TCD/delta values, proving no topology-specific code path is missing.

### `tests/integration/test_reproducibility.py` (additions)

- identical config and seed reproduce identical realized shocks (start day, duration, information day) and identical realized release quantities, extending V1's existing event-tape reproducibility assertion.

## V2.10 Milestones for V2

Building on V1's completed Milestones 0–9 (V1 §9):

### Milestone 10 — Domain and config deltas

Implement V2.4's dataclass changes and V2.5's config schema changes. Done when: `topology_compact.yaml`/`topology_extended.yaml` and the three example scenario configs load into exact expected objects; a V1-shaped scenario file is rejected with a precise schema error (V2.2's explicit no-compatibility-shim decision); all `test_loaders.py` additions pass.

### Milestone 11 — Shock and release realization

Implement `experiments/event_tape.py`'s new functions (V2.7). Done when: `realize_shock`, `realize_shock_group`, and `realize_release_quantity` are deterministic and tested; an all-zero-uncertainty template reproduces V1 bit-for-bit; the new streams are proven distinct; `test_event_tape.py` additions pass.

### Milestone 12 — Deferred releases and new shock effects

Implement `simulation/transition.py`'s deferral logic and the `SUPPLIER_CAPACITY_REDUCTION` operational-state branch (V2.7). Done when: a source-infeasible release defers instead of raising; a deferred release retries and succeeds; `SUPPLIER_CAPACITY_REDUCTION` throttles releases; demand shocks are confirmed to touch no operational state; `test_transition.py` additions pass.

### Milestone 13 — Topology tiers

Build `topology_compact.yaml` and `topology_extended.yaml` exactly per V2.3.1's tables. Done when: `test_topology.py` passes in full, including the Compact-has-no-reroute and Extended-has-multiple-reroutes structural proofs.

### Milestone 14 — Grid experiment configs and end-to-end run

Write the nine `configs/experiments/*.yaml` grid cells (V2.8.1), run `validate-config` on all nine, then run a 3-replication real calibration per V2.8.2 to measure actual per-cell OpenAI spend at full horizon/drain before committing to the full grid. Done when: all nine configs validate; calibration produces a real, measured per-replication cost; `test_paired_experiment.py`'s per-tier full-run addition passes for at least one cell per topology tier.

### Milestone 15 — Analysis grid heatmap

Extend `analysis/plot_results.py` to accept the 2D grid and produce the disruption-robustness heatmap (V2.7). Done when: run against real Milestone-14 calibration output, the heatmap renders correctly with no errors, for at least a partial (non-9-cell) grid.

### Milestone 16 — Full V2 grid execution

Run the full, Yassine-confirmed `replications`-per-cell grid (V2.8.2) end to end, all nine cells, `fail_fast=true` per cell as in V1. Done when: all nine cells complete; `summary.json` exists for each; the cross-cell heatmap and V1's existing per-experiment plots (V1's `analysis/plot_results.py` base functionality, unchanged) both render on the real full-grid output.

Do not start a milestone whose required predecessor is failing, per V1 §9's unchanged discipline.

## V2.11 Acceptance criteria for V2

V2 is complete only when all are true, additively to V1 §9 (which remains fully binding — V2 must never regress a V1 acceptance criterion):

### Architecture

- every V2.6 file/folder addition exists; no unapproved architectural folder was introduced; V2.7's file list is exhaustive (no undocumented file changed).

### Simulation

- Compact, Standard, and Extended topologies all load and run full paired experiments without error;
- an all-zero-uncertainty V2 shock template reproduces the corresponding V1 fixed-shock run's costs and metrics bit-for-bit (the strict-superset proof from V2.9);
- deferred releases correctly resolve once source capacity/availability recovers, and never silently drop a scheduled shipment;
- `DEMAND_SPIKE`/`DEMAND_DROP`/`SUPPLIER_CAPACITY_REDUCTION` all produce measurable, correctly-directioned effects on demand/backlog or release throughput respectively;
- compound events realize correlated start days and independent durations/information delays, and the undisrupted counterfactual removes every member of the group.

### Fairness

- every new randomness source draws from its own named stream, upfront, before any policy decision, per V1 §5's unchanged fairness principle;
- the undisrupted tape still differs from the disrupted tape only in shocks (now including grouped and demand-side shocks) and `newly_known_shock_ids`;
- both policies still see identical observations built from identical shared state, unchanged by any V2 addition;
- no V2 change touches `simulation/routing.py`, `decisions/validator.py`, or the LLM's five approved tools (V2.2's explicit constraint), confirmed by the V2.7 file-change list being exhaustive.

### Grid and metrics

- all nine grid cells validate and, once Yassine confirms a replication count, run to completion;
- the disruption-robustness heatmap (deferred from V1) renders correctly across the full grid;
- V1's existing metrics (TCD, delta, win rates, CI) are calculated identically per cell, with no formula change.

### Quality

- all V1 tests plus all V2.9 additions pass; Ruff and Mypy pass for all touched `src/` files; coverage remains at least 90%; no real API call occurs in any automated test; no secret exists in repository or outputs — every quality bar from V1 §9 holds for V2's additions too.

## V2.12 Final locked decisions for Version 2

| Topic | Version 2 decision |
|---|---|
| Topology | Three tiers — Compact, Standard, Extended — crossed independently with severity (V2.3.1, V2.8.1) |
| Severity | Unchanged from V1 — Light, Medium, Heavy (V1's existing three scenario files) |
| Grid mechanism | Nine flat experiment config files; no new orchestration abstraction (V2.5) |
| Disruption duration | Uncertain — sampled TruncatedNormal per replication (V2.3.3) |
| Disruption start day | Jittered around a planned day per replication (V2.3.3) |
| Disruption information | Delayed by an uncertain number of days after physical start (V2.3.3) |
| Shipment release timing | Unchanged from V1 — fixed schedule (V2.3.5) |
| Shipment release quantity | Uncertain — sampled TruncatedNormal per release (V2.3.5) |
| New shock types | DEMAND_SPIKE, DEMAND_DROP, SUPPLIER_CAPACITY_REDUCTION (V2.3.2) |
| Demand shock mechanism | Realized into event-tape generation parameters, never into runtime operational state (V2.3.2, V2.3.3) |
| Supplier closure | Reuses existing NODE_CLOSURE targeting a SUPPLIER node; no new enum value (V2.3.2) |
| Compound events | Correlated start-day jitter via `event_group_id`; independent duration and information delay per member (V2.3.4) |
| Infeasible releases | Deferred via `pending_releases`, retried daily, never an invariant error (V2.3.7) |
| New randomness streams | `shock_realization`, `release_quantity`, added to V1's `demand`/`edge_delays` (V2.3.6) |
| V1 scenario file compatibility | Not preserved — V1-shaped scenario configs must be migrated to the new schema; no compatibility shim (V2.2, V2.9) |
| Action space, validator, LLM tools, cost model, daily event order (except step 5) | Unchanged from V1 (V2.2) |
| Replications per grid cell | Not locked — Yassine confirms after a real per-cell calibration run, recommended default 50 (V2.8.2) |
| Document structure | Two-part: V1 frozen as Part 1, V2 as an additive Part 2 (this file) |

No coding agent may change these decisions without explicit approval from Yassine.
