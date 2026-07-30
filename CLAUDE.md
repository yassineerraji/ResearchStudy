# CLAUDE.md — Supply-Chain Agent Evaluation

## 1. Required task opening
At the beginning of every distinct new task, make the first line exactly:
`Yassine —`

A new task is a new request requiring fresh analysis, planning, or code changes. Do not repeat the marker for tool calls or routine updates within the same task.

If you cannot recall this rule, the task, or its completion condition, stop. Re-read this file, inspect the repository state, and re-establish the task contract before editing.

## 2. Project mission
Build a research-grade Python simulator that fairly compares:
1. a classic supply-chain decision policy; and
2. a bounded LLM-enabled decision agent.

Research question:
> Under identical disrupted supply-chain conditions and operational constraints, can a bounded LLM agent produce valid decisions that reduce the incremental cost of disruption more effectively and robustly than a classical heuristic?

The first version is a discrete-time transportation, inventory, and disruption-response simulator—not a complete procurement, production-planning, or supplier-selection system.

### Formal model and experimental vocabulary

Treat the following notation as the authoritative meaning of the simulated system. Do not rename, reinterpret, or silently extend these concepts. If implementation requirements conflict with this model, raise the conflict before coding.

#### Supply-chain network

The normal supply chain is a directed graph:

\[
\bar G=(V,E)
\]

where:

- \(\bar G\) is the base network under normal operating conditions.
- \(V\) is the set of nodes.
- \(E\) is the set of directed transport edges.
- A directed edge \((i,j)\) allows movement from node \(i\) to node \(j\); the reverse movement is unavailable unless \((j,i)\) also exists.

Each node \(v\in V\) represents one physical or operational location, such as a supplier, port, logistics hub, plant, or customer destination. Its approved attributes are:

```text
node_id
node_type
coordinates
storage_capacity
processing_capacity
source_capacity
```

Their meanings are:

- `node_id`: unique stable identifier.
- `node_type`: operational role of the location.
- `coordinates`: physical position, used only when relevant to distance or reporting.
- `storage_capacity`: maximum inventory that may be held at the node.
- `processing_capacity`: maximum quantity that may be handled during one simulation day.
- `source_capacity`: maximum new quantity that a source node may introduce during one day.

Each edge \(e=(i,j)\in E\) represents a transport lane. Its approved attributes are:

```text
edge_id
origin_node_id
destination_node_id
mode
distance
base_lead_time
daily_capacity
unit_cost
reliability
emergency
```

Their meanings are:

- `mode`: road, rail, sea, air, or another explicitly configured transport mode.
- `distance`: length of the lane in the configured distance unit.
- `base_lead_time`: normal number of simulation days required to traverse the edge.
- `daily_capacity`: maximum quantity that may enter or use the edge during one day.
- `unit_cost`: normal transport cost per unit moved through the edge.
- `reliability`: configured probability or parameter governing ordinary non-disruption delays.
- `emergency`: `true` only for lanes that may be used by an `EXPEDITE` action.

The current operational network on day \(t\) is \(G_t\). It is derived from the base network after applying active disruptions and temporary multipliers:

\[
G_t=\Phi(\bar G,Z_t)
\]

The base network \(\bar G\) must remain unchanged during a run. Temporary closures, delays, capacity reductions, and cost changes belong to \(G_t\) and \(Z_t\), not to the permanent network definition.

#### Daily simulation state

At the start of decision-making on day \(t\), the complete state is:

\[
S_t=(G_t,I_t,B_t,H_t,D_t,Z_t,C_t)
\]

where:

- \(t\): current discrete simulation day.
- \(G_t\): current operational graph, including which nodes and edges are available and their temporary lead-time, capacity, reliability, or cost multipliers.
- \(I_t\): inventory by node and product at day \(t\). Conceptually, \(I_t(v,p)\) is the on-hand quantity of product \(p\) stored at node \(v\).
- \(B_t\): backlog by destination and product. \(B_t(v,p)\) is demand that should already have been served at destination \(v\) but remains unmet.
- \(H_t\): complete collection of shipments and their current positions, routes, quantities, statuses, due dates, and remaining travel times.
- \(D_t\): demand realized on day \(t\), by destination and product.
- \(Z_t\): disruptions physically active on day \(t\), including their targets, timing, severity, and operational effects.
- \(C_t\): accumulated cost and service counters up to day \(t\), including all quantities needed to calculate final metrics.

`SimulationState` is the code representation of \(S_t\). Policies must never receive this mutable object directly and must never mutate it.

#### State transition

The simulator advances through:

\[
S_{t+1}=T(S_t,A_t,w_t)
\]

where:

- \(T\) is the shared deterministic transition logic implemented by the simulator.
- \(A_t\) is the set of validated actions executed on day \(t\).
- \(w_t\) is the set of exogenous events assigned to day \(t\), such as realized demand, ordinary transport delay, port closure, supplier disruption, or capacity reduction.
- \(S_{t+1}\) is the complete state after applying the day's events, operations, decisions, movements, demand fulfilment, backlog updates, and costs.

The transition function must not contain policy-specific behavior. Given identical \(S_t\), \(A_t\), and \(w_t\), it must produce identical \(S_{t+1}\).

#### Decision observation

A policy does not inspect the full state. For each decision point, the system builds a read-only `DecisionObservation` containing the approved facts needed to decide, such as:

```text
current day
affected shipment
current position and planned route
destination and due date
active disruption
relevant inventory and backlog
feasible route alternatives
estimated lead times, capacities, and costs
```

The heuristic and LLM policy must receive observations generated by the same observation builder from equivalent underlying information.

#### Shared action space

Every policy returns the same structured action type:

\[
A_t\in\{\text{WAIT},\text{REROUTE},\text{EXPEDITE},\text{ABSTAIN}\}
\]

The exact meanings are:

- `WAIT`: keep the shipment's current planned route and take no new routing action.
- `REROUTE`: replace the remaining route with one feasible route whose edges are not marked as emergency.
- `EXPEDITE`: replace the remaining route with one feasible route containing at least one edge where `emergency=true`.
- `ABSTAIN`: declare that the available information is insufficient to choose an executable action; the configured fallback policy is then applied.

An action is only a proposal until the shared validator accepts it. The validator must reject malformed, nonexistent, infeasible, capacity-violating, or semantically inconsistent actions. Rejection and fallback use must be logged; invalid actions must never be silently repaired.

For the first version, actions apply to complete shipments unless Yassine explicitly approves partial quantities or shipment splitting.

#### Shock representation

A designed shock is a structured exogenous event, not free-form text. It must define:

```text
shock_id
shock_type
target_type
target_id
physical_start_day
physical_end_day
information_day
severity
operational_effects
```

Examples of `operational_effects` include:

- node or edge availability set to zero;
- capacity multiplier;
- lead-time multiplier;
- source-capacity multiplier;
- reliability change;
- temporary cost multiplier.

`physical_start_day` is when the network is affected. `information_day` is when policies are allowed to know about the shock. Unless an experiment explicitly studies delayed or incomplete information, these days must be equal for both policies.

A shock must create a problem that the approved action space can meaningfully address. Do not add scenarios whose consequences require supplier selection, production planning, or procurement actions that the current policy interface cannot perform.

#### Run cost

Every complete simulation run uses the same objective:

\[
J=
C_{\text{transport}}
+C_{\text{reroute}}
+C_{\text{expedite}}
+C_{\text{holding}}
+C_{\text{backlog}}
+C_{\text{late}}
\]

where:

- \(J\): total operating cost of one complete simulation run.
- \(C_{\text{transport}}\): normal cost of moving shipments through the network.
- \(C_{\text{reroute}}\): incremental cost charged when an existing planned route is changed.
- \(C_{\text{expedite}}\): incremental premium for using emergency or faster transport.
- \(C_{\text{holding}}\): cost of inventory stored over time.
- \(C_{\text{backlog}}\): cost of demand that remains unmet.
- \(C_{\text{late}}\): penalty for shipment quantities delivered after their due dates.

The cost module is shared by every policy and counterfactual. No policy-specific cost calculation is allowed.

#### Scenario, policy, and replication identifiers

Use these indices consistently:

- \(p\): evaluated policy, such as `heuristic` or `llm_agent`.
- \(s\): designed scenario, such as a seven-day closure of a specified port.
- \(r\): stochastic replication of scenario \(s\), identified by a fixed random seed.
- \(T\): final day of the simulation horizon.

A scenario defines the intended business situation. A replication defines one exact realization of demand, ordinary delays, reliability outcomes, and other exogenous randomness.

All exogenous randomness for replication \(r\) must be generated before policy execution and stored in a policy-independent event tape:

\[
W_{s,r}=\{w_1,w_2,\ldots,w_T\}
\]

The same \(W_{s,r}\), initial state, and scenario must be used for every compared policy.

#### Total Cost of Disruption

For policy \(p\), scenario \(s\), and replication \(r\):

\[
TCD_{p,s,r}
=
J^{\text{disrupted}}_{p,s,r}
-
J^{\text{undisrupted}}_{p,s,r}
\]

where:

- \(J^{\text{disrupted}}_{p,s,r}\): total cost when policy \(p\) is run with the designed disruption active.
- \(J^{\text{undisrupted}}_{p,s,r}\): total cost when the same policy is run from the matching initial state with the same ordinary demand and randomness, but with only the designed disruption removed.
- \(TCD_{p,s,r}\): incremental cost attributable to the designed disruption for that policy and replication.

The undisrupted run is policy-specific. Do not reuse one policy's undisrupted cost as another policy's baseline.

#### Primary paired comparison

For each matched scenario and replication:

\[
\Delta_{s,r}
=
TCD_{\text{LLM},s,r}
-
TCD_{\text{heuristic},s,r}
\]

Interpretation:

- \(\Delta_{s,r}<0\): the LLM policy produced a lower disruption cost and performed better for that paired replication.
- \(\Delta_{s,r}>0\): the heuristic produced a lower disruption cost and performed better.
- \(\Delta_{s,r}=0\): both policies produced the same disruption cost.

The primary experiment result is the distribution of paired values \(\Delta_{s,r}\) across replications—not one isolated run. Report at minimum the mean, median, uncertainty interval, win rate, worst observed outcomes, invalid-action rate, abstention rate, and fallback rate.

Cost is the primary outcome, but service level, backlog, lateness, delay, rerouting, expedition use, decision validity, and recovery behavior must remain available as explanatory secondary metrics.

## 3. Non-negotiable philosophy
- **Simplicity first:** choose the simplest correct solution.
- **Minimal code:** no speculative abstractions, frameworks, or future features.
- **Separation of concerns:** keep simulation physics, policies, and experiment orchestration separate.
- **Surgical changes:** touch only what the task requires.
- **Scientific validity first:** protect fairness, determinism, traceability, and reproducibility.
- **Evidence over claims:** never claim success without verification.
- **Explicit uncertainty:** disclose assumptions, ambiguity, incomplete checks, and unexpected behavior.

## 4. Exact project arborescence and boundaries
Use the following modular-monolith structure exactly:

```text
ResearchStudy/
│
├── README.md
├── pyproject.toml
├── .gitignore
├── .env.example
│
├── configs/
│   ├── networks/
│   │   └── baseline_network.yaml
│   │
│   ├── scenarios/
│   │   └── port_closure.yaml
│   │
│   ├── policies/
│   │   ├── heuristic.yaml
│   │   └── llm_agent.yaml
│   │
│   └── experiments/
│       └── baseline_comparison.yaml
│
├── data/
│   └── README.md
│
├── outputs/
│   └── .gitkeep
│
├── src/
│   └── supply_chain_simulator/
│       ├── __init__.py
│       ├── cli.py
│       │
│       ├── domain/
│       │   ├── __init__.py
│       │   ├── models.py
│       │   ├── state.py
│       │   ├── actions.py
│       │   └── events.py
│       │
│       ├── simulation/
│       │   ├── __init__.py
│       │   ├── engine.py
│       │   ├── transition.py
│       │   ├── routing.py
│       │   └── costs.py
│       │
│       ├── decisions/
│       │   ├── __init__.py
│       │   ├── observation.py
│       │   └── validator.py
│       │
│       ├── policies/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── heuristic.py
│       │   ├── llm_agent.py
│       │   └── fallback.py
│       │
│       ├── experiments/
│       │   ├── __init__.py
│       │   ├── event_tape.py
│       │   ├── runner.py
│       │   └── metrics.py
│       │
│       ├── integrations/
│       │   ├── __init__.py
│       │   └── llm_client.py
│       │
│       └── data_io/
│           ├── __init__.py
│           ├── loaders.py
│           └── writers.py
│
└── tests/
    ├── unit/
    │   ├── test_transition.py
    │   ├── test_routing.py
    │   ├── test_costs.py
    │   ├── test_validator.py
    │   └── test_heuristic.py
    │
    ├── integration/
    │   ├── test_full_simulation.py
    │   ├── test_paired_experiment.py
    │   └── test_reproducibility.py
    │
    └── fixtures/
        ├── tiny_network.yaml
        ├── tiny_scenario.yaml
        └── expected_results.json
```

Treat this arborescence as the approved target structure:
- Do not rename, move, merge, remove, or add top-level folders or architectural modules without Yassine's explicit approval.
- Add a new file only when the task cannot be solved cleanly inside an existing responsibility and the new file has one clear purpose.
- Do not create parallel structures, alternative package roots, duplicated configuration trees, or convenience folders.
- If the repository temporarily lacks part of this tree, create only the pieces required by the current task. Do not scaffold unused files merely to make the tree look complete.

Responsibilities:
- `domain/`: entities, state, actions, and exogenous events.
- `simulation/`: engine, transitions, routing, capacity, and costs.
- `decisions/`: policy-facing observations and shared validation.
- `policies/`: policy protocol, heuristic, LLM agent, and fallbacks.
- `experiments/`: event tapes, paired runs, metrics, orchestration.
- `integrations/`: external-provider boundaries; initially the LLM client.
- `data_io/`: configuration loading and result writing.

Dependency rules:
- `simulation/` never imports a concrete policy.
- Policies receive observations and return actions; they never mutate state.
- Provider SDK code stays inside `integrations/`.
- Experiments assemble components but never redefine simulation physics.
- Routing, validation, costs, observations, and action schemas are shared—not duplicated per policy.

## 5. Scientific invariants
Do not change these silently:
1. Compared policies start from identical cloned states.
2. They use the same scenario, horizon, demand, ordinary randomness, and event tape.
3. They receive equivalent information at the same decision times.
4. They use the same actions, validator, fallback semantics, routing, and cost model.
5. Exogenous randomness is generated outside policy execution and reproducible from recorded seeds.
6. Given the same state, validated actions, and events, the transition is deterministic.
7. The LLM cannot mutate state or invent executable routes; it uses approved tools and submits a structured action.
8. Proposed, rejected, validated, fallback, and executed actions are logged separately.
9. Undisrupted counterfactuals preserve ordinary randomness and remove only the designed disruption.
10. Changes to formal semantics, metrics, event order, fairness, or fallbacks require explicit approval and tests.

## 6. Think-before-coding protocol
Before any non-trivial edit:
1. Inspect `git status` and the current diff.
2. Read the relevant source, tests, configuration, and nearby conventions.
3. State a compact task contract:
   - **Goal**
   - **Allowed scope**
   - **Done when**
4. Identify assumptions, risks, and ambiguities affecting correctness.
5. Raise concerns before changing architecture, formal semantics, public interfaces, dependencies, or unrelated files.

A task is non-trivial if it affects multiple responsibilities, public behavior, experiment validity, data formats, dependencies, or more than one directly related module. For a small unambiguous task, use a one-sentence contract; avoid empty ceremony.

## 7. Surgical-change policy
Allowed scope:
- files explicitly named;
- the smallest directly necessary implementation files;
- directly related tests;
- necessary imports, types, and configuration references.

Ask first unless explicitly required:
- broad refactoring or formatting;
- renaming, moving, or deleting files;
- changing APIs or configuration schemas;
- adding or upgrading dependencies;
- changing the formal model, metrics, event order, actions, or fallback behavior;
- modifying unrelated tests or defects;
- introducing a database, service, framework, abstraction layer, or execution model.

Report unrelated problems separately; do not fix them opportunistically. Preserve behavior outside scope. Never rewrite a whole file when a focused edit is sufficient.

## 8. Goal-driven execution and “done”
Translate every implementation request into a verifiable completion condition before coding.

Done means all applicable conditions hold:
- requested behavior and acceptance criteria are satisfied;
- the diff is the smallest coherent change;
- targeted tests pass;
- broader tests pass when shared behavior may be affected;
- configured lint and type checks pass for touched code;
- the final diff contains no unrelated edits, dead code, debug output, or accidental formatting;
- documentation/configuration changed only when required;
- remaining uncertainty and unverified checks are explicit.

Use exact evidence: commands, outcomes, and changed files. Never say “all good” or “should work” without verification. If blocked, stop and explain the blocker, verified facts, and smallest decision needed from Yassine.

## 9. Mandatory file-level explanations
Every Python file in the project—including all `__init__.py` files, source modules, command-line modules, and test files—must begin with a module-level docstring before any import, constant, class, or executable statement.

The opening docstring must use a few complete sentences and explain, in this order:
1. **Local role:** what this file does inside the folder where it is located.
2. **System role:** how that responsibility contributes to the complete simulator and fair policy comparison.
3. **Important boundary:** what the file deliberately does not own when that distinction prevents confusion.

The explanation must be:
- accurate for the current implementation;
- written in plain language understandable to a non-technical reader;
- specific to the file, not copied boilerplate;
- concise but comprehensive enough to explain why the file exists;
- updated whenever the file's responsibility materially changes.

Required Python pattern:

```python
"""Builds policy-facing observations from the current simulation state.

Inside the decisions package, this module selects and organizes the facts that
a decision policy is allowed to see. In the full system, it ensures that the
heuristic and LLM agent receive equivalent information before proposing an
action. It does not execute actions or mutate the simulation state.
"""
```

Do not use empty or vague docstrings such as `"Utilities."`, `"Models."`, or `"Tests for the module."`

For non-Python, human-authored files, provide the same plain-language explanation at the top using the file format's native documentation syntax whenever the format supports it:
- YAML, `.env.example`, and `.gitignore`: leading `#` comments;
- Markdown: an opening explanatory paragraph;
- TOML: leading `#` comments when appropriate.

Machine-readable formats that do not support comments, generated outputs, and intentionally empty marker files such as `.gitkeep` are exempt. Do not corrupt a schema or add fake data fields solely to simulate a docstring.

Before declaring a task done, inspect every created or materially modified file and verify that its opening explanation exists, remains accurate, and satisfies this rule.

## 10. Python standards
- Prefer the standard library before dependencies.
- Prefer small functions, dataclasses, enums, and `Protocol` over deep inheritance.
- Type public functions, methods, and core data structures.
- Keep core domain models explicit; avoid untyped dictionaries.
- Prefer pure deterministic functions for transitions, costs, routing, and metrics.
- Keep state mutation deliberate and localized.
- Fail explicitly with precise errors; never silently coerce invalid domain data.
- Keep files cohesive; do not create one file per class by default.
- Never create catch-all `utils.py`, `helpers.py`, `common.py`, or `manager.py`.
- Comment invariants and non-obvious research semantics—not obvious syntax.
- Add no speculative TODOs, dead code, compatibility layers, or premature optimization.
- Follow existing project patterns unless they conflict with an explicit rule here.

## 11. Testing standards
Protect scientific correctness with:
- unit tests for transitions, routing, capacity, costs, validation, and heuristics;
- a complete tiny-simulation integration test;
- paired-run tests proving identical initial states and event tapes;
- fixed-seed reproducibility tests;
- counterfactual tests proving only the designed disruption changes;
- invalid-action and fallback audit tests.

Use tiny fixtures with manually checkable outcomes. Mock external LLM calls, not domain logic. Tests must not call a live model API; use a fake client returning predefined structured responses. For a bug fix, reproduce it in a test before applying the smallest fix.

## 12. Tooling and verification
Use commands already declared in `pyproject.toml` and project documentation. Do not install tooling without approval. If tooling is absent, propose the smallest setup first.

Typical configured checks may include:
```bash
python -m pytest <targeted tests>
python -m pytest
ruff check .
mypy src
git diff --check
```
Run checks appropriate to the task and disclose every relevant check not run.

## 13. Stop-and-ask conditions
Stop before editing when:
- requirements have materially different interpretations;
- a choice changes fairness or research meaning;
- requested behavior conflicts with tests or this file;
- required domain semantics or data are missing;
- a dependency or broad refactor seems necessary;
- unexpected user changes overlap the task;
- a destructive, migratory, or irreversible action is implied.

For low-risk, local, reversible details, state the assumption and proceed. Do not ask questions answerable by reading the repository.

## 14. End-of-task report
Finish implementation tasks with:
- **Implemented:** behavior changed.
- **Files changed:** exact paths.
- **Verified:** exact commands and outcomes.
- **Not verified:** omitted checks and why.
- **Risks or follow-up:** only concrete remaining concerns.

Never hide failures, warnings, fallbacks, or partial completion.

## 15. Maintaining this compass
Do not modify `CLAUDE.md` unless Yassine explicitly asks. A current explicit instruction may override a workflow preference here, but conflicts with scientific invariants must be surfaced before implementation.

Keep transient task notes out of this file. Add only durable, project-wide rules that should apply in future sessions.
