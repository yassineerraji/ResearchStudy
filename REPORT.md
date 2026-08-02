# Project Status Report

**Purpose:** a snapshot compass for deciding whether to launch the real 100-replication baseline experiment. Written before that run — nothing in this report depends on its results.

**As of:** commit `fd8e076` ("Milestone 9 completed, last step is to run the 100 reps run"), working tree clean.

**Headline numbers:** 323 automated tests, all passing. 93.6% code coverage (gate is 90%). Ruff and Mypy strict both clean. Two real, paid, end-to-end runs against the live OpenAI API have succeeded. The 100-replication baseline run has **not** been executed yet.

---

## Part 1 — How the system works, in plain words

### What it's actually trying to answer

There's a small toy supply chain: a supplier, a couple of ports/a hub, and a factory ("plant") that needs a steady stream of parts. One day, the primary port shuts down for a week. The project asks: **when that disruption hits, does a policy driven by a real AI model handle it better (cheaper) than a simple hand-written rulebook?**

To answer that fairly, both policies are shown the *exact same* disrupted world and the *exact same* undisrupted "what if this never happened" world, so the comparison isn't accidentally rigged by one policy getting an easier scenario.

### The five things you configure before any run

Nothing is a command-line flag — every run is fully described by five YAML files, so a run can always be reproduced later just by keeping those files:

1. **Network** (`configs/networks/baseline_network.yaml`) — the map: nodes, transport lanes, their capacities/costs/speeds/reliability, starting inventory, how demand behaves each day, and the recurring shipment schedule.
2. **Scenario** (`configs/scenarios/port_closure.yaml`) — the disruption: what breaks, when it starts, when it ends, when the policies are told about it.
3. **Heuristic policy config** — two numbers: how many days late a shipment must be before the rulebook will pay for an emergency shipment, and a cost-tie-breaking tolerance.
4. **LLM policy config** — which model, temperature, how many tool calls it's allowed per decision, timeouts/retries, what happens if it fails (fall back to the rulebook or just wait), and whether to actually call the real API ("LIVE") or replay a previously recorded run ("REPLAY").
5. **Experiment config** — how many warm-up days, how long the experiment runs, how many extra "drain" days are allowed for straggling shipments to finish, how many independent repetitions ("replications") to run, the random seed, and which output files to write.

Plus two environment variables (`OPENAI_API_KEY`, `LLM_MODEL`) that live only in the git-ignored `.env` file — never in a config file, never written to any output.

### What happens when you launch a run, step by step

1. **Load & validate.** All five config files are parsed and checked. A typo, a broken reference, an out-of-range number — any of it stops the run immediately with a clear error before anything is simulated.
2. **Build the two policies.** The rulebook (pure arithmetic) and the AI agent (wraps a real or replayed OpenAI connection).
3. **Create one output folder** for this whole experiment, named with the experiment id and a timestamp.
4. **For each replication** (an independent repetition, e.g. replication 1, 2, 3, ... — think of it as re-running the same experiment with a fresh dice roll each time, to see if the result holds up or was a fluke):
   - A random seed is derived from the replication number, so replication 7 always gets the exact same "dice rolls" no matter how many times you run the experiment.
   - **One shared random future is generated up front**: every day's demand, every day's ordinary transport delays for every lane, and the fixed shipment-release schedule. This is generated *before* either policy makes a single decision, so neither policy can be given an easier random draw than the other.
   - A second copy of that same future is made with only the port closure removed — same demand, same delays, same shipments, just no disruption. This is the fair "what if nothing bad happened" baseline.
   - A 20-day **warm-up** runs with no decisions at all (shipments just follow their default route), purely to get a realistic number of shipments already in transit before anything interesting happens.
   - That warmed-up moment is frozen and copied **four times** — once per combination of (rulebook or AI) × (disrupted or undisrupted world). All four start from an identical situation.
   - **Each of the four runs plays out day by day** until the horizon ends (plus a few extra "drain" days if shipments are still in transit). Every day: the disruption turns on/off as scheduled, shipments arrive, new shipments are released, today's demand is served from stock (or added to a backlog if stock runs out), and — only for shipments that actually need a decision that day (blocked route, running late, stuck waiting, etc.) — the relevant policy is asked what to do. The rulebook answers instantly with math. The AI agent has a back-and-forth with the model: it can ask a handful of read-only questions ("what does this shipment look like?", "what are my routing options?") before submitting one final decision. Whatever either policy proposes gets checked against the exact same rulebook of "is this even a legal move" — if it's not (or the AI declines to decide), a safe fallback kicks in and that's recorded too, so nothing invalid ever actually happens to the simulation.
   - Every day's costs (transport, rerouting, rush shipping, storage, backlog, lateness, and a penalty if things still aren't resolved by the very end) pile up separately for each of the four runs.
   - The disruption's real cost for each policy is *(cost with the disruption) − (cost without it)*. The final comparison is *(AI's disruption cost) − (rulebook's disruption cost)* — negative means the AI came out cheaper.
   - Everything is written to disk as it happens: the random future used, every decision made (by both policies, valid or not), every AI conversation (what it asked, what it was told, what it answered, how many tokens, how long it took), daily snapshots, and this replication's final comparison.
5. **After every replication is done**, the summary numbers are computed: average cost difference per policy, the average "who won" gap with a confidence interval, how often each policy won, and some percentile numbers — written to one final summary file, plus printed to the screen.

### Parallel or sequential?

**Everything is strictly sequential** — one replication after another, one of the four branches after another, one day after another, one decision after another. This is deliberate: no multi-threading, no parallel branches, nothing running at the same time. It makes the whole thing slower but far easier to reason about and guaranteed reproducible.

### What you can and can't change per launch

You *can* change, by editing the relevant YAML file before a run: the network layout and costs, the disruption's timing/severity, either policy's tuning knobs, the AI model and its call budget, how many days the experiment covers, how many replications to run, the random seed, and which output files get written.

You *cannot* override any of this from the command line — that's intentional, so a saved config file is always the complete, honest record of exactly what produced a given result.

---

## Part 2 — What's actually guaranteed, backed by passing tests

323 automated tests currently pass (0 failing, 0 skipped), covering 93.6% of the code (the project's own bar is 90%). Below is what each area of that suite actually proves, not just what exists.

- **The basic building blocks are structurally sound.** Every domain object (nodes, edges, products, shipments, actions, shocks) rejects invalid values (negative costs, backwards date ranges, contradictory shipment status, oversized rationale text, etc.) exactly where it should.
- **Bad configuration is caught before anything runs**, with specific, readable errors — tested against a long list of deliberately broken configs (duplicate ids, unknown references, a route that doesn't actually connect, a path that tries to escape the repository, and more).
- **The random "future" driving each replication is provably fair and reproducible**: the same seed always produces the exact same demand/delay/release schedule; different replications produce different ones; the undisrupted counterfactual is proven to differ from the disrupted one *only* in the disruption itself — demand, delays, and shipment releases are identical.
- **The day-by-day physics are individually verified**: shipment release, in-transit movement, arrival, delivery, demand being served from stock first and backlog second, what happens when a lane or node is unavailable, what happens when capacity runs out for a day — each checked against hand-calculated expected outcomes.
- **Every cost formula is checked against a hand-calculated number** — transport, rerouting, rushed shipping, storage, backlog, lateness penalties, and the end-of-run penalty for anything still unresolved.
- **Every possible way a proposed decision can be rejected is individually tested** (wrong shipment, already delivered, route doesn't start/end where it should, route uses something currently unavailable, capacity too small, calling "rush shipping" on a normal route or "reroute" on an emergency one, and so on) — and a valid decision of each kind (wait / reroute / rush / decline) is proven to pass.
- **The rulebook policy's exact decision rule is verified**, including every tie-breaking case (identical costs, no alternative available, a decision that's cheapest but not yet urgent enough to justify rush shipping).
- **A full miniature version of the whole simulation is proven correct end to end** — a tiny 3-stop network, run for several days both with and without a disruption, checked against numbers computed by hand, plus that same tiny scenario run through the real rulebook and the real safety-fallback logic.
- **The fairness of the four-branch comparison is proven**: all four branches genuinely start from an identical frozen state (proper independent copies, not shared/mutated state), and the final "who won by how much" arithmetic is exact.
- **Reproducibility is proven for everything except the live AI call itself**: the same setup run twice produces byte-for-byte identical results for the rulebook policy, and for the AI policy when it's driven by a scripted stand-in or by *replaying* a previously recorded real conversation.
- **The AI integration's plumbing is verified without spending any money**: the back-and-forth tool-call loop, retrying a dropped connection with increasing waits, *not* retrying a fundamentally bad request, never storing or leaking the API key, and the "replay a past conversation" mode correctly detecting a missing or duplicated record — all checked against a scripted stand-in for OpenAI's API, never the real network.
- **The AI agent's decision-making logic is verified in isolation**: a normal answer becomes a real decision; a broken or nonsensical answer safely becomes "decline to decide" with the correct reason recorded; running out of allowed questions safely becomes "decline to decide" too; each of its five tools returns the right information.
- **A real, deliberately planted fairness bug was found and fixed, and is now guarded by a regression test**: earlier in this project, the simulator could let a policy see a future disruption's exact details before it was officially "announced." That's fixed, and there's now a test proving a disruption never appears in what a policy is shown before its announcement day.
- **Beyond the automated suite, three real calls to the live OpenAI API have actually happened** (not scripted stand-ins): one single hand-built decision, and two small real 2-replication experiments run through the actual command-line tool. All three produced valid, sensible, secret-free results. The second small experiment (after a prompt fix, see below) had a **0% invalid-decision rate** across 24 real AI decisions.

---

## Part 3 — What's still untested, or doesn't work

Being direct about the gaps, in rough order of how much they matter before spending on the 100-replication run:

- **The real 100-replication baseline experiment has never been run.** Everything known about "does the AI actually beat the rulebook" comes from a deliberately shortened, twice-run 2-replication test on a *modified* config (shorter horizon, shorter drain, only 2 replications) — not the real, locked `baseline_comparison.yaml`. The actual cost, win rate, and runtime of the real experiment are genuinely unknown until it runs; the numbers in this report are estimates.
- **The command-line tool itself (`cli.py`) has 0% automated test coverage.** There is no `test_cli.py`. It's been exercised manually twice (config validation, and two real runs) and both worked — but there's no automated safety net for its argument parsing, exit-code mapping, or its error handling when, say, the API key environment variable is missing or a replay trace file doesn't exist. Those specific failure paths have never actually been triggered, in a test or for real.
- **`README.md` is completely empty (0 bytes).** Anyone cloning this repository fresh has no written instructions for installing it, setting up `.env`, or running anything. This was explicitly deferred rather than fixed, since the project's own build plan treats it as a wrap-up item once there's a real run to document.
- **`.env.example` no longer exists in the working tree** (it was deleted at some point outside of this conversation — visible in the latest commit's diff). The project's own file-structure contract calls for this template file to exist. It holds no secrets either way, but it's a real, current gap worth knowing about.
- **The AI's behavior at the real horizon length (60 days + up to 30 drain days) is untested.** Verification so far only went up to a 30-day horizon / 15-day drain. A longer run means more decisions and more chances for a rare edge case (unusual rationale text, hitting the tool-call limit, a shipment released right at the very end behaving oddly near the boundary) that simply hasn't had the opportunity to show up yet.
- **Live AI reproducibility is inherently limited, by design.** Every other part of this system is proven byte-for-byte reproducible for a given seed. The live AI branch is not, and cannot be — even at the lowest randomness setting, the same prompt can get a slightly different real answer twice. This is an accepted, documented limitation, not a bug: exact re-runs of a *specific* AI decision are handled separately, by replaying a saved transcript instead of calling the API again.
- **The "replay a real recorded run" mode has only been tested with a manufactured example.** The mechanics are proven (matching, detecting a missing/duplicate record), but the specific combination of *"take a file actually produced by a real live run, feed it back into the command-line tool in replay mode"* has not been exercised end to end.
- **One prompt-quality issue was found and fixed via manual inspection, and there are likely others not yet surfaced.** The first small real run showed the AI confusing "rush shipping" with "just a good normal route" 20-38% of the time. That specific issue is now fixed and confirmed (0% invalid rate on the retest) — but this was found by manually reading roughly 50 real decisions total. A 100-replication run will likely involve a much larger number of real decisions, which is exactly the kind of larger sample that could surface a *different* rare issue that hasn't happened yet in the small samples seen so far.
- **A required error type from the project's own contract doesn't seem to exist yet.** `ActionValidationError` is named as a required exception, but nothing in the codebase currently defines or raises it — validation failures are represented a different way (as a result value, not a thrown error) and that works fine, but it's a discrepancy from the letter of the spec worth being aware of.
- **There's no resume/checkpoint mechanism.** The real experiment is configured to stop the *entire* 100-replication run on the very first failure of any kind (a strict setting called "fail fast," inherited from the locked baseline config). If something goes wrong on replication 63 — a dropped connection that exhausts its retries, for instance — there is no way to resume from there; the whole run would need to be started over from replication 1.
- **Nothing has been tested on any machine other than this one** (macOS). No claim is made, or tested, about behavior on Linux or Windows.
- **The cost and time estimates for the real 100-replication run are projections from a much smaller sample**, not measurements. They should be treated as a rough planning number, not a guarantee.

---

## Where things stand right now

- Two real, small, live-API runs already exist on disk under `outputs/` from this session — useful as a reference for what real output looks like, but not the actual experiment.
- Every quality gate that can run without spending money (unit tests, integration tests, linting, type checking, coverage) is green.
- The only remaining step in the project's own build plan before "Version 1 complete" is the real 100-replication run itself.
