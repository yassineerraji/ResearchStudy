// Project explainer. Static content only — no API calls — adapted from the
// repository's own README and the V1/V2 slide decks (reports/*.html) so a
// visitor understands the research question, the model, and the fairness
// mechanism before they open the Findings page or the Results Gallery.

import { Link } from 'react-router-dom'
import styles from './Landing.module.css'

const ACTIONS = [
  { name: 'WAIT', description: 'Keep the shipment on its current plan and let it proceed as scheduled.' },
  { name: 'REROUTE', description: 'Switch to a different non-emergency route, at that route’s own cost and lead time.' },
  { name: 'EXPEDITE', description: 'Pay for a rush, emergency-only lane to arrive faster at a much higher cost.' },
  { name: 'ABSTAIN', description: 'Decline to decide — control falls back to the heuristic, then a safe WAIT, and both are logged.' },
]

export default function Landing() {
  return (
    <article className={styles.landing}>
      <h1>Does a bounded AI agent handle a supply-chain disruption better than a rulebook?</h1>
      <p className={styles.lede}>
        This project builds a small, realistic logistics network, breaks it in a controlled and
        repeatable way, and lets two different decision-makers &mdash; a transparent classical
        heuristic and a bounded, tool-using LLM agent &mdash; respond to the exact same disruption
        under the exact same conditions, so their results can be compared fairly. It has run in two
        stages: <strong>Version 1</strong> tested one fixed network under three disruption
        severities; <strong>Version 2</strong> extended that to three differently-shaped networks
        crossed with those same severities, after auditing V1&rsquo;s results turned up a real
        methodological gap worth fixing first.
      </p>

      <section>
        <h2>The setup</h2>
        <p>
          A supplier, one or more ports, one or more hubs, and a factory that needs a steady stream
          of parts every day. Partway through each run, part of the network is disrupted &mdash; a
          port or hub closes, capacity drops, demand spikes &mdash; for a real but uncertain window:
          in Version 2, neither the exact start day, how long it lasts, nor how late either
          decision-maker even learns about it is fixed in advance. Every shipment already moving,
          and every new one released while the disruption is active, needs a decision.
        </p>
      </section>

      <section>
        <h2>Four moves, nothing else</h2>
        <p>
          Whenever a shipment actually needs a decision &mdash; its route is blocked, it&rsquo;s
          running late, or it&rsquo;s been stuck waiting too long &mdash; a policy chooses exactly
          one of four actions. Neither policy can invent a route, skip a broken node, split a
          shipment, or make up a number.
        </p>
        <div className={styles.pillRow}>
          {ACTIONS.map((action) => (
            <div key={action.name} className={styles.actionCard}>
              <span className={styles.actionName}>{action.name}</span>
              <span className={styles.actionDescription}>{action.description}</span>
            </div>
          ))}
        </div>
      </section>

      <section>
        <h2>How one trial stays fair</h2>
        <p>
          Both decision-makers see identical facts, built from the same read-only snapshot of the
          world, and are held to the exact same validation rules. From one random seed, the
          simulator clones <strong>four</strong> branches after a shared warm-up period: heuristic
          &times; undisrupted, heuristic &times; disrupted, LLM &times; undisrupted, LLM &times;
          disrupted. Subtracting each policy&rsquo;s own undisrupted cost from its own disrupted
          cost isolates the real cost of the disruption for that policy alone
          (<code>TCD = disrupted cost &minus; undisrupted cost</code>); the difference between the
          two policies&rsquo; TCD is what actually gets compared, replication after replication,
          each with fresh random demand, delays, and (in V2) a freshly-realized disruption.
        </p>
      </section>

      <section>
        <h2>From V1 to V2</h2>
        <p>
          Auditing Version 1&rsquo;s real, 300-replication result surfaced something the design
          hadn&rsquo;t anticipated: every single replication in every severity profile agreed on
          the winner &mdash; 100%/0%, never once mixed. Three compounding causes, three fixes:
        </p>
        <div className={styles.versionGrid}>
          <div className={styles.versionCard}>
            <h3>V1&rsquo;s gap</h3>
            <p>One fixed 5-node network, regardless of how the comparison was expected to generalize.</p>
          </div>
          <div className={styles.versionCard}>
            <h3>V2&rsquo;s fix</h3>
            <p>Three topology tiers &mdash; Compact, Standard, Extended &mdash; crossed with three severities: a 3&times;3 grid instead of one point.</p>
          </div>
          <div className={styles.versionCard}>
            <h3>V1&rsquo;s gap</h3>
            <p>The disruption&rsquo;s timing, duration, and disclosure were fully known in advance, every replication.</p>
          </div>
          <div className={styles.versionCard}>
            <h3>V2&rsquo;s fix</h3>
            <p>Start day, duration, and information delay are each sampled fresh per replication &mdash; genuine, unresolved-until-it-happens uncertainty.</p>
          </div>
          <div className={styles.versionCard}>
            <h3>V1&rsquo;s gap</h3>
            <p>Shipment size and shock variety were both fixed &mdash; one port closure, exactly 40 units, every time.</p>
          </div>
          <div className={styles.versionCard}>
            <h3>V2&rsquo;s fix</h3>
            <p>Randomized shipment quantity, plus demand-spike, supplier-shortfall, and compound/cascading shock types alongside the original network shocks.</p>
          </div>
        </div>
      </section>

      <section>
        <h2>What this site shows</h2>
        <p>
          <Link to="/findings">Findings</Link> walks through the audited results directly: how the
          verdict flipped with severity in V1, and how it flips again with topology in V2. The{' '}
          <Link to="/gallery">Results Gallery</Link> browses every completed experiment &mdash; the
          network, the disruption, every shipment-level decision either policy made and why
          (including the LLM agent&rsquo;s actual tool-call reasoning), and the final cost and
          service metrics across every repeated trial. <Link to="/runs/new">Run Your Own</Link>{' '}
          launches a real, live comparison against the OpenAI API, using your own API key and
          model choice.
        </p>
      </section>
    </article>
  )
}
