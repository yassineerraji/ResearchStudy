// Project explainer. Static content only — no API calls — summarizing the
// research question and method from the repository's own README so a
// visitor understands what the Results Gallery is showing before they open
// it.

import { Link } from 'react-router-dom'
import styles from './Landing.module.css'

export default function Landing() {
  return (
    <article className={styles.landing}>
      <h1>Does a bounded AI agent handle a supply-chain disruption better than a rulebook?</h1>
      <p className={styles.lede}>
        This project builds a small, realistic logistics network, breaks it in a controlled and
        repeatable way, and lets two different decision-makers — a transparent classical
        heuristic and a bounded, tool-using LLM agent — respond to the exact same disruption
        under the exact same conditions, so their results can be compared fairly.
      </p>

      <section>
        <h2>The setup</h2>
        <p>
          A supplier, two ports, a hub, and a factory that needs a steady stream of parts every
          day. Partway through each run, the primary port closes for a week. Shipments already
          moving, and every new one released while it&rsquo;s closed, need a decision: wait it
          out, take a longer detour, or pay extra for a rush shipment.
        </p>
      </section>

      <section>
        <h2>The comparison</h2>
        <p>
          Both decision-makers see identical facts and are held to the exact same rules about
          what counts as a valid action. Neither can invent a route, skip a broken node, or make
          up numbers — an LLM policy that declines to answer or produces something invalid falls
          back to the heuristic automatically, and that fallback is recorded. Each run is also
          repeated once <em>without</em> the disruption, so the real cost of the disruption for
          each policy is isolated: <code>(disrupted cost) − (undisrupted cost)</code>. Whichever
          policy&rsquo;s disruption cost is lower wins that replication.
        </p>
      </section>

      <section>
        <h2>What this site shows</h2>
        <p>
          The <Link to="/gallery">Results Gallery</Link> browses completed experiments: the
          network the shipments move through, the day the port closes, every shipment-level
          decision either policy made and why, and the final cost and service metrics compared
          across up to 100 repeated trials per experiment.
        </p>
      </section>
    </article>
  )
}
