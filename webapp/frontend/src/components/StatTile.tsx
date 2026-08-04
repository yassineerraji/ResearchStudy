// A single labeled number, per the dataviz skill's stat-tile pattern: for a
// single headline metric, a plain number reads faster than a chart. Used in
// rows on the Gallery card and the RunDetail summary strip.

import styles from './StatTile.module.css'

interface StatTileProps {
  label: string
  value: string
  sub?: string
}

export default function StatTile({ label, value, sub }: StatTileProps) {
  return (
    <div className={styles.tile}>
      <div className={styles.label}>{label}</div>
      <div className={styles.value}>{value}</div>
      {sub ? <div className={styles.sub}>{sub}</div> : null}
    </div>
  )
}
