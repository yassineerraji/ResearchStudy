// Renders the logistics network as a small SVG diagram.
//
// Node latitude/longitude are optional in the research package's own schema
// (Node/NodeConfig — see webapp/backend's plan notes) and are frequently
// null, so this deliberately does not try to be a geo-map. Instead it lays
// nodes out in columns by node_type (SUPPLIER -> PORT -> HUB -> PLANT ->
// CUSTOMER), which is a faithful, deterministic picture of this network:
// it's a small directed acyclic graph, not a general graph needing a force
// layout.

import { useMemo } from 'react'
import type { NetworkConfigContent, Shock } from '../api/types'
import styles from './NetworkDiagram.module.css'

const RANK_ORDER = ['SUPPLIER', 'PORT', 'HUB', 'PLANT', 'CUSTOMER']

interface Point {
  x: number
  y: number
}

function layout(network: NetworkConfigContent, width: number, height: number): Map<string, Point> {
  const byRank = new Map<number, string[]>()
  for (const node of network.nodes) {
    const rank = RANK_ORDER.includes(node.node_type)
      ? RANK_ORDER.indexOf(node.node_type)
      : RANK_ORDER.length
    const bucket = byRank.get(rank) ?? []
    bucket.push(node.node_id)
    byRank.set(rank, bucket)
  }
  const ranks = [...byRank.keys()].sort((a, b) => a - b)
  const positions = new Map<string, Point>()
  const marginX = 60
  const marginY = 40
  ranks.forEach((rank, columnIndex) => {
    const nodeIds = byRank.get(rank)!
    const x =
      ranks.length === 1
        ? width / 2
        : marginX + (columnIndex * (width - 2 * marginX)) / (ranks.length - 1)
    nodeIds.forEach((nodeId, rowIndex) => {
      const y =
        nodeIds.length === 1
          ? height / 2
          : marginY + (rowIndex * (height - 2 * marginY)) / (nodeIds.length - 1)
      positions.set(nodeId, { x, y })
    })
  })
  return positions
}

interface NetworkDiagramProps {
  network: NetworkConfigContent
  shocks: Shock[]
  activeShockIds: string[]
}

const WIDTH = 640
const HEIGHT = 260
const NODE_RADIUS = 22

export default function NetworkDiagram({ network, shocks, activeShockIds }: NetworkDiagramProps) {
  const positions = useMemo(() => layout(network, WIDTH, HEIGHT), [network])

  const activeTargets = useMemo(() => {
    const nodeIds = new Set<string>()
    const edgeIds = new Set<string>()
    for (const shock of shocks) {
      if (!activeShockIds.includes(shock.shock_id)) continue
      if (shock.target_type === 'NODE') nodeIds.add(shock.target_id)
      if (shock.target_type === 'EDGE') edgeIds.add(shock.target_id)
    }
    return { nodeIds, edgeIds }
  }, [shocks, activeShockIds])

  return (
    <svg
      className={styles.diagram}
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      role="img"
      aria-label="Logistics network diagram"
    >
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--text-muted)" />
        </marker>
        <marker id="arrow-active" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--status-critical)" />
        </marker>
      </defs>

      {network.edges.map((edge) => {
        const from = positions.get(edge.origin_node_id)
        const to = positions.get(edge.destination_node_id)
        if (!from || !to) return null
        const isActive = activeTargets.edgeIds.has(edge.edge_id)
        const angle = Math.atan2(to.y - from.y, to.x - from.x)
        const startX = from.x + Math.cos(angle) * NODE_RADIUS
        const startY = from.y + Math.sin(angle) * NODE_RADIUS
        const endX = to.x - Math.cos(angle) * (NODE_RADIUS + 6)
        const endY = to.y - Math.sin(angle) * (NODE_RADIUS + 6)
        return (
          <g key={edge.edge_id}>
            <line
              x1={startX}
              y1={startY}
              x2={endX}
              y2={endY}
              stroke={isActive ? 'var(--status-critical)' : 'var(--gridline)'}
              strokeWidth={isActive ? 2 : 1.5}
              strokeDasharray={edge.emergency ? '4 3' : undefined}
              markerEnd={isActive ? 'url(#arrow-active)' : 'url(#arrow)'}
            />
            <title>
              {edge.edge_id} ({edge.mode}
              {edge.emergency ? ', emergency' : ''})
            </title>
          </g>
        )
      })}

      {network.nodes.map((node) => {
        const point = positions.get(node.node_id)
        if (!point) return null
        const isActive = activeTargets.nodeIds.has(node.node_id)
        return (
          <g key={node.node_id} transform={`translate(${point.x}, ${point.y})`}>
            <circle
              r={NODE_RADIUS}
              fill="var(--surface-1)"
              stroke={isActive ? 'var(--status-critical)' : 'var(--baseline)'}
              strokeWidth={isActive ? 2.5 : 1.5}
            />
            <text textAnchor="middle" dy="0.35em" className={styles.nodeLabel}>
              {node.node_type.slice(0, 4)}
            </text>
            <text textAnchor="middle" y={NODE_RADIUS + 16} className={styles.nodeName}>
              {node.name}
            </text>
            <title>{node.node_id}</title>
          </g>
        )
      })}
    </svg>
  )
}
