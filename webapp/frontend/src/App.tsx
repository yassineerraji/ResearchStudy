// Top-level layout: a header/nav shared by every page, and the routed
// page content below it. Individual pages own their own logic and API
// calls — this component only wires up navigation.

import { NavLink, Outlet } from 'react-router-dom'
import './App.css'

export default function App() {
  return (
    <div className="app-shell">
      <header className="app-header">
        <NavLink to="/" className="brand">
          Supply-Chain Agent Evaluation
        </NavLink>
        <nav className="app-nav">
          <NavLink to="/" end>
            About
          </NavLink>
          <NavLink to="/findings">Findings</NavLink>
          <NavLink to="/gallery">Results Gallery</NavLink>
          <NavLink to="/runs/new">Run Your Own</NavLink>
        </nav>
      </header>
      <main className="app-main">
        <Outlet />
      </main>
      <footer className="app-footer">
        Research simulator comparing a classical heuristic and a bounded LLM agent on
        supply-chain disruption response.
      </footer>
    </div>
  )
}
