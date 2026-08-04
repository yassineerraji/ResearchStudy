"""FastAPI backend for the hosted Supply-Chain Agent Evaluation frontend.

Inside `webapp/backend`, this package is the HTTP layer that lets a browser
explore and (in later milestones) launch experiments built on top of the
`supply_chain_simulator` research package. In the full system, it is a thin
façade: it reuses the core package's config models, validation, and output
file formats rather than reimplementing simulation logic. It does not own
any simulation physics, policy behavior, or scientific validity — those
remain entirely the responsibility of `supply_chain_simulator`.
"""
