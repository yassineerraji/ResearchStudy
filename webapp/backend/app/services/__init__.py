"""Business logic behind the API: config handling and result reading.

Inside `app`, this package does the actual work each router exposes —
wrapping `supply_chain_simulator`'s config models and validation, and reading
experiment output files. In the full backend, it is the only layer that
imports from `supply_chain_simulator` directly, so routers stay HTTP-only.
It does not define HTTP request/response shapes itself; those live in
`app.schemas`.
"""
