"""HTTP route handlers, grouped by feature area.

Inside `app`, this package holds one module per feature (`configs`,
`gallery`, and later `runs`). In the full backend, routers only parse
requests, call into `app.services`, and shape responses — they contain no
business logic themselves. It does not own config validation, file parsing,
or run orchestration; those live in `app.services`.
"""
