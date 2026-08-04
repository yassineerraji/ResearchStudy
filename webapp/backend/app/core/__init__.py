"""Shared backend infrastructure: paths and error mapping.

Inside `app`, this package holds the small pieces every router and service
needs but that don't belong to any one feature: where the repository and its
`outputs/`/sandbox directories live, and how core-package exceptions map to
HTTP responses. In the full backend, it is the single place that knows about
the host filesystem layout, so routers and services never hardcode paths. It
does not implement any config parsing or simulation orchestration itself.
"""
