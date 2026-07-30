"""The boundary between the project and an external LLM provider.

Inside the project, this package is the only place that sends a network
request to a language-model provider or its SDK. In the full system, it is
used exclusively by the LLM policy, so that no other part of the simulator —
least of all the shared physics — could ever accidentally depend on an
external service. It does not see or hold simulation state; it only exchanges
already-built prompts, tool calls, and structured responses.
"""
