"""The read-only view and shared rulebook policies decide against.

Inside the project, this package builds the immutable `DecisionObservation`
that a shipment-level decision is based on, and validates whatever action a
policy proposes in response. In the full system, it is what guarantees the
heuristic and the LLM agent see equivalent information and are held to the
same feasibility rules, so neither can win by seeing more or being allowed
more. It does not execute actions, does not mutate simulation state, and does
not decide anything itself.
"""
