"""Assembles the paired, fair comparison experiment from shared components.

Inside the project, this package generates the policy-independent random
event tape, runs the four-branch paired experiment (heuristic/LLM * 
undisrupted/disrupted) for every replication, and computes the resulting
cost-of-disruption metrics. In the full system, it is what turns the shared
simulation physics and the two policies into the actual research result — the
paired TCD and delta statistics. It orchestrates existing components only; it
does not redefine simulation physics, decision rules, or cost formulas.
"""
