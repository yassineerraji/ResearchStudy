"""The shared physics of the simulator: engine, transitions, routing, costs.

Inside the project, this package owns everything that happens to the world
once a day begins — shipment movement, arrivals, capacity allocation, route
estimation, and every cost calculation. In the full system, it is run
identically regardless of which policy is deciding, which is what makes the
heuristic-versus-LLM comparison fair: the physics never knows or cares which
policy produced a validated action. It does not choose actions, does not
import a concrete policy, and does not call an external LLM provider.
"""
