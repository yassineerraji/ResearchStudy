"""Shared vocabulary of the simulator: entities, state, actions, and events.

Inside the project, this package defines the network, product, shipment,
action, and disruption concepts that every other package builds on — nodes,
edges, the mutable daily state, the four decision actions, and exogenous
events like demand and shocks. In the full system, it is the one package that
both the physics and the policies agree on, so a shipment or a route means
exactly the same thing everywhere. It does not implement any simulation
behavior, decision logic, or I/O, and it does not import any other project
package.
"""
