"""Root package of the Supply-Chain Agent Evaluation simulator.

This package ties together the network/state model, the shared simulation
physics, the classical heuristic and LLM decision policies, and the paired
experiment orchestration that compares them fairly. In the full system, it is
the installable unit that the command-line interface and the test suite both
import. It does not itself contain simulation logic, decision logic, or
experiment logic — those live in the subpackages, each with one clear
responsibility, as fixed by CLAUDE.md.
"""
