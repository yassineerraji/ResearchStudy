"""Loads configuration and writes experiment results to disk.

Inside the project, this package parses and validates the YAML configuration
files into typed objects, resolves file paths, and serializes every required
experiment output — manifests, metrics, decision traces, and summaries. In
the full system, it is the only package that touches the filesystem for
configuration and results, keeping that concern separate from simulation and
experiment logic. It does not implement simulation behavior, decision logic,
or statistical calculations itself.
"""
