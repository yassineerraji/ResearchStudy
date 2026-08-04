"""Test package for the hosted web backend.

Makes `tests` importable so test modules can share fixtures/constants (e.g.
`from tests.conftest import FAKE_EXPERIMENT_DIR_NAME`) via a normal package
import instead of relative-path tricks. Holds no test logic itself.
"""
