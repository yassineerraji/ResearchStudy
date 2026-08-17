"""Unit tests for cli.py's dotenv loader.

Inside tests/unit, this file checks _load_dotenv's parsing rules in
isolation: it loads simple KEY=VALUE lines, skips comments/blanks, never
overrides an already-set environment variable, and never loads a blank
value (so an unfilled-in placeholder line behaves as if absent). It does
not test the rest of cli.py (argument parsing, run orchestration), which
has no dedicated test file in this project and is exercised indirectly
through every integration test that drives the underlying components
cli.py wires together.
"""

from __future__ import annotations

import os
from pathlib import Path

from supply_chain_simulator.cli import _load_dotenv


class TestLoadDotenv:
    def test_missing_file_is_a_no_op(self, tmp_path: Path) -> None:
        _load_dotenv(tmp_path)  # no .env present; must not raise

    def test_loads_simple_key_value_pairs(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("TEST_DOTENV_KEY", raising=False)
        (tmp_path / ".env").write_text("TEST_DOTENV_KEY=abc123\n")
        _load_dotenv(tmp_path)
        assert os.environ["TEST_DOTENV_KEY"] == "abc123"

    def test_skips_comments_and_blank_lines(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("TEST_DOTENV_KEY", raising=False)
        (tmp_path / ".env").write_text("# a comment\n\nTEST_DOTENV_KEY=abc123\n")
        _load_dotenv(tmp_path)
        assert os.environ["TEST_DOTENV_KEY"] == "abc123"

    def test_strips_surrounding_quotes(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("TEST_DOTENV_KEY", raising=False)
        (tmp_path / ".env").write_text('TEST_DOTENV_KEY="abc123"\n')
        _load_dotenv(tmp_path)
        assert os.environ["TEST_DOTENV_KEY"] == "abc123"

    def test_never_overrides_an_already_set_variable(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("TEST_DOTENV_KEY", "from_shell")
        (tmp_path / ".env").write_text("TEST_DOTENV_KEY=from_dotenv\n")
        _load_dotenv(tmp_path)
        assert os.environ["TEST_DOTENV_KEY"] == "from_shell"

    def test_blank_value_is_treated_as_absent(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("TEST_DOTENV_KEY", raising=False)
        (tmp_path / ".env").write_text("TEST_DOTENV_KEY=\n")
        _load_dotenv(tmp_path)
        assert "TEST_DOTENV_KEY" not in os.environ
