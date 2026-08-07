"""Tests for the in-memory run bookkeeping in `app.services.run_registry`.

Inside `tests`, this module exercises `RunRegistry` directly (via
`asyncio.run`, no HTTP layer or subprocess involved) — state transitions,
unknown-run lookups, and list ordering — since this is the one module every
other M3 piece depends on being correct.
"""

from __future__ import annotations

import asyncio

import pytest

from app.services.run_registry import RunNotFoundError, RunRegistry, RunStatus


def test_create_then_get_returns_queued_record() -> None:
    async def scenario() -> None:
        registry = RunRegistry()
        created = await registry.create("run-1", total_replications=5)
        assert created.status == RunStatus.QUEUED
        assert created.total_replications == 5
        assert created.completed_replications == 0

        fetched = await registry.get("run-1")
        assert fetched.run_id == "run-1"
        assert fetched.status == RunStatus.QUEUED

    asyncio.run(scenario())


def test_get_unknown_run_raises() -> None:
    async def scenario() -> None:
        registry = RunRegistry()
        with pytest.raises(RunNotFoundError):
            await registry.get("does-not-exist")

    asyncio.run(scenario())


def test_update_mutates_fields() -> None:
    async def scenario() -> None:
        registry = RunRegistry()
        await registry.create("run-1", total_replications=3)
        updated = await registry.update(
            "run-1", status=RunStatus.RUNNING, completed_replications=1
        )
        assert updated.status == RunStatus.RUNNING
        assert updated.completed_replications == 1

        fetched = await registry.get("run-1")
        assert fetched.status == RunStatus.RUNNING
        assert fetched.completed_replications == 1

    asyncio.run(scenario())


def test_update_unknown_run_raises() -> None:
    async def scenario() -> None:
        registry = RunRegistry()
        with pytest.raises(RunNotFoundError):
            await registry.update("does-not-exist", status=RunStatus.RUNNING)

    asyncio.run(scenario())


def test_list_all_orders_newest_first() -> None:
    async def scenario() -> None:
        registry = RunRegistry()
        await registry.create("older", total_replications=1)
        await asyncio.sleep(0.01)
        await registry.create("newer", total_replications=1)

        records = await registry.list_all()
        assert [r.run_id for r in records] == ["newer", "older"]

    asyncio.run(scenario())
