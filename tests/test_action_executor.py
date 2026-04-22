"""Unit tests for action/executor.py."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pydantic")

from action.executor import Executor
from shared.event_bus import read_events
from shared.paths import stream_paths
from shared.schemas.events import OrchestratorDecision


def _setup_tmp_stream(tmpdir: Path) -> None:
    os.environ["MDK_STREAM_DIR"] = str(tmpdir / "stream")
    os.environ["MDK_MEMORY_DIR"] = str(tmpdir / "memory")


def _decision(
    action: str = "throttle",
    autonomy: str = "L3_bounded_auto",
    params: dict[str, Any] | None = None,
    pending: bool = False,
) -> OrchestratorDecision:
    return OrchestratorDecision(
        decision_id="dec_test_01",
        flag_id="flg_test_01",
        miner_id="m042",
        action=action,  # type: ignore[arg-type]
        action_params=params or {"target_hashrate_pct": 0.80, "duration_min": 60},
        autonomy_level=autonomy,  # type: ignore[arg-type]
        confidence=0.78,
        reasoning_trace="test trace",
        consulted_agents=["voltage_agent", "power_agent"],
        total_cost_usd=0.012,
        total_latency_ms=2100.0,
        pending_human_approval=pending,
    )


def test_l3_throttle_executes_with_rollback() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _setup_tmp_stream(Path(tmp))
        executor = Executor()
        decision = _decision()
        at = executor.handle_decision(decision)
        assert at.status == "executed"
        assert at.action == "throttle"
        assert at.rollback_ts_scheduled is not None
        assert at.rollback_ts_scheduled > datetime.now(tz=timezone.utc)


def test_l4_queues_for_human() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _setup_tmp_stream(Path(tmp))
        executor = Executor()
        decision = _decision(action="shutdown", autonomy="L4_human_only", pending=True)
        at = executor.handle_decision(decision)
        assert at.status == "queued_for_human"
        assert at.rollback_ts_scheduled is None


def test_l2_alert_executes_no_mutation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _setup_tmp_stream(Path(tmp))
        executor = Executor()
        decision = _decision(action="alert_operator", autonomy="L2_suggest", params={})
        at = executor.handle_decision(decision)
        assert at.status == "executed"
        assert at.action == "alert_operator"


def test_l1_observe_logs_only() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _setup_tmp_stream(Path(tmp))
        executor = Executor()
        decision = _decision(action="observe", autonomy="L1_observe", params={})
        at = executor.handle_decision(decision)
        assert at.status == "executed"
        assert at.action == "observe"
        assert at.rollback_ts_scheduled is None


def test_shutdown_at_l3_is_rejected() -> None:
    """Safety backstop — Maestro should never send an irreversible action at L3."""
    with tempfile.TemporaryDirectory() as tmp:
        _setup_tmp_stream(Path(tmp))
        executor = Executor()
        decision = _decision(action="shutdown", autonomy="L3_bounded_auto", params={})
        at = executor.handle_decision(decision)
        assert at.status == "rejected"


def test_fleet_handle_called_on_mutating_l3() -> None:
    calls: list[tuple[str, str, dict[str, Any]]] = []

    class FakeFleet:
        def apply_action(self, miner_id: str, action: str, params: dict[str, Any]) -> None:
            calls.append((miner_id, action, params))

    with tempfile.TemporaryDirectory() as tmp:
        _setup_tmp_stream(Path(tmp))
        executor = Executor(fleet_handle=FakeFleet())
        executor.handle_decision(_decision())
        assert calls == [("m042", "throttle", {"target_hashrate_pct": 0.80, "duration_min": 60})]


def test_emits_action_taken_envelope_to_bus() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _setup_tmp_stream(Path(tmp))
        Executor().handle_decision(_decision())
        sp = stream_paths()
        events = [env for env in read_events(sp.actions) if env.event == "action_taken"]
        assert events, "no action_taken envelope was emitted"
        assert events[0].data["decision_id"] == "dec_test_01"
