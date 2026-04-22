"""Tests for ab_experiment/metrics.py — metric computation against fixture event logs.

Uses synthetic JSONL fixtures written in-memory (no real simulator runs).
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ab_experiment.metrics import (
    PRE_FAULT_CATCH_WINDOW_HOURS,
    TrackMetrics,
    _collect_actions,
    _collect_faults,
    _collect_flags,
    _parse_ts,
    compute_track_metrics,
)


# ---------------------------------------------------------------------------
# Fixtures — synthetic JSONL helpers
# ---------------------------------------------------------------------------


def _ts(offset_seconds: float = 0.0) -> str:
    """UTC timestamp with optional offset from epoch base."""
    base = datetime(2026, 4, 20, 10, 0, 0, tzinfo=timezone.utc)
    from datetime import timedelta
    return (base + timedelta(seconds=offset_seconds)).isoformat().replace("+00:00", "Z")


def _telemetry_event(miner_id: str, ts: str, fault: str | None = None) -> dict:
    return {
        "event": "telemetry_tick",
        "ts": ts,
        "source": "simulator",
        "data": {
            "miner_id": miner_id,
            "miner_model": "S19j Pro",
            "hashrate_th": 100.0,
            "hashrate_expected_th": 104.0,
            "temp_chip_c": 75.0,
            "temp_amb_c": 24.0,
            "power_w": 3250.0,
            "voltage_v": 12.0,
            "fan_rpm": [5800, 5800, 5800, 5800],
            "operating_mode": "balanced",
            "uptime_s": 86400.0,
            "env": {
                "site_temp_c": 23.0,
                "site_humidity_pct": 40.0,
                "elec_price_usd_kwh": 0.07,
                "hashprice_usd_per_th_day": 0.058,
            },
            "fault_injected": fault,
        },
    }


def _flag_event(
    flag_id: str, miner_id: str, ts: str, severity: str = "warn", flag_type: str = "voltage_drift"
) -> dict:
    return {
        "event": "flag_raised",
        "ts": ts,
        "source": "detector",
        "data": {
            "flag_id": flag_id,
            "miner_id": miner_id,
            "flag_type": flag_type,
            "severity": severity,
            "confidence": 0.75,
            "source_tool": "rule_engine",
            "evidence": {"metric": "voltage_v", "window_min": 30.0},
            "raw_score": 0.8,
        },
    }


def _decision_event(
    decision_id: str,
    flag_id: str,
    miner_id: str,
    ts: str,
    action: str = "throttle",
    cost_usd: float = 0.02,
    latency_ms: float = 1500.0,
) -> dict:
    return {
        "event": "orchestrator_decision",
        "ts": ts,
        "source": "orchestrator",
        "data": {
            "decision_id": decision_id,
            "flag_id": flag_id,
            "miner_id": miner_id,
            "action": action,
            "action_params": {"target_hashrate_pct": 0.80, "duration_min": 60},
            "autonomy_level": "L3_bounded_auto",
            "confidence": 0.78,
            "reasoning_trace": "Test reasoning trace for m042",
            "consulted_agents": ["voltage_agent"],
            "total_cost_usd": cost_usd,
            "total_latency_ms": latency_ms,
            "pending_human_approval": False,
        },
    }


def _action_event(
    action_id: str, decision_id: str, miner_id: str, ts: str, action: str = "throttle"
) -> dict:
    return {
        "event": "action_taken",
        "ts": ts,
        "source": "action",
        "data": {
            "action_id": action_id,
            "decision_id": decision_id,
            "miner_id": miner_id,
            "action": action,
            "status": "executed",
            "outcome_expected": "hashrate drops",
            "outcome_observed": None,
            "rollback_ts_scheduled": None,
        },
    }


def _write_jsonl(path: Path, events: list[dict]) -> None:
    with path.open("w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestParseTs:
    def test_z_suffix(self):
        ts = _parse_ts("2026-04-20T10:00:00Z")
        assert ts is not None
        assert ts.tzinfo is not None

    def test_plus_utc(self):
        ts = _parse_ts("2026-04-20T10:00:00+00:00")
        assert ts is not None

    def test_none_input(self):
        assert _parse_ts(None) is None

    def test_invalid_string(self):
        assert _parse_ts("not-a-date") is None


class TestCollectFaults:
    def test_no_faults(self):
        events = [_telemetry_event("m001", _ts(0), fault=None)]
        result = _collect_faults(events)
        assert result == {}

    def test_single_fault_onset(self):
        events = [
            _telemetry_event("m001", _ts(0), fault=None),
            _telemetry_event("m001", _ts(100), fault="chip_instability"),
            _telemetry_event("m001", _ts(200), fault="chip_instability"),
        ]
        result = _collect_faults(events)
        assert "m001" in result
        assert len(result["m001"]) == 1  # only first onset

    def test_multiple_miners(self):
        events = [
            _telemetry_event("m001", _ts(100), fault="power_sag"),
            _telemetry_event("m002", _ts(200), fault="cooling_degradation"),
        ]
        result = _collect_faults(events)
        assert "m001" in result
        assert "m002" in result


class TestCollectActions:
    def test_empty(self):
        assert _collect_actions([]) == []

    def test_parses_action(self):
        events = [_action_event("act_001", "dec_001", "m042", _ts(100))]
        result = _collect_actions(events)
        assert len(result) == 1
        assert result[0]["miner_id"] == "m042"
        assert result[0]["action"] == "throttle"

    def test_skips_non_action_events(self):
        events = [
            _flag_event("flg_001", "m001", _ts(0)),
            _action_event("act_001", "dec_001", "m042", _ts(100)),
        ]
        result = _collect_actions(events)
        assert len(result) == 1


class TestCollectFlags:
    def test_parses_flag(self):
        events = [_flag_event("flg_001", "m042", _ts(0))]
        result = _collect_flags(events)
        assert len(result) == 1
        assert result[0]["flag_id"] == "flg_001"
        assert result[0]["miner_id"] == "m042"


# ---------------------------------------------------------------------------
# Integration: compute_track_metrics on fixture JSONL files
# ---------------------------------------------------------------------------


class TestComputeTrackMetrics:
    def test_happy_path_with_catch(self, tmp_path: Path):
        """Track catches a fault that occurred after an action within the window."""
        # Fault onset at t=3600s (1 hour into sim time)
        fault_ts = _ts(3600)
        # Action on m042 at t=3000s (10 min before fault) — within 1h window
        action_ts = _ts(3000)
        flag_ts = _ts(2900)

        telemetry = [
            _telemetry_event("m042", _ts(0), fault=None),
            _telemetry_event("m042", fault_ts, fault="chip_instability"),
        ]
        flags = [_flag_event("flg_001", "m042", flag_ts)]
        decisions = [_decision_event("dec_001", "flg_001", "m042", _ts(2950))]
        actions = [_action_event("act_001", "dec_001", "m042", action_ts)]

        _write_jsonl(tmp_path / "telemetry.jsonl", telemetry)
        _write_jsonl(tmp_path / "flags.jsonl", flags)
        _write_jsonl(tmp_path / "decisions.jsonl", decisions)
        _write_jsonl(tmp_path / "actions.jsonl", actions)

        m = compute_track_metrics(
            track="A",
            telemetry_path=tmp_path / "telemetry.jsonl",
            flags_path=tmp_path / "flags.jsonl",
            decisions_path=tmp_path / "decisions.jsonl",
            actions_path=tmp_path / "actions.jsonl",
        )

        assert m.faults_injected == 1
        assert m.faults_caught == 1
        assert m.flags_raised == 1
        assert m.total_actions == 1
        assert m.total_cost_usd == pytest.approx(0.02, abs=0.001)

    def test_no_catch_if_action_outside_window(self, tmp_path: Path):
        """Action taken 2 hours before fault — outside 1h window, so no catch."""
        fault_ts = _ts(7200)  # fault at 2h
        action_ts = _ts(0)    # action at 0h — 2h before fault, outside 1h window

        telemetry = [_telemetry_event("m001", fault_ts, fault="power_sag")]
        flags = [_flag_event("flg_001", "m001", _ts(0))]
        decisions = [_decision_event("dec_001", "flg_001", "m001", _ts(10))]
        actions = [_action_event("act_001", "dec_001", "m001", action_ts)]

        _write_jsonl(tmp_path / "telemetry.jsonl", telemetry)
        _write_jsonl(tmp_path / "flags.jsonl", flags)
        _write_jsonl(tmp_path / "decisions.jsonl", decisions)
        _write_jsonl(tmp_path / "actions.jsonl", actions)

        m = compute_track_metrics(
            track="A",
            telemetry_path=tmp_path / "telemetry.jsonl",
            flags_path=tmp_path / "flags.jsonl",
            decisions_path=tmp_path / "decisions.jsonl",
            actions_path=tmp_path / "actions.jsonl",
        )

        assert m.faults_injected == 1
        assert m.faults_caught == 0  # outside window

    def test_false_positive_detection(self, tmp_path: Path):
        """Action on miner with no fault nearby = false positive."""
        # No faults in telemetry, but Track B fires an alert on m001
        telemetry = [_telemetry_event("m001", _ts(0), fault=None)]
        flags = [_flag_event("flg_001", "m001", _ts(0), severity="warn")]
        decisions = [_decision_event("dec_001", "flg_001", "m001", _ts(50), action="alert_operator")]
        actions = [_action_event("act_001", "dec_001", "m001", _ts(60), action="alert_operator")]

        _write_jsonl(tmp_path / "telemetry.jsonl", telemetry)
        _write_jsonl(tmp_path / "flags.jsonl", flags)
        _write_jsonl(tmp_path / "decisions.jsonl", decisions)
        _write_jsonl(tmp_path / "actions.jsonl", actions)

        m = compute_track_metrics(
            track="B",
            telemetry_path=tmp_path / "telemetry.jsonl",
            flags_path=tmp_path / "flags.jsonl",
            decisions_path=tmp_path / "decisions.jsonl",
            actions_path=tmp_path / "actions.jsonl",
        )

        assert m.faults_injected == 0
        assert m.false_positives == 1
        assert m.false_positive_rate == pytest.approx(1.0)

    def test_empty_streams(self, tmp_path: Path):
        """All empty files — should return zero metrics without error."""
        for fname in ("telemetry.jsonl", "flags.jsonl", "decisions.jsonl", "actions.jsonl"):
            (tmp_path / fname).touch()

        m = compute_track_metrics(
            track="A",
            telemetry_path=tmp_path / "telemetry.jsonl",
            flags_path=tmp_path / "flags.jsonl",
            decisions_path=tmp_path / "decisions.jsonl",
            actions_path=tmp_path / "actions.jsonl",
        )

        assert m.faults_injected == 0
        assert m.flags_raised == 0
        assert m.total_actions == 0
        assert m.total_cost_usd == 0.0

    def test_reasoning_snapshots_captured(self, tmp_path: Path):
        """Track A reasoning snapshots are populated from decision events."""
        decisions = [
            _decision_event(f"dec_{i:03d}", f"flg_{i:03d}", "m001", _ts(i * 100), cost_usd=0.01)
            for i in range(6)
        ]
        _write_jsonl(tmp_path / "telemetry.jsonl", [])
        _write_jsonl(tmp_path / "flags.jsonl", [])
        _write_jsonl(tmp_path / "decisions.jsonl", decisions)
        _write_jsonl(tmp_path / "actions.jsonl", [])

        m = compute_track_metrics(
            track="A",
            telemetry_path=tmp_path / "telemetry.jsonl",
            flags_path=tmp_path / "flags.jsonl",
            decisions_path=tmp_path / "decisions.jsonl",
            actions_path=tmp_path / "actions.jsonl",
            max_snapshots=5,
        )

        # Should cap at max_snapshots
        assert len(m.reasoning_snapshots) <= 5

    def test_latency_computed(self, tmp_path: Path):
        """Median latency from flag to decision is computed correctly."""
        # Flag at t=0, decision at t=10 → 10s latency
        flags = [_flag_event("flg_001", "m001", _ts(0))]
        decisions = [_decision_event("dec_001", "flg_001", "m001", _ts(10))]

        _write_jsonl(tmp_path / "telemetry.jsonl", [])
        _write_jsonl(tmp_path / "flags.jsonl", flags)
        _write_jsonl(tmp_path / "decisions.jsonl", decisions)
        _write_jsonl(tmp_path / "actions.jsonl", [])

        m = compute_track_metrics(
            track="A",
            telemetry_path=tmp_path / "telemetry.jsonl",
            flags_path=tmp_path / "flags.jsonl",
            decisions_path=tmp_path / "decisions.jsonl",
            actions_path=tmp_path / "actions.jsonl",
        )

        assert m.median_latency_s == pytest.approx(10.0, abs=0.1)
