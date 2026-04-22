"""Smoke test: 1-minute mock A/B run — asserts ab_results.json exists and has both tracks.

This test spawns real subprocesses (simulator, ingest, detector, agents, action executor,
track_b mapper) and runs them for 1 simulated minute at 10x speed (6 real seconds).
It does NOT make real Claude API calls (MDK_AGENT_MOCK=1).

Marked slow — skip in CI unless --run-slow is passed.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

# Mark this test as slow (subprocess + real I/O)
pytestmark = pytest.mark.slow


@pytest.fixture
def tmp_run_dir(tmp_path: Path) -> Path:
    """Provide a writable temp directory for the A/B run."""
    run_dir = tmp_path / "ab_runs"
    run_dir.mkdir()
    return run_dir


def test_ab_smoke_run(tmp_run_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """Run a 1-minute mock A/B experiment and verify ab_results.json has both tracks."""
    # Set mock mode
    monkeypatch.setenv("MDK_AGENT_MOCK", "1")

    # Import the public runner API
    from ab_experiment.runner import run_ab_experiment

    # 1-minute simulated run: real wall time = 60s / 10x = 6 seconds
    results = run_ab_experiment(
        scenario="smoke",
        duration_min=1,
        output_dir=str(tmp_run_dir),
        seed=42,
        n_miners=5,  # minimal miners for speed
        mock_mode=True,
    )

    # --- ab_results.json must exist ---
    report_path = Path(results.report_path)
    assert report_path.exists(), f"ab_results.json not found at {report_path}"

    with report_path.open("r") as f:
        data = json.load(f)

    # --- Schema checks ---
    assert "track_a" in data, "ab_results.json missing 'track_a'"
    assert "track_b" in data, "ab_results.json missing 'track_b'"
    assert "comparison" in data, "ab_results.json missing 'comparison'"

    # Both tracks should have the core metric fields
    for track_key in ("track_a", "track_b"):
        track = data[track_key]
        assert "flags_raised" in track, f"{track_key} missing flags_raised"
        assert "faults_injected" in track, f"{track_key} missing faults_injected"
        assert "faults_caught" in track, f"{track_key} missing faults_caught"
        assert "total_actions" in track, f"{track_key} missing total_actions"
        assert "action_counts" in track, f"{track_key} missing action_counts"
        assert isinstance(track["flags_raised"], int)
        assert isinstance(track["faults_injected"], int)

    # Track B should have zero API cost
    assert data["track_b"]["total_cost_usd"] == 0.0, "Track B should have no API cost"

    # ABResults dataclass fields
    assert results.run_a_flags_raised >= 0
    assert results.run_b_flags_raised >= 0
    assert results.run_a_actions_taken >= 0
    assert results.run_b_actions_taken >= 0
    assert results.total_cost_usd >= 0.0
    assert results.cost_per_flag_usd >= 0.0
    assert isinstance(results.per_agent_breakdown, dict)
    assert results.report_path != ""


def test_ab_results_json_structure_only(tmp_run_dir: Path):
    """Lightweight test: verify ab_results.json generation without subprocess runs.

    Creates a synthetic ABMetricSummary and checks the JSON output format.
    This runs fast (no subprocesses).
    """
    from ab_experiment.metrics import ABMetricSummary, TrackMetrics
    from ab_experiment.report_ab import write_results_json

    # Build synthetic metrics
    track_a = TrackMetrics(
        track="A",
        flags_raised=10,
        faults_injected=3,
        faults_caught=2,
        action_counts={"throttle": 4, "alert_operator": 3, "observe": 3},
        total_actions=10,
        false_positives=2,
        false_positive_rate=0.2,
        total_cost_usd=0.15,
        median_latency_s=2.5,
        reasoning_snapshots=[
            {
                "miner_id": "m001",
                "action": "throttle",
                "confidence": 0.85,
                "consulted_agents": ["voltage_agent"],
                "reasoning_trace": "Voltage drift confirmed by voltage_agent.",
                "cost_usd": 0.02,
                "latency_ms": 1800.0,
            }
        ],
    )

    track_b = TrackMetrics(
        track="B",
        flags_raised=10,
        faults_injected=3,
        faults_caught=1,
        action_counts={"throttle": 3, "alert_operator": 4, "observe": 3},
        total_actions=10,
        false_positives=3,
        false_positive_rate=0.3,
        total_cost_usd=0.0,
        median_latency_s=0.01,
    )

    summary = ABMetricSummary(
        track_a=track_a,
        track_b=track_b,
        run_id="smoke_test_abc123",
        scenario="smoke",
        duration_min=1,
        seed=42,
    )

    out_path = write_results_json(summary, tmp_run_dir)
    assert out_path.exists()

    with out_path.open() as f:
        data = json.load(f)

    assert data["run_id"] == "smoke_test_abc123"
    assert data["track_a"]["faults_caught"] == 2
    assert data["track_b"]["faults_caught"] == 1
    assert data["track_b"]["total_cost_usd"] == 0.0
    assert "catch_rate_delta" in data["comparison"]
