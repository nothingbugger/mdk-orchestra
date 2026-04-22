"""A/B experiment runner — public API and main orchestration loop.

Public API (matches shared/specs/interfaces.md §6):

    run_ab_experiment(scenario, duration_min, output_dir, seed) -> ABResults
    ABResults (dataclass)

Architecture
------------
Two tracks run as subprocess pairs (one pair per track), each with their
own temp directory tree under output_dir/<run_id>/track_{a,b}/stream/.

Track A: simulator + ingest + deterministic_tools + agents/maestro + action
Track B: simulator + ingest + deterministic_tools + track_b_mapper (rule-based)

Both tracks use the same simulator seed so fault scenarios are identical.
Their actions diverge because Track A routes through Maestro, Track B uses
a fixed severity→action mapping.

Implementation: subprocess-based (simpler than threads, avoids GIL / shared
state collisions). Each sub-process is driven by the module's existing
run_* loop functions. The runner:
  1. Launches all processes.
  2. Waits for duration_min * 60 seconds.
  3. Terminates all processes.
  4. Calls metrics.compute_ab_summary on the output dirs.
  5. Calls report_ab.generate_report.
  6. Returns ABResults.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from ab_experiment.metrics import ABMetricSummary, compute_ab_summary
from ab_experiment.report_ab import generate_report

_LOG = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Public API dataclass (matches interfaces.md §6)
# ---------------------------------------------------------------------------


@dataclass
class ABResults:
    """Comparative metrics returned by run_ab_experiment.

    Matches the dataclass spec in shared/specs/interfaces.md §6.
    """

    run_a_flags_raised: int
    run_b_flags_raised: int
    run_a_actions_taken: int
    run_b_actions_taken: int
    run_a_faults_caught_pre: int
    run_b_faults_caught_pre: int
    total_cost_usd: float
    cost_per_flag_usd: float
    per_agent_breakdown: dict[str, Any]
    report_path: str

    # Extended fields (beyond the spec minimum)
    run_id: str = ""
    scenario: str = ""
    figures: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Subprocess launcher helpers
# ---------------------------------------------------------------------------

# Python interpreter to use (same venv that launched us)
_PYTHON = sys.executable

# Simulated time per real second (10x acceleration by default)
_SIM_WALL_SPEED = 10.0
# Simulator tick interval (5 seconds simulated)
_SIM_TICK_S = 5.0


def _env_for_track(track_stream_dir: Path, track_memory_dir: Path, extra: dict) -> dict:
    """Build an OS env dict for a subprocess, pointing to the track's dirs."""
    env = os.environ.copy()
    env["MDK_STREAM_DIR"] = str(track_stream_dir)
    env["MDK_MEMORY_DIR"] = str(track_memory_dir)
    env.update(extra)
    return env


def _launch(
    module: str,
    args: list[str],
    env: dict,
    log_path: Path,
    cwd: Path,
) -> subprocess.Popen:
    """Launch `python -m <module> <args>` as a background process.

    Args:
        module: dotted module path (e.g. 'simulator.main').
        args: extra command-line args.
        env: environment dict.
        log_path: file to capture stdout+stderr.
        cwd: working directory for the subprocess.

    Returns:
        Running Popen handle.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [_PYTHON, "-m", module] + args
    log_fh = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=str(cwd),
    )
    _LOG.info("proc.launched", module=module, pid=proc.pid, log=str(log_path))
    return proc


def _terminate(procs: list[subprocess.Popen]) -> None:
    """Best-effort terminate all processes."""
    for proc in procs:
        if proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass

    # Give them a moment, then kill stragglers
    deadline = time.monotonic() + 5.0
    for proc in procs:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Track setup helpers
# ---------------------------------------------------------------------------


def _setup_track_dirs(run_dir: Path, track: str) -> tuple[Path, Path, Path]:
    """Create and return (stream_dir, memory_dir, logs_dir) for a track."""
    stream_dir = run_dir / f"track_{track}" / "stream"
    memory_dir = run_dir / f"track_{track}" / "memory"
    logs_dir = run_dir / f"track_{track}" / "logs"
    for d in (stream_dir, memory_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)
    return stream_dir, memory_dir, logs_dir


def _launch_track_a(
    run_dir: Path,
    seed: int,
    duration_s: float,
    n_miners: int,
    mock_mode: bool,
    cwd: Path,
    fault_mix: str = "random",
) -> list[subprocess.Popen]:
    """Launch all Track A subprocesses (full agent fleet).

    Returns list of Popen handles.
    """
    stream_dir, memory_dir, logs_dir = _setup_track_dirs(run_dir, "a")
    base_env = _env_for_track(
        stream_dir,
        memory_dir,
        {"MDK_AGENT_MOCK": "1" if mock_mode else "0"},
    )

    procs: list[subprocess.Popen] = []

    # 1. Simulator (uses --duration and --output matching simulator/main.py CLI)
    procs.append(
        _launch(
            "simulator.main",
            [
                f"--n-miners={n_miners}",
                f"--seed={seed}",
                f"--duration={int(duration_s)}",
                f"--speed={_SIM_WALL_SPEED}",
                f"--fault-mix={fault_mix}",
                f"--output={stream_dir / 'telemetry.jsonl'}",
            ],
            base_env,
            logs_dir / "simulator.log",
            cwd,
        )
    )

    # 2. Ingest (uses --input-stream, --kpi-output, --snap-output)
    procs.append(
        _launch(
            "ingest.main",
            [
                f"--input-stream={stream_dir / 'telemetry.jsonl'}",
                f"--kpi-output={stream_dir / 'kpis.jsonl'}",
                f"--snap-output={stream_dir / 'snapshots.jsonl'}",
            ],
            base_env,
            logs_dir / "ingest.log",
            cwd,
        )
    )

    # 3. Deterministic tools (uses --telemetry-stream, --flag-output)
    procs.append(
        _launch(
            "deterministic_tools.main",
            [
                f"--telemetry-stream={stream_dir / 'telemetry.jsonl'}",
                f"--flag-output={stream_dir / 'flags.jsonl'}",
                "--sensitivity=medium",
            ],
            base_env,
            logs_dir / "detector.log",
            cwd,
        )
    )

    # 4. Maestro (uses --flag-stream; decision output via MDK_STREAM_DIR)
    procs.append(
        _launch(
            "agents.main",
            [
                f"--flag-stream={stream_dir / 'flags.jsonl'}",
                "--from-start",
            ],
            base_env,
            logs_dir / "agents.log",
            cwd,
        )
    )

    # 5. Action executor (uses --decision-stream; from-start to catch all)
    procs.append(
        _launch(
            "action.main",
            [
                f"--decision-stream={stream_dir / 'decisions.jsonl'}",
                "--from-start",
            ],
            base_env,
            logs_dir / "action.log",
            cwd,
        )
    )

    return procs


def _launch_track_b(
    run_dir: Path,
    seed: int,
    duration_s: float,
    n_miners: int,
    cwd: Path,
    fault_mix: str = "random",
) -> list[subprocess.Popen]:
    """Launch all Track B subprocesses (deterministic-only baseline)."""
    stream_dir, memory_dir, logs_dir = _setup_track_dirs(run_dir, "b")
    base_env = _env_for_track(stream_dir, memory_dir, {"MDK_AGENT_MOCK": "1"})

    procs: list[subprocess.Popen] = []

    # 1. Simulator (same seed as Track A)
    procs.append(
        _launch(
            "simulator.main",
            [
                f"--n-miners={n_miners}",
                f"--seed={seed}",
                f"--duration={int(duration_s)}",
                f"--speed={_SIM_WALL_SPEED}",
                f"--fault-mix={fault_mix}",
                f"--output={stream_dir / 'telemetry.jsonl'}",
            ],
            base_env,
            logs_dir / "simulator.log",
            cwd,
        )
    )

    # 2. Ingest
    procs.append(
        _launch(
            "ingest.main",
            [
                f"--input-stream={stream_dir / 'telemetry.jsonl'}",
                f"--kpi-output={stream_dir / 'kpis.jsonl'}",
                f"--snap-output={stream_dir / 'snapshots.jsonl'}",
            ],
            base_env,
            logs_dir / "ingest.log",
            cwd,
        )
    )

    # 3. Deterministic tools
    procs.append(
        _launch(
            "deterministic_tools.main",
            [
                f"--telemetry-stream={stream_dir / 'telemetry.jsonl'}",
                f"--flag-output={stream_dir / 'flags.jsonl'}",
                "--sensitivity=medium",
            ],
            base_env,
            logs_dir / "detector.log",
            cwd,
        )
    )

    # 4. Track B mapper (rule-based, replaces Maestro)
    procs.append(
        _launch(
            "ab_experiment.track_b_runner",
            [
                f"--flags={stream_dir / 'flags.jsonl'}",
                f"--decisions={stream_dir / 'decisions.jsonl'}",
                f"--actions={stream_dir / 'actions.jsonl'}",
            ],
            base_env,
            logs_dir / "track_b_mapper.log",
            cwd,
        )
    )

    return procs


# ---------------------------------------------------------------------------
# Main experiment function (public API)
# ---------------------------------------------------------------------------


def run_ab_experiment(
    scenario: str,
    duration_min: int = 60,
    output_dir: str = "/run/mdk_fleet/ab_runs/",
    seed: int = 42,
    n_miners: int = 20,
    mock_mode: bool = True,
    fault_mix: str = "random",
) -> ABResults:
    """Run two simulations with identical seed, compare metrics.

    Args:
        scenario: scenario config name (used for output directory naming).
        duration_min: simulated duration in minutes. Real wall time =
            duration_min * 60 / SIM_WALL_SPEED.
        output_dir: root directory for A/B run outputs.
        seed: RNG seed — both tracks use this, so fault scenarios are identical.
        n_miners: number of miners per track (20 default for speed; 50 for full run).
        mock_mode: if True, sets MDK_AGENT_MOCK=1 (no real Claude API calls).

    Returns:
        ABResults dataclass with comparative metrics.
    """
    run_id = f"{scenario}_{uuid.uuid4().hex[:8]}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Project root (cwd for subprocesses so imports resolve)
    cwd = Path(__file__).resolve().parent.parent

    # Convert duration to simulated seconds
    duration_s = duration_min * 60.0
    # Real wall time for this run
    wall_duration_s = duration_s / _SIM_WALL_SPEED

    _LOG.info(
        "ab_experiment.start",
        run_id=run_id,
        scenario=scenario,
        seed=seed,
        duration_min=duration_min,
        wall_duration_s=round(wall_duration_s, 1),
        mock_mode=mock_mode,
        n_miners=n_miners,
        output=str(run_dir),
    )

    all_procs: list[subprocess.Popen] = []

    try:
        # Launch Track A (full fleet)
        track_a_procs = _launch_track_a(
            run_dir=run_dir,
            seed=seed,
            duration_s=duration_s,
            n_miners=n_miners,
            mock_mode=mock_mode,
            cwd=cwd,
            fault_mix=fault_mix,
        )
        all_procs.extend(track_a_procs)

        # Brief stagger so Track B doesn't race on shared kernel resources
        time.sleep(0.5)

        # Launch Track B (deterministic baseline)
        track_b_procs = _launch_track_b(
            run_dir=run_dir,
            seed=seed,
            duration_s=duration_s,
            n_miners=n_miners,
            cwd=cwd,
            fault_mix=fault_mix,
        )
        all_procs.extend(track_b_procs)

        _LOG.info(
            "ab_experiment.running",
            total_procs=len(all_procs),
            wall_duration_s=round(wall_duration_s, 1),
        )

        # Wait for the simulated duration (real wall time)
        time.sleep(wall_duration_s)

    finally:
        _LOG.info("ab_experiment.terminating", n_procs=len(all_procs))
        _terminate(all_procs)

    # Give file writes a moment to flush
    time.sleep(1.0)

    _LOG.info("ab_experiment.computing_metrics", run_dir=str(run_dir))

    # Compute metrics from event logs
    summary: ABMetricSummary = compute_ab_summary(
        run_dir=run_dir,
        scenario=scenario,
        seed=seed,
        duration_min=duration_min,
        run_id=run_id,
    )

    # Generate report artifacts
    artifacts = generate_report(summary, run_dir)

    # Build ABResults (matches interfaces.md §6 spec)
    a = summary.track_a
    b = summary.track_b
    cost_per_flag = (
        a.total_cost_usd / max(a.flags_raised, 1) if a.total_cost_usd > 0 else 0.0
    )

    results = ABResults(
        run_a_flags_raised=a.flags_raised,
        run_b_flags_raised=b.flags_raised,
        run_a_actions_taken=a.total_actions,
        run_b_actions_taken=b.total_actions,
        run_a_faults_caught_pre=a.faults_caught,
        run_b_faults_caught_pre=b.faults_caught,
        total_cost_usd=round(a.total_cost_usd, 6),
        cost_per_flag_usd=round(cost_per_flag, 6),
        per_agent_breakdown={
            "track_a_action_counts": a.action_counts,
            "track_b_action_counts": b.action_counts,
            "track_a_false_positives": a.false_positives,
            "track_b_false_positives": b.false_positives,
            "track_a_median_latency_s": round(a.median_latency_s, 3),
            "track_b_median_latency_s": round(b.median_latency_s, 3),
        },
        report_path=artifacts.get("ab_results_json", str(run_dir / "ab_results.json")),
        run_id=run_id,
        scenario=scenario,
        figures=artifacts,
    )

    _LOG.info(
        "ab_experiment.complete",
        run_id=run_id,
        track_a_caught=a.faults_caught,
        track_b_caught=b.faults_caught,
        total_cost_usd=round(a.total_cost_usd, 6),
        report=results.report_path,
    )

    return results
