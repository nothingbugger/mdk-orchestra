"""Metric computation for the A/B experiment.

Reads the JSONL event streams produced by Track A and Track B and computes:
1. Detection: faults injected, faults caught (action taken within pre-fault window).
2. Action profile: action counts by type per track.
3. False positives: actions on miners with no fault within 24h.
4. Cost: total Claude API tokens + USD (track A only).
5. Latency: median flag-to-action time per track.
6. Qualitative: reasoning trace snapshots from track A.

All metric computation is post-hoc (reads complete event logs after the run).
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import structlog

from shared.event_bus import read_events
from shared.schemas.events import Envelope

_LOG = structlog.get_logger(__name__)

# How far back from a fault onset to count a catch (simulated time)
PRE_FAULT_CATCH_WINDOW_HOURS: float = 1.0
# How far forward from an action to count a false positive (simulated time)
FALSE_POSITIVE_WINDOW_HOURS: float = 24.0


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class TrackMetrics:
    """Computed metrics for one track (A or B)."""

    track: str  # "A" or "B"

    # Detection
    flags_raised: int = 0
    faults_injected: int = 0
    faults_caught: int = 0  # action taken in pre-fault window
    fault_catch_details: list[dict[str, Any]] = field(default_factory=list)

    # Action profile
    action_counts: dict[str, int] = field(default_factory=dict)
    total_actions: int = 0

    # False positives
    false_positives: int = 0
    false_positive_rate: float = 0.0

    # Cost (track A only; 0 for B)
    total_tokens_input: int = 0
    total_tokens_output: int = 0
    total_cost_usd: float = 0.0

    # Latency (flag_ts to action_ts), in seconds
    flag_to_action_latencies_s: list[float] = field(default_factory=list)
    median_latency_s: float = 0.0

    # Qualitative (track A only)
    reasoning_snapshots: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ABMetricSummary:
    """Combined metrics from both tracks."""

    track_a: TrackMetrics
    track_b: TrackMetrics
    run_id: str = ""
    scenario: str = ""
    duration_min: int = 0
    seed: int = 42


# ---------------------------------------------------------------------------
# Event log readers
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, return list of parsed dicts. Skips bad lines."""
    if not path.exists():
        return []
    results: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return results


def _parse_ts(ts_raw: Any) -> datetime | None:
    """Parse a timestamp field from an event envelope into a UTC datetime."""
    if ts_raw is None:
        return None
    if isinstance(ts_raw, datetime):
        return ts_raw.replace(tzinfo=timezone.utc) if ts_raw.tzinfo is None else ts_raw
    ts_str = str(ts_raw)
    if ts_str.endswith("Z"):
        ts_str = ts_str[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(ts_str)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Per-track analysis
# ---------------------------------------------------------------------------


def _collect_faults(telemetry_events: list[dict[str, Any]]) -> dict[str, list[datetime]]:
    """Scan telemetry events and find the *first* onset of each fault per miner.

    Returns: {miner_id: [first_fault_onset_ts, ...]}  (typically 0–1 per miner).
    """
    first_onset: dict[str, datetime | None] = {}

    for raw in telemetry_events:
        data = raw.get("data", {})
        fault_tag = data.get("fault_injected")
        if not fault_tag:
            continue
        miner_id = data.get("miner_id", "")
        ts = _parse_ts(raw.get("ts"))
        if ts is None:
            continue
        existing = first_onset.get(miner_id)
        if existing is None or ts < existing:
            first_onset[miner_id] = ts

    # Convert to list of timestamps per miner (filter out None)
    result: dict[str, list[datetime]] = {}
    for mid, ts in first_onset.items():
        if ts is not None:
            result[mid] = [ts]
    return result


def _collect_actions(action_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parse action_taken events into a structured list for analysis."""
    actions = []
    for raw in action_events:
        if raw.get("event") != "action_taken":
            continue
        data = raw.get("data", {})
        ts = _parse_ts(raw.get("ts"))
        if ts is None:
            continue
        actions.append(
            {
                "ts": ts,
                "miner_id": data.get("miner_id", ""),
                "action": data.get("action", "unknown"),
                "action_id": data.get("action_id", ""),
                "decision_id": data.get("decision_id", ""),
            }
        )
    return actions


def _collect_flags(flag_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parse flag_raised events into a structured list."""
    flags = []
    for raw in flag_events:
        if raw.get("event") != "flag_raised":
            continue
        data = raw.get("data", {})
        ts = _parse_ts(raw.get("ts"))
        if ts is None:
            continue
        flags.append(
            {
                "ts": ts,
                "flag_id": data.get("flag_id", ""),
                "miner_id": data.get("miner_id", ""),
                "severity": data.get("severity", "info"),
                "flag_type": data.get("flag_type", ""),
            }
        )
    return flags


def _collect_decisions(decision_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parse orchestrator_decision events for cost + latency extraction."""
    decisions = []
    for raw in decision_events:
        if raw.get("event") != "orchestrator_decision":
            continue
        data = raw.get("data", {})
        ts = _parse_ts(raw.get("ts"))
        if ts is None:
            continue
        decisions.append(
            {
                "ts": ts,
                "decision_id": data.get("decision_id", ""),
                "flag_id": data.get("flag_id", ""),
                "miner_id": data.get("miner_id", ""),
                "action": data.get("action", "unknown"),
                "total_cost_usd": float(data.get("total_cost_usd", 0.0)),
                "total_latency_ms": float(data.get("total_latency_ms", 0.0)),
                "reasoning_trace": data.get("reasoning_trace", ""),
                "consulted_agents": data.get("consulted_agents", []),
                "confidence": float(data.get("confidence", 0.0)),
            }
        )
    return decisions


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------


def compute_track_metrics(
    track: str,
    telemetry_path: Path,
    flags_path: Path,
    decisions_path: Path,
    actions_path: Path,
    max_snapshots: int = 5,
) -> TrackMetrics:
    """Compute all metrics for one track from its event log files.

    Args:
        track: "A" or "B".
        telemetry_path: telemetry.jsonl for this track.
        flags_path: flags.jsonl for this track.
        decisions_path: decisions.jsonl for this track.
        actions_path: actions.jsonl for this track.
        max_snapshots: max reasoning trace snapshots to capture (track A).

    Returns:
        Populated TrackMetrics.
    """
    m = TrackMetrics(track=track)

    # Load all event files
    telemetry_events = _load_jsonl(telemetry_path)
    flag_events = _load_jsonl(flags_path)
    decision_events = _load_jsonl(decisions_path)
    action_events = _load_jsonl(actions_path)

    _LOG.info(
        "metrics.loading",
        track=track,
        telemetry_count=len(telemetry_events),
        flags_count=len(flag_events),
        decisions_count=len(decision_events),
        actions_count=len(action_events),
    )

    # ---- parse events ----
    faults_by_miner = _collect_faults(telemetry_events)
    actions = _collect_actions(action_events)
    flags = _collect_flags(flag_events)
    decisions = _collect_decisions(decision_events)

    m.flags_raised = len(flags)
    m.faults_injected = len(faults_by_miner)

    # ---- action profile ----
    m.total_actions = len(actions)
    for act in actions:
        key = act["action"]
        m.action_counts[key] = m.action_counts.get(key, 0) + 1

    # ---- fault detection: catch = action on miner within pre-fault window ----
    window = timedelta(hours=PRE_FAULT_CATCH_WINDOW_HOURS)
    caught_miners: set[str] = set()

    for miner_id, onset_tss in faults_by_miner.items():
        for onset_ts in onset_tss:
            window_start = onset_ts - window
            # Any action on this miner in [onset - window, onset + epsilon]?
            for act in actions:
                if act["miner_id"] != miner_id:
                    continue
                if window_start <= act["ts"] <= onset_ts + timedelta(hours=1):
                    caught_miners.add(miner_id)
                    m.fault_catch_details.append(
                        {
                            "miner_id": miner_id,
                            "fault_onset_ts": onset_ts.isoformat(),
                            "action_ts": act["ts"].isoformat(),
                            "action": act["action"],
                            "lead_time_min": (onset_ts - act["ts"]).total_seconds() / 60,
                        }
                    )
                    break

    m.faults_caught = len(caught_miners)

    # ---- false positives: actions on miners with no fault within 24h ----
    fp_window = timedelta(hours=FALSE_POSITIVE_WINDOW_HOURS)
    false_positive_count = 0

    for act in actions:
        if act["action"] in ("observe",):
            # observe-only actions don't count as false positives
            continue
        miner_id = act["miner_id"]
        act_ts = act["ts"]
        fault_tss = faults_by_miner.get(miner_id, [])
        # Is there a fault within 24h of this action?
        has_nearby_fault = any(
            abs((f - act_ts).total_seconds()) <= fp_window.total_seconds()
            for f in fault_tss
        )
        if not has_nearby_fault:
            false_positive_count += 1

    m.false_positives = false_positive_count
    m.false_positive_rate = (
        false_positive_count / m.total_actions if m.total_actions > 0 else 0.0
    )

    # ---- cost (track A decisions carry cost_usd) ----
    m.total_cost_usd = sum(d["total_cost_usd"] for d in decisions)

    # ---- latency: flag_ts to decision_ts ----
    # Build flag_id -> flag_ts index
    flag_ts_by_id = {f["flag_id"]: f["ts"] for f in flags}

    latencies: list[float] = []
    for dec in decisions:
        fts = flag_ts_by_id.get(dec["flag_id"])
        if fts is not None:
            latency_s = (dec["ts"] - fts).total_seconds()
            if latency_s >= 0:
                latencies.append(latency_s)

    m.flag_to_action_latencies_s = latencies
    m.median_latency_s = statistics.median(latencies) if latencies else 0.0

    # ---- qualitative: reasoning snapshots (track A only) ----
    if track == "A":
        real_decisions = [
            d for d in decisions if d.get("reasoning_trace") and d["total_cost_usd"] > 0
        ]
        if not real_decisions:
            real_decisions = decisions  # mock mode — include anyway
        for dec in real_decisions[:max_snapshots]:
            m.reasoning_snapshots.append(
                {
                    "miner_id": dec["miner_id"],
                    "action": dec["action"],
                    "confidence": dec["confidence"],
                    "consulted_agents": dec["consulted_agents"],
                    "reasoning_trace": dec["reasoning_trace"][:500],
                    "cost_usd": dec["total_cost_usd"],
                    "latency_ms": dec["total_latency_ms"],
                }
            )

    _LOG.info(
        "metrics.computed",
        track=track,
        flags_raised=m.flags_raised,
        faults_injected=m.faults_injected,
        faults_caught=m.faults_caught,
        total_actions=m.total_actions,
        false_positives=m.false_positives,
        total_cost_usd=round(m.total_cost_usd, 4),
        median_latency_s=round(m.median_latency_s, 3),
    )
    return m


def compute_ab_summary(
    run_dir: Path,
    scenario: str,
    seed: int,
    duration_min: int,
    run_id: str,
) -> ABMetricSummary:
    """Compute metrics for both tracks from a completed A/B run directory.

    Expects:
        run_dir/track_a/stream/{telemetry.jsonl, flags.jsonl, decisions.jsonl, actions.jsonl}
        run_dir/track_b/stream/{...}

    Args:
        run_dir: root directory for this A/B run.
        scenario: scenario name (for labeling).
        seed: simulator seed used.
        duration_min: configured run duration.
        run_id: unique run identifier.

    Returns:
        ABMetricSummary with both tracks populated.
    """
    track_a_stream = run_dir / "track_a" / "stream"
    track_b_stream = run_dir / "track_b" / "stream"

    track_a = compute_track_metrics(
        track="A",
        telemetry_path=track_a_stream / "telemetry.jsonl",
        flags_path=track_a_stream / "flags.jsonl",
        decisions_path=track_a_stream / "decisions.jsonl",
        actions_path=track_a_stream / "actions.jsonl",
    )

    track_b = compute_track_metrics(
        track="B",
        telemetry_path=track_b_stream / "telemetry.jsonl",
        flags_path=track_b_stream / "flags.jsonl",
        decisions_path=track_b_stream / "decisions.jsonl",
        actions_path=track_b_stream / "actions.jsonl",
    )

    return ABMetricSummary(
        track_a=track_a,
        track_b=track_b,
        run_id=run_id,
        scenario=scenario,
        duration_min=duration_min,
        seed=seed,
    )
