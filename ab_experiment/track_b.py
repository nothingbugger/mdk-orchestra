"""Track B — deterministic rule-based action mapper (no Maestro, no LLMs).

Implements the baseline for the A/B comparison:
- info  flag -> observe
- warn  flag -> alert_operator
- crit  flag -> throttle (to 80%, 60 min)

No specialist consultation, no LLM calls, zero API cost.
The mapper tails the flags stream and emits orchestrator_decision +
action_taken events, so metrics code can treat both tracks symmetrically.
"""

from __future__ import annotations

import itertools
import time
import uuid
from pathlib import Path
from typing import Any

import structlog

from shared.event_bus import tail_events, write_event
from shared.paths import stream_paths
from shared.schemas.events import ActionKind, ActionTaken, AutonomyLevel, OrchestratorDecision

_LOG = structlog.get_logger(__name__)

# ---- rule table -----------------------------------------------------------

# Maps severity -> (action, autonomy_level, action_params)
_SEVERITY_RULES: dict[str, tuple[str, str, dict[str, Any]]] = {
    "info": ("observe", "L1_observe", {}),
    "warn": ("alert_operator", "L2_suggest", {}),
    "crit": ("throttle", "L3_bounded_auto", {"target_hashrate_pct": 0.80, "duration_min": 60}),
}

_action_counter = itertools.count(1)
_decision_counter = itertools.count(1)


def _next_decision_id() -> str:
    return f"dec_b_{next(_decision_counter):05d}"


def _next_action_id() -> str:
    return f"act_b_{next(_action_counter):05d}"


# ---- outcome strings (mirrors action/executor.py) -------------------------


def _outcome_expected(action: str, miner_id: str, params: dict[str, Any]) -> str:
    if action == "throttle":
        target = params.get("target_hashrate_pct", 0.80)
        minutes = params.get("duration_min", 60)
        return (
            f"[track_b] Hashrate on {miner_id} drops to {int(target * 100)}%, "
            f"temp falls, voltage stabilizes within {minutes} min."
        )
    if action == "alert_operator":
        return f"[track_b] Operator alerted about {miner_id}; no autonomous fleet change."
    return f"[track_b] {action} logged for {miner_id}."


# ---- fleet handle integration ---------------------------------------------


def apply_rule_action(
    flag: dict[str, Any],
    flag_ts: float,
    fleet_handle: Any | None,
    decisions_path: Path,
    actions_path: Path,
) -> None:
    """Map one flag to a deterministic action and emit decision + action events.

    Args:
        flag: the flag_raised.data dict.
        flag_ts: wall-clock time the flag was processed (for latency tracking).
        fleet_handle: optional FleetStateAdapter to mutate the sim in-process.
        decisions_path: stream file for orchestrator_decision events (track B).
        actions_path: stream file for action_taken events (track B).
    """
    severity = flag.get("severity", "info")
    action, autonomy, params = _SEVERITY_RULES.get(severity, ("observe", "L1_observe", {}))
    miner_id: str = flag["miner_id"]
    flag_id: str = flag["flag_id"]

    decision = OrchestratorDecision(
        decision_id=_next_decision_id(),
        flag_id=flag_id,
        miner_id=miner_id,  # type: ignore[arg-type]
        action=action,  # type: ignore[arg-type]
        action_params=params,
        autonomy_level=autonomy,  # type: ignore[arg-type]
        confidence=1.0,  # deterministic — no uncertainty
        reasoning_trace=(
            f"[track_b_rule] flag={flag.get('flag_type')} severity={severity} "
            f"→ rule action={action} (no LLM consultation)"
        ),
        consulted_agents=[],
        total_cost_usd=0.0,
        total_latency_ms=(time.monotonic() - flag_ts) * 1000.0,
        pending_human_approval=False,
    )

    write_event(
        "orchestrator_decision",
        "orchestrator",
        decision,
        stream_path=decisions_path,
        also_live=False,
    )

    # Apply fleet mutation if handle present (throttle only)
    if fleet_handle is not None and action == "throttle":
        try:
            fleet_handle.apply_action(miner_id, action, params)
        except Exception as exc:
            _LOG.error("track_b.fleet_apply_failed", miner_id=miner_id, exc=str(exc))

    action_taken = ActionTaken(
        action_id=_next_action_id(),
        decision_id=decision.decision_id,
        miner_id=miner_id,  # type: ignore[arg-type]
        action=action,  # type: ignore[arg-type]
        status="executed",
        outcome_expected=_outcome_expected(action, miner_id, params),
        outcome_observed=None,
        rollback_ts_scheduled=None,
    )

    write_event(
        "action_taken",
        "action",
        action_taken,
        stream_path=actions_path,
        also_live=False,
    )

    _LOG.info(
        "track_b.action_emitted",
        flag_id=flag_id,
        miner_id=miner_id,
        severity=severity,
        action=action,
    )


def run_track_b_mapper(
    flags_path: Path,
    decisions_path: Path,
    actions_path: Path,
    fleet_handle: Any | None = None,
    stop_when: Any = None,
) -> None:
    """Tail flags stream and apply deterministic rules for Track B.

    Runs until `stop_when()` returns True (for A/B runner integration).

    Args:
        flags_path: track B flags JSONL file.
        decisions_path: where to write orchestrator_decision events (track B).
        actions_path: where to write action_taken events (track B).
        fleet_handle: optional FleetStateAdapter to apply throttle in-process.
        stop_when: callable() -> bool — stop loop when True.
    """
    _LOG.info(
        "track_b.mapper.starting",
        flags=str(flags_path),
        decisions=str(decisions_path),
        actions=str(actions_path),
    )

    for env in tail_events(flags_path, from_start=True, stop_when=stop_when):
        if env.event != "flag_raised":
            continue
        flag_ts = time.monotonic()
        try:
            apply_rule_action(
                flag=env.data,
                flag_ts=flag_ts,
                fleet_handle=fleet_handle,
                decisions_path=decisions_path,
                actions_path=actions_path,
            )
        except Exception as exc:
            _LOG.error("track_b.mapper_error", error=str(exc))
