"""Execute (or queue) actions decided by Maestro.

The executor is intentionally dumb: it translates an
`orchestrator_decision` into an `action_taken` event. What "executing"
means in the prototype is:

- **L1_observe**: log only. status=executed, action=observe.
- **L2_suggest**: alert the operator (log + status=executed with
  action=alert_operator). No fleet mutation.
- **L3_bounded_auto**: status=executed. If a fleet handle was provided
  at construction time, call `fleet_handle.apply_action(...)`. The
  simulator currently runs in a separate process, so the default
  out-of-process mode is log-only; the A/B experiment plugs a real
  handle when it runs simulator + executor in-process.
- **L4_human_only**: status=queued_for_human. Set
  `rollback_ts_scheduled` when the action has a natural expiry
  (throttle → now + duration_min).

Actions never bypass this module. Per Daniele's separation-of-concerns
rule, Maestro itself never touches the simulator — everything flows
through here (or through the equivalent deterministic mapping used by
the A/B Track B baseline).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

import structlog

from shared.event_bus import tail_events, write_event
from shared.paths import stream_paths
from shared.schemas.events import (
    ActionKind,
    ActionTaken,
    AutonomyLevel,
    OrchestratorDecision,
)

_LOG = structlog.get_logger(__name__)


class FleetHandle(Protocol):
    """Protocol for an in-process fleet mutator.

    The A/B runner builds an instance that wraps a live simulator
    `FleetState` and exposes `apply_action`. In the CLI / out-of-process
    path this is None and the executor logs only.
    """

    def apply_action(
        self, miner_id: str, action: str, params: dict[str, Any]
    ) -> None:  # pragma: no cover — interface
        ...


class Executor:
    """Translate decisions into action_taken envelopes."""

    def __init__(self, fleet_handle: FleetHandle | None = None) -> None:
        self.fleet_handle = fleet_handle

    def handle_decision(self, decision: OrchestratorDecision) -> ActionTaken:
        """Process one decision, emit action_taken, return it."""
        status, outcome_expected, rollback_ts = self._classify(decision)
        mutated = False
        if status == "executed" and self.fleet_handle is not None and _is_mutating(decision.action):
            try:
                self.fleet_handle.apply_action(
                    decision.miner_id, decision.action, decision.action_params
                )
                mutated = True
            except Exception as exc:  # noqa: BLE001 — log, don't crash the loop
                _LOG.error(
                    "fleet_apply_failed",
                    miner_id=decision.miner_id,
                    action=decision.action,
                    exc=str(exc),
                )
                status = "failed"

        action_taken = ActionTaken(
            action_id=f"act_{uuid.uuid4().hex[:12]}",
            decision_id=decision.decision_id,
            miner_id=decision.miner_id,
            action=decision.action,
            status=status,
            outcome_expected=outcome_expected,
            outcome_observed=None,
            rollback_ts_scheduled=rollback_ts,
        )
        write_event("action_taken", "action", action_taken)
        _LOG.info(
            "action_emitted",
            action_id=action_taken.action_id,
            decision_id=decision.decision_id,
            miner_id=decision.miner_id,
            action=decision.action,
            autonomy=decision.autonomy_level,
            status=status,
            mutated_fleet=mutated,
        )
        return action_taken

    # ------------------------------------------------------------------
    # classification
    # ------------------------------------------------------------------

    def _classify(
        self, decision: OrchestratorDecision
    ) -> tuple[str, str, datetime | None]:
        """Decide (status, outcome_expected, rollback_ts) from autonomy + action."""
        level: AutonomyLevel = decision.autonomy_level
        action: ActionKind = decision.action

        if level == "L4_human_only" or decision.pending_human_approval:
            return (
                "queued_for_human",
                f"Operator must approve {action} on {decision.miner_id} before it runs.",
                None,
            )

        if level == "L1_observe":
            return (
                "executed",
                f"Logged situation on {decision.miner_id}; no fleet change.",
                None,
            )

        if level == "L2_suggest":
            return (
                "executed",
                f"Operator alerted about {decision.miner_id}; no autonomous fleet change.",
                None,
            )

        # L3_bounded_auto — actual action. Describe the expected effect.
        params = decision.action_params or {}
        if action == "throttle":
            target = params.get("target_hashrate_pct", 0.80)
            minutes = params.get("duration_min", 60)
            rollback = datetime.now(tz=timezone.utc) + timedelta(minutes=int(minutes))
            expected = (
                f"Hashrate on {decision.miner_id} drops to {int(target * 100)}%, "
                f"temp falls 5–8°C, voltage stabilizes within 10 min. "
                f"Auto-rollback at {rollback.isoformat()}."
            )
            return ("executed", expected, rollback)

        if action == "migrate_workload":
            minutes = params.get("duration_min", 60)
            rollback = datetime.now(tz=timezone.utc) + timedelta(minutes=int(minutes))
            return (
                "executed",
                f"Workload migrated off {decision.miner_id} for {minutes} min.",
                rollback,
            )

        if action == "schedule_maintenance":
            return (
                "executed",
                f"Maintenance ticket queued for {decision.miner_id}.",
                None,
            )

        if action in ("alert_operator", "observe"):
            return (
                "executed",
                f"{action} recorded for {decision.miner_id}; no fleet change.",
                None,
            )

        # `shutdown` / `human_review` should have been L4; if we got here
        # it's a Maestro misclassification — emit rejected to surface it.
        return (
            "rejected",
            f"Action {action} requires L4 but came in at {level}. Rejected by executor.",
            None,
        )


def _is_mutating(action: ActionKind) -> bool:
    return action in ("throttle", "migrate_workload", "shutdown")


def apply_to_fleet(
    miner_id: str, action: str, params: dict[str, Any], fleet_handle: FleetHandle | None
) -> None:
    """Convenience shim used by the A/B runner when it wants to invoke
    the mutation path without building a full decision/envelope flow."""
    if fleet_handle is None:
        _LOG.debug("apply_to_fleet_noop", miner_id=miner_id, action=action)
        return
    fleet_handle.apply_action(miner_id, action, params)


def run_action_executor(
    decision_stream: str | Path | None = None,
    fleet_handle: FleetHandle | None = None,
    from_start: bool = False,
    stop_after: float | None = None,
    max_decisions: int | None = None,
) -> None:
    """Main loop. Tails decisions.jsonl, emits action_taken per decision."""
    path = Path(decision_stream) if decision_stream else stream_paths().decisions
    executor = Executor(fleet_handle=fleet_handle)
    processed = 0
    for env in tail_events(path, from_start=from_start, stop_after=stop_after):
        if env.event != "orchestrator_decision":
            continue
        decision = env.typed_data()  # type: ignore[assignment]
        if not isinstance(decision, OrchestratorDecision):  # defensive
            continue
        executor.handle_decision(decision)
        processed += 1
        if max_decisions is not None and processed >= max_decisions:
            return
