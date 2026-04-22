"""FleetHandle adapter — connects the action executor to a live FleetState.

The action executor (action/executor.py) expects a FleetHandle protocol with
`apply_action(miner_id, action, params)`. This module provides a concrete
implementation that mutates a simulator FleetState in-process.

For the A/B runner, each track gets its own FleetStateAdapter wrapping its
own FleetState instance, so Track A and Track B simulator states diverge only
from agent decisions, not shared state.
"""

from __future__ import annotations

from typing import Any

import structlog

from simulator.miner_sim import MinerState

_LOG = structlog.get_logger(__name__)

# Throttle target applied when action=throttle (if not specified in params)
_DEFAULT_THROTTLE_PCT: float = 0.80


class FleetStateAdapter:
    """Wraps a simulator.fleet_sim.FleetState and implements the FleetHandle protocol.

    Args:
        fleet: the FleetState to mutate. Must be the *same* FleetState instance
               that the simulator loop is advancing.
    """

    def __init__(self, fleet: Any) -> None:
        self._fleet = fleet  # simulator.fleet_sim.FleetState
        # Index miners by id for O(1) lookup
        self._miners: dict[str, MinerState] = {m.miner_id: m for m in fleet.miners}

    def apply_action(self, miner_id: str, action: str, params: dict[str, Any]) -> None:
        """Mutate the FleetState in response to an action decision.

        Supported actions with fleet-side effects:
            throttle   -> set miner operating_mode to 'eco' (nearest sim proxy for throttle).
                          If target_hashrate_pct >= 0.90 → stays 'balanced'.
                          If < 0.70 → 'eco'. Otherwise 'eco'.
            shutdown   -> set fault_type to 'shutdown' so the sim reports the miner as halted.
            observe / alert_operator / etc. -> no-op (logged only).

        Args:
            miner_id: e.g. 'm042'.
            action: action string from ActionKind.
            params: action-specific params (e.g. {'target_hashrate_pct': 0.80}).
        """
        miner = self._miners.get(miner_id)
        if miner is None:
            _LOG.warning("fleet_adapter.unknown_miner", miner_id=miner_id, action=action)
            return

        if action == "throttle":
            target_pct = float(params.get("target_hashrate_pct", _DEFAULT_THROTTLE_PCT))
            new_mode = "eco" if target_pct < 0.90 else "balanced"
            old_mode = miner.operating_mode
            miner.operating_mode = new_mode  # type: ignore[assignment]
            _LOG.info(
                "fleet_adapter.throttle_applied",
                miner_id=miner_id,
                from_mode=old_mode,
                to_mode=new_mode,
                target_pct=target_pct,
            )

        elif action == "shutdown":
            # Mark as shutdown — the miner still exists in the fleet but we
            # flag it so metrics can count it as a "stop" action.
            miner.fault_type = miner.fault_type or "shutdown"
            miner.operating_mode = "eco"  # type: ignore[assignment]
            _LOG.info("fleet_adapter.shutdown_applied", miner_id=miner_id)

        else:
            # observe, alert_operator, migrate_workload, schedule_maintenance,
            # human_review — no physical state change.
            _LOG.debug("fleet_adapter.no_op_action", miner_id=miner_id, action=action)
