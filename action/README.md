# `action/` — Action Executor

Layer 4 of MDK Fleet. Consumes `orchestrator_decision` events, emits `action_taken`.

## What it does

Per Maestro's separation-of-concerns rule (Daniele, 2026-04-20):

> Maestro never reaches into the simulator. All miner-facing commands
> pass through `action/executor.py → simulator.fleet.apply_action`.

The executor:

1. Tails `events/decisions.jsonl`.
2. For each `orchestrator_decision`, decides `action_taken.status` based on `autonomy_level`:
   - `L1_observe` → `status=executed`, no fleet mutation, log only.
   - `L2_suggest` → `status=executed`, `action=alert_operator`, no fleet mutation.
   - `L3_bounded_auto` → `status=executed`. If a fleet handle is wired (A/B in-process mode), call `fleet_handle.apply_action(miner_id, action, params)`. Otherwise log-only (prototype default).
   - `L4_human_only` → `status=queued_for_human`, no mutation, no rollback (operator will dispatch).
3. Computes `outcome_expected` (free-form description for the dashboard) and `rollback_ts_scheduled` (for reversible L3 actions).
4. Emits `action_taken` to the bus.

## Running

```bash
# Out-of-process (simulator runs elsewhere; actions are simulated / logged):
python -m action.main

# With replay of existing decisions:
python -m action.main --from-start --stop-after 60
```

## In-process integration (A/B runner)

```python
from action.executor import Executor

# ab_experiment builds this
class SimFleetAdapter:
    def __init__(self, fleet_state):
        self.fleet = fleet_state

    def apply_action(self, miner_id, action, params):
        if action == "throttle":
            target = params.get("target_hashrate_pct", 0.80)
            miner = self.fleet.miners[miner_id]
            miner.operating_mode = "eco" if target <= 0.85 else "balanced"
        elif action == "shutdown":
            self.fleet.miners[miner_id].online = False
        # etc.

executor = Executor(fleet_handle=SimFleetAdapter(fleet_state))
```

## Design choices

- **Out-of-process default = log only.** The CLI entry point runs without a fleet handle because the simulator, ingest, detector, and maestro all run as separate processes. "Simulated actions" in this mode = an entry in `events/actions.jsonl` that the dashboard renders. No real miner is mutated in the prototype anyway.
- **Rollback scheduling is on the executor**, not on Maestro. Maestro describes what it wants (`throttle 80% for 60min`); the executor turns that into a concrete `rollback_ts_scheduled` which the dashboard and A/B runner can consume.
- **L3 with a non-reversible action (`shutdown`) is rejected** with `status=rejected`. Maestro should have escalated to L4 — this is a safety backstop.

## Not here

- Actual fleet mutation (that's `FleetHandle.apply_action` — optional, provided by callers).
- Human-approval UI (that's the dashboard's `/decisions` page with the L4 queue).
- Rollback execution (future work — a timer loop that flips `status` back when `rollback_ts_scheduled` fires).
