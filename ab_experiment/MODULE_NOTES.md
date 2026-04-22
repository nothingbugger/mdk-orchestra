# ab_experiment — Module Notes

**Author**: Claude Code Sonnet (feat/ab_experiment branch)
**Date**: 2026-04-20
**Status**: complete + smoke-tested

## What was built

A/B experiment validation module for MDK Fleet. Runs two parallel tracks against
the same simulator seed and computes comparative metrics.

### Architecture choice: subprocess model

The spec offers two options: in-process threads or subprocesses. We chose
**subprocesses** because:
- Avoids shared-state bugs (each track has its own Python interpreter, no GIL contention)
- Debuggable via per-track log files under `track_{a,b}/logs/`
- Mirrors how the actual live system runs (each module as a daemon process)
- Simpler to reason about at the cost of slightly more startup latency

Each track runs 5 subprocesses (Track A) or 4 (Track B, no Maestro):
- `simulator.main` — identical seed per track, isolated stream dir
- `ingest.main` — tails track-local telemetry
- `deterministic_tools.main` — emits flags to track-local stream
- `agents.main` (A only) / `ab_experiment.track_b_runner` (B) — decision layer
- `action.main` (A only) — action execution

### `fleet_adapter.py` — apply_action

The spec requires a `FleetHandle` adapter. The executor module (`action/executor.py`)
already defines the `FleetHandle` protocol. We implemented `FleetStateAdapter` in
`fleet_adapter.py`, which wraps a live `FleetState` and mutates `miner.operating_mode`
on `throttle` and sets `miner.fault_type` on `shutdown`.

In subprocess mode (the actual runner), the fleet handle is `None` (executor logs
only, no in-process mutation). The adapter is used in test/in-process mode.
**We did NOT modify `simulator/fleet_sim.py`** — the `apply_action` interface is
fully implemented in `ab_experiment/fleet_adapter.py` as a wrapper.

### Track B rule mapping (spec-compliant)

Per spec:
- `info` → `observe`
- `warn` → `alert_operator`
- `crit` → `throttle` (80%, 60 min)

The mapper emits `orchestrator_decision` + `action_taken` events (zero cost, zero
LLM latency) so the metrics layer treats both tracks symmetrically.

### Metrics computation

Post-hoc: reads completed JSONL streams after run terminates. Key design choices:

- **Fault catch criterion**: any non-observe action on a miner within
  `PRE_FAULT_CATCH_WINDOW_HOURS=1` hour before fault onset → catch.
- **False positive**: action on miner with no fault within `FALSE_POSITIVE_WINDOW_HOURS=24` hours.
- **Latency**: `flag_ts` → `decision_ts` (not action_ts, since decisions trigger actions).

### Figures

Four matplotlib figures, all styled with `shared/design/tokens.matplotlib_rcparams()`:
1. `action_timeline.png` — bar chart of action counts per type, both tracks
2. `catch_rate.png` — side-by-side bars: faults injected / caught / false positives
3. `cost_vs_catch.png` — scatter: cost vs catch rate
4. `reasoning_trace.png` — card rendering of one A track reasoning trace

Figures fail gracefully if matplotlib is missing.

## Deviations from spec

1. **`--duration-s` vs `--duration`**: the simulator's actual CLI uses `--duration`
   (simulated seconds), not `--duration-s`. Runner calls it correctly.

2. **`--decision-output` not in agents/main.py**: agents/main.py uses `MDK_STREAM_DIR`
   to route decisions, not a CLI arg. Runner sets `MDK_STREAM_DIR` per-track, which
   routes decisions to the correct track's `decisions.jsonl`. Works correctly.

3. **`n_miners` default = 20**: reduced from 50 for smoke run speed. Full run
   can use `--n-miners 50`.

4. **Simulated time tracking**: the fault-catch window uses the simulated timestamps
   embedded in event envelopes (`ts` field), not wall-clock time. This is correct —
   faults are scheduled in simulated time.

5. **In-process fleet mutation**: not used in subprocess mode. The FleetStateAdapter
   exists for future in-process mode or tests, but the runner's subprocesses run
   isolated. Track B's throttle actions are logged but don't mutate the Track B
   simulator (which runs as a separate process). This is acceptable — the simulator
   fault timings are pre-scheduled by seed and don't change based on actions.

## Running notes

- `--duration-min 1` → 6 real seconds (10x speed factor)
- `--duration-min 60` → 6 real minutes (10x speed factor)
- All subprocess logs under `<run_dir>/track_{a,b}/logs/*.log` — tail these for debugging
- `MDK_AGENT_MOCK=1` is set automatically unless `--api-mode` is passed

## Tests

- `tests/test_ab_metrics.py` — unit tests for metric computation (no subprocesses, fast)
- `tests/test_ab_smoke.py` — integration smoke (spawns real subprocesses, ~10s)
  - Marked `@pytest.mark.slow` — run with `pytest -m slow tests/test_ab_smoke.py`
  - The `test_ab_results_json_structure_only` test is fast (no subprocesses)
