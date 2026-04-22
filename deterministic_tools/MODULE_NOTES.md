# MODULE_NOTES — deterministic_tools

Author: Claude Code (Sonnet 4.6), feat/detector branch  
Date: 2026-04-20

---

## What was built

### Fully working (immediately active)

- **`RuleEngineFlagger`** — hard-threshold rules for all 4 fault signatures
  (thermal_runaway, voltage_drift, hashrate_degradation, fan_anomaly). Uses
  `_SustainedTracker` for accurate "X minutes continuously" semantics. Per
  (miner_id, flag_type) cooldown prevents flag storms. Activates on tick 1.

- **`sensitivity.yaml` + `config.py`** — three named sensitivity profiles (low,
  medium, high). All timing windows and confidence thresholds are controlled
  through the yaml; no magic numbers in flagger code.

- **`runner.py` (`run_detector`)** — main loop, tails `telemetry.jsonl`, runs all
  three flaggers per tick, emits `FlagRaised` via `write_event`. Separate daemon
  thread for KPI stream tailing (non-blocking; degrades gracefully if ingest is
  not yet running).

- **`main.py`** — `python -m deterministic_tools.main` entry point with argparse.

- **`base.py`** — `Flagger` Protocol (runtime_checkable), `FlagResult` dataclass,
  `MinerHistory` per-miner rolling state. Matches `interfaces.md §3` exactly.

- **Tests** — three test files covering rule-engine thresholds (just
  below/above), voltage band logic, cooldown behavior, flagger protocol
  conformance, schema round-trip (FlagRaised → JSON → parse_event), MinerHistory
  bounded eviction, ML flagger activate/inactive states.

### Shipped but training-required

- **`IsolationForestFlagger`** — complete pipeline including bootstrap
  accumulation, model fitting, persistence to `models/if_v2.pkl`, and reload on
  startup. If the pkl does not exist, the flagger accumulates bootstrap ticks
  autonomously during the live run. It will activate automatically once 1440
  clean ticks have been collected (~2 h of fleet-wide data at 5 s/tick for 50
  miners). Can also be trained offline via `train.py`.

- **`XGBoostFlagger`** — complete inference pipeline; disabled until
  `models/xgb_predictor.pkl` exists. The training pipeline in `train.py`
  collects labelled look-ahead windows and fits the model.

- **`train.py`** — `python -m deterministic_tools.train` entry point. Replays
  (or tails live) the telemetry stream to collect bootstrap data for both ML
  models.

---

## Design decisions

### Fleet-wide Isolation Forest vs per-miner

Chose **fleet-wide**. Reasoning:
- All 50 miners are the same model (S19j Pro) on the same site. The feature
  distribution is homogeneous; a global model captures the normal operating
  envelope well.
- Per-miner requires ~120 individual bootstrap samples per miner before fitting
  (10 min at 5 s/tick). Global needs 1440 samples total — same time, but
  simpler and one pkl.
- The per-miner manufacturing variance (voltage/temp/hashrate bias) is small
  relative to fault-induced deviations.

Downside: a miner with systematically higher temps (e.g. a hotter rack slot)
may get flagged as anomalous even when healthy. Mitigated by the IF
`contamination=0.05` setting and confidence thresholds in `sensitivity.yaml`.

### XGBoost label: 30-min look-ahead via fault_injected

`fault_injected` is read **only** in `train.py` during label construction. The
inference path (`xgboost_flagger.py`'s `evaluate()`) accepts only
`TelemetryTick` + `MinerHistory` — no access to `fault_injected`. This is
asserted in code and documented prominently with inline comments.

### Sustained-condition tracking (SustainedTracker)

`_SustainedTracker` resets the streak to zero on any single False sample. This
is deliberately strict — a brief dip below the threshold resets the clock.
Alternative: exponential moving average. Chose streak for correctness with the
spec language ("sustained for X minutes").

### KPI stream tailing (background thread)

The KPI stream is tailed in a daemon thread. If `ingest` is not running,
`kpis.jsonl` will be empty or missing — the thread will silently wait.
`MinerHistory.last_kpi()` returns None in this case, and both ML flaggers
fall back to HSI=0.0, TE=50.0 defaults. Correct behavior; no crash.

### Flag priority within a tick

When multiple rules fire in the same tick, `run_detector` emits ALL of them
(one per flagger per tick, up to 3 per miner). Each flagger's cooldown is
tracked independently. This is intentional — the rule engine and IF may raise
flags for different aspects of the same tick; the orchestrator should see both.

### Flag ID format

`flg_NNNNN` — sequential per-process-run. Not globally unique across restarts.
If the system is long-running and flags are persisted across restarts, callers
should use `(ts, flag_id)` as the dedup key.

---

## Deviations from spec

None that break the contract. Minor notes:

1. **`interfaces.md §3` `run_detector` signature** lists `input_stream` and
   `flag_output` as `str` types. Our implementation also accepts `None` (uses
   canonical paths from `shared.paths`). Backward-compatible.

2. **`FlagEvidence`** in `shared/schemas/events.py` has `model_config =
   ConfigDict(extra="allow")`, which lets flaggers include flagger-specific
   evidence fields beyond `metric` and `window_min`. This was used to carry
   `current_value`, `ratio`, `failed_fans`, etc. No schema change needed.

3. **`MinerHistory` is not in `interfaces.md`** by name, but the protocol says
   `miner_history: "MinerHistory"`. Our implementation matches the implied
   shape.

---

## Open questions / items for Maestro

1. **`kpi_update` timing**: ingest emits one `kpi_update` per `telemetry_tick`.
   The detector's KPI background thread will lag by at most one poll cycle
   (100 ms). The XGBoost features use `last_kpi()` which may be one tick stale.
   This is acceptable for a 30-min prediction target.

2. **Model retraining**: once trained, models are static. A production system
   would retrain periodically on confirmed-clean data. Out of scope for this
   prototype; noted for the report.

3. **Fault types not mapped to `flag_type`**: `hashboard_failure` in the
   simulator causes hashrate degradation. The rule engine correctly catches this
   via `hashrate_degradation` flag. The event schema does not have a
   `hashboard_failure` flag_type — this is correct per the closed enum in
   `shared/schemas/events.py`.
