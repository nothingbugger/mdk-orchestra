# deterministic_tools — Pre-Failure Flaggers

Consumes the `telemetry_tick` stream (and optionally `kpi_update`), runs three
flaggers in parallel, and emits `flag_raised` events that the Maestro
orchestrator dispatches to specialist agents.

## Quick start

```bash
# Start the detector (rule engine active immediately):
python -m deterministic_tools.main

# With explicit sensitivity:
python -m deterministic_tools.main --sensitivity high

# Train ML models from a live or recorded stream (runs for 20 min by default):
python -m deterministic_tools.train --duration-min 20

# After training, restart the detector to load the new pkl files.
```

## Three flaggers

### 1. `RuleEngineFlagger` — always active

Hard-threshold rules, no training required. Runs on every `telemetry_tick`.

| Rule | Condition | Duration | Flag type | Severity |
|------|-----------|----------|-----------|----------|
| Thermal runaway | `temp_chip_c > 85 °C` | > 10 min (medium) | `thermal_runaway` | `crit` |
| Voltage drift (warn) | `voltage_v` outside [11.4, 12.6] V | > 5 min (medium) | `voltage_drift` | `warn` |
| Voltage drift (crit) | `voltage_v` outside [11.0, 13.0] V | same | `voltage_drift` | `crit` |
| Hashrate drop | `hashrate_th < 0.80 × expected` | > 5 min (medium) | `hashrate_degradation` | `warn` |
| Fan anomaly | any fan < 3000 RPM | > 2 min (medium) | `fan_anomaly` | `warn` |

Each (miner_id, flag_type) pair has a cooldown (default 10 min at medium
sensitivity) to prevent flag storms.

### 2. `IsolationForestFlagger` — requires training

Fleet-wide Isolation Forest on 5 features: `voltage_v`, `temp_chip_c`,
`hashrate_th`, `power_w`, `fan_rpm_mean`.

- **Startup with `models/if_v2.pkl` present** → loads and activates immediately.
- **No pkl file** → accumulates 1440 clean bootstrap ticks (~2 h at 5 s/tick for
  50 miners), fits the model, saves it, then activates. Clean = `fault_injected
  is None` (checked only in bootstrap; inference is blind to ground truth).

Emits `flag_type=anomaly_composite`.

### 3. `XGBoostFlagger` — requires training

Predicts probability that hashrate will drop > 20% in the next 30 min.
Features: rolling mean/std of hashrate, voltage, temp, power (1-min and 6-min
windows), plus HSI and TE from KPI updates.

- **Startup with `models/xgb_predictor.pkl` present** → loads and activates.
- **No pkl file** → disabled. Run `python -m deterministic_tools.train` first.

Labels (training only): positive if `fault_injected != None` in any of the
next 360 ticks (~30 min). **`fault_injected` is NEVER read at inference time.**

Emits `flag_type=hashrate_degradation` with `source_tool=xgboost_predictor`.

## Sensitivity profiles

Defined in `sensitivity.yaml`. Three named levels:

| Level | Sustained windows | Cooldown | Effect |
|-------|----------|----------|--------|
| `low` | longer (10-15 min) | 20 min | fewer flags, fewer FPs, lower LLM cost |
| `medium` | default (5-10 min) | 10 min | balanced |
| `high` | shorter (1-5 min) | 5 min | more catches, more noise |

Pass `--sensitivity high` to `main.py` or configure via `run_detector()`.

## Training flow

```
Simulator → telemetry_tick events → train.py reads stream for N minutes
→ collects clean samples (IF) + labelled look-ahead windows (XGBoost)
→ fits both models
→ saves models/if_v2.pkl + models/xgb_predictor.pkl
```

```bash
python -m deterministic_tools.train \
    --duration-min 30 \
    --stream /run/mdk_fleet/stream/telemetry.jsonl
```

After training completes, restart the detector. Both ML flaggers will load
their models and activate within the first event.

## Schema compliance

All output is emitted via `shared.event_bus.write_event("flag_raised", "detector", ...)` using
the `FlagRaised` Pydantic model from `shared/schemas/events.py`. No local
schema redefinition.

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `MDK_STREAM_DIR` | `/run/mdk_fleet/stream` | Override stream root |

## File layout

```
deterministic_tools/
├── __init__.py          # public API
├── base.py              # Flagger Protocol, FlagResult, MinerHistory
├── config.py            # sensitivity profile loader
├── sensitivity.yaml     # per-flagger sensitivity profiles
├── rule_engine_flagger.py
├── isolation_forest_flagger.py
├── xgboost_flagger.py
├── runner.py            # run_detector() — main loop
├── main.py              # entry point: python -m deterministic_tools.main
├── train.py             # entry point: python -m deterministic_tools.train
└── README.md
```
