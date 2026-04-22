# simulator — MDK Fleet telemetry generator

Real-time streaming telemetry for a 50-miner Bitcoin ASIC fleet.
Emits `telemetry_tick` events to a JSONL stream; everything downstream (ingest,
deterministic tools, agents, dashboard) reads from that stream.

---

## How to run

```bash
# Default: 50 miners, 5-second ticks, run forever (Ctrl-C to stop)
python -m simulator.main

# With seed for reproducibility, run 1 simulated hour at 10× real speed
python -m simulator.main --seed 42 --duration 3600 --speed 10

# 50 miners, no faults, custom output path
python -m simulator.main --n-miners 50 --no-faults --output /tmp/telemetry.jsonl

# Full option list
python -m simulator.main --help
```

Output path defaults to `events/telemetry.jsonl` (repo-local fallback when
`/run/mdk_fleet/stream/` is not writable). Set `MDK_STREAM_DIR` env var to
override. The live dashboard copy is also written to `events/live.jsonl`.

---

## Physics calibration

Target hardware: **Antminer S19j Pro** (Bitmain, 2021 production run).

| Mode | Hashrate | Power | Eff (J/TH) |
|------|----------|-------|------------|
| turbo | 115 TH/s | 3600 W | 31.3 |
| balanced | 104 TH/s | 3250 W | 31.3 |
| eco | 88 TH/s | 2700 W | 30.7 |

### Per-miner variance
Each miner draws a small random bias at creation time (seed-reproducible):
- Hashrate: ±1.8 TH/s (1σ) — unit-to-unit manufacturing spread
- Voltage: ±40 mV (1σ) — PSU trim variation
- Chip temp: ±1.5 °C (1σ) — heatsink contact variation

### Thermal model
```
target_chip_temp = ambient_temp + (power / nominal_power) × 52°C × mode_factor + miner_bias
actual_chip_temp = 0.88 × target + 0.12 × previous_chip_temp   # inertia blend
```
The 52 °C ambient rise comes from Bitmain datasheet (chip junction temp at
rated power in 25 °C ambient). Thermal inertia constant (0.12) models the
aluminum heatsink + airflow response delay.

### Fan model
4 fans per miner (inlet + exhaust pair). RPM scales linearly with chip temp
above 70 °C at ~35 RPM/°C, clamped to [4500, 7200] RPM.

### Environmental feeds
| Feed | Model |
|------|-------|
| Site temp | Diurnal sine: mean 22 °C, ±4 °C amplitude, peak at 14:00 UTC |
| Site humidity | Mean-reverting random walk, 20–75 % |
| Electricity price | Base 6.5 ¢/kWh + 1.8 ¢ peak surcharge 15:00–19:00 UTC |
| Hashprice | Mean-reverting walk around $0.055/TH/day |

---

## Fault signatures

Faults are pre-scheduled at fleet startup. Each fault has a **pre-onset window**
during which telemetry shifts gradually — this is what the deterministic tools
are designed to detect.

| Fault | Pre-onset window | Key signature |
|-------|-----------------|---------------|
| `chip_instability` | 20–60 min | Hashrate variance increases (3→15 TH/s noise), chip temp spikes |
| `cooling_degradation` | 15–40 min | One fan RPM drops toward 0, chip temp rises +12 °C |
| `power_sag` | 30–75 min | Voltage drifts −350 mV, hashrate becomes unstable |
| `hashboard_failure` | 5–20 min | Hashrate drops ~33% in steps (hashboard count: 3→2) |

The `fault_injected` field in the event is:
- `null` during normal operation and before pre-onset begins
- The fault tag string (e.g., `"power_sag"`) from pre-onset through active phase

Only ~25% of miners get a fault scheduled (configurable via
`_FAULT_INJECTION_PROBABILITY` in `fleet_sim.py`). Fault onset ranges from
tick 60 to tick 720 (5–60 simulated minutes).

---

## Operating mode distribution

| Mode | Fleet fraction |
|------|---------------|
| turbo | 20% |
| balanced | 65% |
| eco | 15% |

---

## Running tests

```bash
# Once deps are installed:
pytest tests/test_simulator_smoke.py -v
```

Tests cover: valid envelope per tick, miner_id format, fault_injected semantics
(null before onset, tag after), seed reproducibility, environmental bounds,
physics bounds.

---

## Open issues / deviations

See `simulator/MODULE_NOTES.md` for full list.

1. **Python 3.14 on dev machine** — no Python 3.12 available at time of writing.
   Syntax checked with 3.14 via `py_compile`; tests must be run after installing
   Python 3.12 + deps.

2. **Aging factor** — `tick_miner` accepts an `aging_factor` parameter but
   `fleet_sim.py` does not yet wire in a long-term hashrate degradation model.
   Currently defaults to 1.0. Future: compute from uptime.

3. **Fault clearance** — once a fault activates it never clears in the current
   model. Future: add a recovery tick threshold to simulate restart / repair.

4. **Mode changes** — miners keep their initial operating mode for the whole run.
   Future: the orchestrator could trigger mode changes via an action feed.
