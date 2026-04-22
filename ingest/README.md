# MDK Fleet — `ingest/` module

Consumes the real-time `telemetry_tick` stream from the simulator, computes
**True Efficiency (TE)** and **Health State Index (HSI)** per miner on every
tick, and emits `kpi_update` and `fleet_snapshot` events for downstream
consumers (dashboard, deterministic tools).

---

## Running

```bash
# Default paths (auto-detected, see shared/paths.py):
python -m ingest.main

# Explicit paths:
python -m ingest.main \
    --input-stream /run/mdk_fleet/stream/telemetry.jsonl \
    --kpi-output   /run/mdk_fleet/stream/kpis.jsonl \
    --snap-output  /run/mdk_fleet/stream/snapshots.jsonl \
    --snap-interval 1.0

# Debug logging:
python -m ingest.main --log-level DEBUG
```

Stream paths default to the canonical layout from `shared/paths.py`, with
automatic fallback to the local `events/` directory if `/run/mdk_fleet/` is
not writable (useful on dev machines without root access).  Override via:

```bash
export MDK_STREAM_DIR=/path/to/your/events
python -m ingest.main
```

---

## True Efficiency (TE) formula

### Overview

TE measures how efficiently a miner converts electricity into economic value,
relative to the current fleet distribution.  Scale: **0 = worst decile, 100 = top decile**.

### Full formula (from BRIEFING Appendix B)

```
TE(t) = 100 · clip( (log r(t) − log r_p5) / (log r_p95 − log r_p5), 0, 1 )

r(t)  = V(t) / C(t)
V(t)  = H_eff(t) · P_hashprice(t)
C(t)  = [P_asic(t) + P_cool(t)] · ρ_elec(t) + α · σ_hash(t)
```

| Symbol | Description | Source |
|--------|-------------|--------|
| `H_eff` | Effective hashrate (TH/s) | `hashrate_th` in telemetry |
| `P_hashprice` | Network hashprice (USD/TH/day) | `env.hashprice_usd_per_th_day` |
| `P_asic` | ASIC power draw (W) | `power_w` in telemetry |
| `P_cool` | Cooling power (W) | Estimated as `P_COOL_FRACTION × P_asic` |
| `ρ_elec` | Electricity tariff (USD/kWh) | `env.elec_price_usd_kwh` |
| `σ_hash` | Rolling std of hashrate (30-min window, TH/s) | Computed in `features.py` |
| `α` | Stability-penalty weight | `ALPHA = 0.5` (default) |
| `r_p5`, `r_p95` | Fleet 5th/95th percentile of `r(t)` | Computed over live fleet |

`V(t)` and `C(t)` are both expressed in **USD/day**:
- `V(t) = H_eff × P_hashprice`
- `C(t) = (P_asic + P_cool) / 1000 × ρ_elec × 24 + α × σ_hash`

### Calibration knobs

All knobs live in `ingest/thresholds.py` for operator review:

| Knob | Default | Effect |
|------|---------|--------|
| `ALPHA` | `0.5` | Higher → penalises hashrate-unstable miners more |
| `P_COOL_FRACTION` | `0.18` | Cooling overhead as fraction of ASIC power |
| `WARMUP_MIN_MINERS` | `5` | Miners needed before TE computation activates |
| `WARMUP_MIN_TICKS_PER_MINER` | `60` | Ticks per miner needed for warmup (~5 min) |
| `TE_WARMUP_DEFAULT` | `50.0` | TE placeholder during warmup |

### Warmup period

During warmup (`fleet.is_warmed_up() == False`), TE defaults to 50.0 for all
miners — the fleet-average placeholder.  Warmup ends when at least
`WARMUP_MIN_MINERS` miners each have `WARMUP_MIN_TICKS_PER_MINER` ticks
recorded.  With default settings (50 miners, 5s tick) this is ≈5 minutes after
startup.

### Cooling power estimation

`P_cool` is not directly telemetered.  We estimate it as a fixed fraction of
`P_asic` (default `P_COOL_FRACTION = 0.18`).  This approximates the
air-cooling overhead measured in the v1 dataset (S19-class miners in a typical
rack).  See `MODULE_NOTES.md` for the rationale.

---

## Health State Index (HSI) formula

HSI measures a miner's operational health on a scale of **0 (critical) to 100 (healthy)**.

```
HSI = 100 × (1 − weighted_mean(thermal_stress, hashrate_variability, hot_time_frac))
```

### Components (each in [0, 1], higher = worse)

| Component | Weight | Formula |
|-----------|--------|---------|
| `thermal_stress` | 0.40 | `clip( (T_now − T_baseline) / (3σ, floor 5°C), 0, 1 )` |
| `hashrate_variability` | 0.35 | Coefficient of variation (σ/μ) of hashrate, 30-min window, clipped |
| `hot_time_frac` | 0.25 | Fraction of 30-min ticks above `HOT_TEMP_THRESHOLD_C` (80°C) |

### Miner status classification

| Status | Condition | TTF proxy |
|--------|-----------|-----------|
| `shut` | No tick for >`STALE_TIMEOUT_S` (120s) | Offline |
| `imm`  | HSI < 40 OR TE < 20 | < 6h |
| `warn` | HSI < 65 OR TE < 35 | 6–24h |
| `ok`   | All thresholds clear | > 24h |

All thresholds are in `ingest/thresholds.py`.

---

## Open issues

1. **`P_cool` is a fixed fraction** — a future improvement would be to
   infer cooling power from fan RPM × airflow models.  This requires fan-curve
   calibration data not currently in telemetry.

2. **TE percentiles are single-point (latest r per miner)** — using a rolling
   window of r values per miner (e.g. last 6h) would give more stable
   percentiles.  Current implementation stores the most recent r for each
   miner; with 50 miners this is already reasonable.

3. **Fleet snapshot uses latest KPI per miner** — there is no time-windowing
   of snapshot entries, so a miner with a very old last-tick but not yet stale
   will show potentially stale KPIs.

4. **`σ_hash` units** — σ_hash is TH/s while cost terms are USD/day.
   The α weight bridges this dimensional mismatch empirically.  The
   `ALPHA = 0.5` default produces reasonable results based on v1 calibration.

---

## Module layout

```
ingest/
├── __init__.py       # module marker
├── main.py           # CLI entry point
├── runner.py         # run_ingest() — the main loop
├── kpi.py            # compute_te(), compute_hsi(), compute_miner_status()
├── features.py       # MinerWindow, FleetState — rolling state
├── thresholds.py     # configurable constants
└── README.md         # this file
```
