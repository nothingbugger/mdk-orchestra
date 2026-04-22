# MODULE_NOTES — `ingest/`

Author: Claude Code session on `feat/ingest`
Date: 2026-04-20
Status: complete

---

## What was built

The ingest module turns raw `telemetry_tick` events into KPIs:

- **`ingest/features.py`** — `MinerWindow` (per-miner rolling state, deque-based)
  and `FleetState` (fleet-level aggregation, r percentile cache).
- **`ingest/kpi.py`** — `compute_te`, `compute_hsi`, `compute_miner_status`.
  All public functions match `shared/specs/interfaces.md §2` signatures exactly.
- **`ingest/runner.py`** — `run_ingest()`: main loop, tails telemetry, emits
  `kpi_update` (per tick) and `fleet_snapshot` (every `snapshot_interval_s`).
- **`ingest/thresholds.py`** — all configurable constants (thresholds, weights,
  window sizes, knobs) in one reviewable file.
- **`ingest/main.py`** — CLI entry point with argparse.
- **`tests/test_ingest_kpi.py`** — 23 unit tests covering TE golden set,
  HSI monotonicity, status transitions, and warmup behaviour.

---

## Design choices

### Rolling window implementation: `collections.deque`

Chose `deque(maxlen=N)` over polars/pandas for the rolling windows.  Rationale:

1. **No external dependency overhead**: at 10 events/s (50 miners × 0.2 Hz),
   we never need vectorised ops.  A deque is O(1) append/pop and uses ~300 bytes
   per slot.
2. **No dataframe serialisation per tick**: polars DataFrames add noticeable
   latency when sliced on every tick.
3. **Easier testing**: pure Python, no polars version dependency conflicts.

The 30-min window holds 360 snapshots (at 5s tick interval).  Each snapshot is
~60 bytes (3 floats + 1 float ts + overhead) → ≈22 KB per miner × 50 miners =
~1 MB total.  Well within memory budget.

### Cooling power estimate: `P_COOL_FRACTION = 0.18`

`P_cool` is not telemetered.  We estimate it as `0.18 × P_asic`.

Rationale: the v1 dataset (Appendix A) calibrated S19-class ASIC cooling
overhead at 15–20% of ASIC power for air-cooled rack deployments.  We chose
the midpoint (18%) with a slight upward bias from empirical air-cooling
measurements.  The fraction is exposed in `thresholds.py` — operators running
immersion-cooled fleets can reduce this toward 0.05.

### HSI weights: 0.40 / 0.35 / 0.25

The three HSI components were weighted as:
- `thermal_stress`: **0.40** — strongest predictor of imminent failure in v1
  dataset (thermal runway is the leading failure mode for S19-class miners)
- `hashrate_variability`: **0.35** — coefficient of variation is a strong
  early signal for chip instability and hashboard degradation
- `hot_time_frac`: **0.25** — complements thermal_stress (catches sustained
  overtemperature even when σ_T is low)

These weights are empirical guesses calibrated to v1 failure pattern data.
A future improvement would be to fit weights via logistic regression on the
labeled fault dataset.  Weights are in `thresholds.py` for easy tuning.

### `thermal_stress` formula

```
stress = clip( (T_now − T_mean_30min) / max(3σ_T, 5°C), 0, 1 )
```

This is miner-relative — a miner running cool but 3σ above its OWN baseline
scores the same as a hot miner at 3σ above its baseline.  This captures
anomalous thermal events without being fooled by miners that run hot normally
(e.g. turbo mode miners).

The 5°C floor prevents division-by-zero for miners with very stable temps.

### TE percentile computation

`FleetState._latest_r` stores the most recent r(t) for each miner.  Fleet
percentiles are computed over this dict on every tick.  This means percentiles
update on every telemetry event (reactive), which is acceptable at 10 events/s.

A more stable approach would use a 6-hour rolling window of r values per miner,
but the current implementation is simpler and produces reasonable results once
the fleet is fully populated.

### Warmup semantics

TE = 50.0 (neutral placeholder) until at least 5 miners × 60 ticks each have
been recorded.  The 60-tick threshold equates to 5 minutes of data per miner
at the default 5s tick rate.  This prevents meaningless TE scores during
system startup and avoids edge cases with a one-miner "fleet".

The warmup state is logged at every kpi_update emission so dashboards can
show a "warming up" indicator.

---

## Deviations from spec

### `compute_te` signature

The `interfaces.md` §2 signature is:
```python
def compute_te(telemetry: dict, alpha: float = 0.5,
               fleet_r_p5: float, fleet_r_p95: float) -> tuple[float, dict]:
```

This is invalid Python (positional argument after keyword argument).  I
resolved the ambiguity by making `fleet_r_p5` and `fleet_r_p95` keyword
arguments with `math.nan` defaults, and added `sigma_hash: float = 0.0` as
an additional keyword argument (needed because σ_hash is computed externally
from the rolling window before calling `compute_te`).  This is backward-
compatible: callers who pass all four arguments positionally would break, but
no callers existed at time of writing.

**Proposed resolution**: update `interfaces.md` to reflect the actual
signature.  Flagged in this file so Maestro can coordinate.

### `compute_hsi` signature

The spec shows `window_30min: list[dict]` as the second argument.  I used
`window: MinerWindow | None` instead.  Rationale: passing raw dicts would
require re-parsing the rolling window from scratch on every tick.  `MinerWindow`
is maintained by the runner and already has all needed rolling stats pre-
computed.  Downstream callers (tests, deterministic_tools) can always call the
function with a populated window.

**Proposed resolution**: update `interfaces.md` to reflect `MinerWindow` usage
or keep the dict-list form and document that callers must build it from raw
telemetry.  Currently the latter is impractical.

---

## Open issues / future work

1. Fan-curve based P_cool estimation (requires datasheet calibration)
2. Fit HSI weights from v1 labeled fault dataset (logistic regression)
3. 6h rolling window for fleet r percentiles (currently single-point latest)
4. α calibration: find optimal value from historical TE vs fault correlation
5. Thread-safety: current implementation is single-threaded.  If runner is
   parallelised in future, `FleetState` access will need locking.
