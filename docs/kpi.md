# KPI Design

The ingest layer emits two per-miner KPIs on every telemetry tick:

- **TE (True Efficiency)** — a purely economic fleet-relative ratio
- **HSI (Hashrate Stability Index)** — an operational-stress composite

Both live in `ingest/kpi.py`. Thresholds and weights are centralised in
`ingest/thresholds.py`. The wire schema is in `shared/schemas/events.py`
(`TeComponents`, `HsiComponents`, `KpiUpdate`).

## True Efficiency (TE)

**Formula (v2):**

```
TE(t) = 100 · clip((log r(t) − log r_p5) / (log r_p95 − log r_p5), 0, 1)

  r(t) = V(t) / C(t)
  V(t) = hashrate (TH/s) · hashprice (USD/TH/day)         → USD/day
  C(t) = P_asic (W) / 1000 · elec_price (USD/kWh) · 24 h  → USD/day
```

Scaled by the fleet's 5th and 95th percentile of `log r(t)`. During
warmup TE defaults to **50.0**. Clipped to `[0, 100]`.

**What TE does NOT include** — by design:

- No cooling-fraction multiplier on `P_asic`. Cooling overhead lives in
  HSI as part of the thermal-stress term.
- No `α · σ_hash` in the denominator. Hashrate instability lives in HSI
  as its own stress term.
- No voltage or operating-mode input. Those are operational-safety
  concerns; TE is about revenue-versus-cost, not health.

This separation is deliberate: operators comparing miners on pure
economics (which rigs make the most money right now?) and on safety
(which rigs are about to hurt themselves?) want two different numbers.

## Hashrate Stability Index (HSI)

**Formula (v2):**

```
HSI = 100 · (1 − clip(Σ w_i · s_i, 0, 1))
```

Four stress components, each in `[0, 1]` (higher = worse):

| Component | Formula | Weight |
|---|---|---:|
| thermal_stress     | `clip((site_temp − 20) / 30, 0, 1) + hot_time_fraction` | `0.35` |
| voltage_stress     | `clip(|V − 12.0| / 12.0, 0, 1)` | `0.25` |
| mode_stress        | lookup: `eco → 0.00, balanced → 0.10, turbo → 0.30` | `0.15` |
| instability_stress | `σ_h / μ_h` over the 30-min rolling window, clipped to 1 | `0.25` |

Weights sum to 1.00. Constants live in `ingest/thresholds.py` under
the `HSI_W_*` names.

**Semantics:**

- `HSI = 100` → no stress on any axis.
- `HSI = 50` → warmup default (returned when no rolling window is
  available; distinguishes cold-start from "confirmed healthy").
- `HSI = 0` → every stress axis saturated.

**Severity bands** (from `ingest/thresholds.py`, enforced in
`compute_miner_status`):

| HSI | Status |
|---:|---|
| `≤ 40` | `imm` (imminent failure) |
| `≤ 65` | `warn` |
| `> 65` | `ok` |

## TE ↔ HSI relation

TE and HSI are independent. TE does not consume HSI, HSI does not
consume TE. The only shared signal is the rolling hashrate — `σ_h`
enters HSI as the instability term, but no longer enters TE.

Both metrics are co-emitted in every `kpi_update` event.

## Dashboard rendering

The dashboard (see `dashboard/app.py`) reads both metrics from
`kpis.jsonl` and surfaces them:

- **Fleet page (`/`):** per-miner status chip (derived from HSI + TE
  via `compute_miner_status`)
- **Miner detail (`/miner/<id>`):** TE and HSI trajectories + the
  component breakdowns so an operator can see which stress term is
  dragging HSI down
- **JSON polling (`/api/fleet`):** `te`, `hsi`, and the status token
  for each miner

## Calibration notes

The weights in v2 are the **design defaults** from the pitch report.
They have been verified to produce:

- a `TE` distribution that spans most of `[0, 100]` on a diverse fleet
  (mean ~63, stdev ~30 on a 10-miner balanced-mix simulation)
- an `HSI` distribution clustered in the healthy band (mean ~90, stdev
  ~10 on a clean simulation), with the mean dropping as faults
  accumulate

Re-tuning should be driven by real deployment data, not unit tests.
The existing test suite (`tests/test_ingest_kpi.py`) pins directional
invariants — not specific golden values — so weight adjustments won't
silently break it.

## Replay compatibility

The canonical API replay (`examples/demo_replay/`) was generated
under the v1 formula. Its `kpi_update` records carry the old
components (`p_cool_w`, `sigma_hash`, `alpha`, the old 3-component
HSI). The wire schema (`TeComponents`, `HsiComponents`) uses
`extra="allow"` so the dashboard can ingest both v1 and v2 records
without crashing, though per-miner panel rendering works from the
top-level `te` and `hsi` scalars (which are range-compatible).
