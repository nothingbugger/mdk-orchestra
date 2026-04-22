"""Configurable KPI thresholds for miner status classification.

These thresholds are the single source of truth for `compute_miner_status`.
Reviewers and operators can tune these without touching computation logic.

Status ladder (in order of increasing severity):
    ok   — miner performing normally
    warn — early warning, consider scheduling inspection
    imm  — imminent failure risk, take action within hours
    shut — miner offline / timed out (no recent telemetry tick)

Mapping rule (from BRIEFING §ingest spec):
    shut : no tick received for > STALE_TIMEOUT_S seconds
    imm  : HSI < IMM_HSI or TE < IMM_TE
    warn : HSI < WARN_HSI or TE < WARN_TE
    ok   : all thresholds clear
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Staleness (shut)
# ---------------------------------------------------------------------------

STALE_TIMEOUT_S: float = 120.0
"""Seconds since the last tick before a miner is considered shut/offline."""

# ---------------------------------------------------------------------------
# Imminent failure (imm)
# ---------------------------------------------------------------------------

IMM_HSI: float = 40.0
"""HSI below this → imm status."""

IMM_TE: float = 20.0
"""TE below this → imm status."""

# ---------------------------------------------------------------------------
# Warning (warn)
# ---------------------------------------------------------------------------

WARN_HSI: float = 65.0
"""HSI below this → warn status (unless already imm)."""

WARN_TE: float = 35.0
"""TE below this → warn status (unless already imm)."""

# ---------------------------------------------------------------------------
# TE computation — percentile warmup
# ---------------------------------------------------------------------------

WARMUP_MIN_MINERS: int = 5
"""Minimum distinct miners required before fleet percentiles are computed."""

WARMUP_MIN_TICKS_PER_MINER: int = 60
"""Minimum ticks recorded per miner before warmup is considered complete.
With default 5s tick interval this equals 5 minutes of data per miner."""

TE_WARMUP_DEFAULT: float = 50.0
"""TE value assigned during warmup period (fleet-average placeholder)."""

# ---------------------------------------------------------------------------
# Rolling windows
# ---------------------------------------------------------------------------

ROLLING_WINDOW_30MIN_TICKS: int = 360
"""Number of ticks in a 30-minute window at 5s tick interval.
30 * 60 / 5 = 360.  Resize if simulator tick_interval_s changes."""

ROLLING_WINDOW_6H_TICKS: int = 4320
"""Number of ticks in a 6-hour window.  Used for r(t) fleet percentiles.
6 * 3600 / 5 = 4320."""

# ---------------------------------------------------------------------------
# TE formula knobs (v2 — purely economic, stability moved to HSI)
# ---------------------------------------------------------------------------
#
# Retained as dead code for reference; the current TE formula is
# TE = 100 * percentile_scale(log(V/C)) with
#   V = hashrate * hashprice
#   C = P_asic * elec_price * 24h
# No cooling fraction, no sigma penalty — those live in HSI now.
#
# ALPHA: float = 0.5              # moved to HSI (was sigma_hash penalty weight)
# P_COOL_FRACTION: float = 0.18   # dropped (cooling overhead no longer in TE)


# ---------------------------------------------------------------------------
# HSI formula knobs (v2 — 4 weighted stress components)
# ---------------------------------------------------------------------------

HSI_W_THERMAL: float = 0.35
"""Weight for thermal stress (site_temp + hot_time_fraction)."""

HSI_W_VOLTAGE: float = 0.25
"""Weight for voltage deviation from 12 V nominal."""

HSI_W_MODE: float = 0.15
"""Weight for operating-mode stress (eco / balanced / turbo)."""

HSI_W_INSTABILITY: float = 0.25
"""Weight for hashrate coefficient of variation (sigma_h / mean_h)."""
# Sum of the four = 1.00

MODE_STRESS: dict[str, float] = {
    "eco":      0.00,
    "balanced": 0.10,
    "turbo":    0.30,
}
"""Per-operating-mode stress contribution (dimensionless, 0..1)."""

SITE_TEMP_STRESS_COLD: float = 20.0
"""Ambient temperature (°C) below which site-temp stress is zero."""

SITE_TEMP_STRESS_RANGE_C: float = 30.0
"""Span over which site-temp stress ramps linearly from 0 to 1.
At site_temp = 20 °C stress is 0; at 50 °C it is 1 (clipped)."""

VOLTAGE_NOMINAL_V: float = 12.0
"""Nominal PSU rail voltage. Deviation from this drives the voltage-stress term."""

HOT_TEMP_THRESHOLD_C: float = 80.0
"""Chip temperature above which a tick is counted as 'hot' for hot_time_frac."""
