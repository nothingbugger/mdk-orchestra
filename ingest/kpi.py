"""Live KPI computation: True Efficiency (TE) and Health State Index (HSI).

v2 — KPI alignment with the pitch architecture:
  - TE is a purely economic metric; stability and cooling penalties
    were moved to HSI.
  - HSI now blends four stress components: thermal, voltage, operating
    mode, and hashrate instability.

Public API (matches shared/specs/interfaces.md §2):

    compute_te(telemetry, fleet_r_p5, fleet_r_p95)     → (te, components)
    compute_hsi(telemetry, window_30min)               → (hsi, components)
    compute_miner_status(te, hsi)                      → status_str

TE formula (v2)
---------------

    TE(t) = 100 · clip((log r(t) − log r_p5) / (log r_p95 − log r_p5), 0, 1)

    r(t) = V(t) / C(t)
    V(t) = hashrate (TH/s) · hashprice (USD/TH/day)            → USD/day
    C(t) = P_asic (W) / 1000 · elec_price (USD/kWh) · 24 h     → USD/day

    No cooling-fraction multiplier on P_asic (moved to HSI's thermal term).
    No σ_hash penalty in the denominator (moved to HSI's instability term).

    During warmup TE = 50.0.

HSI formula (v2)
----------------

    HSI = 100 · (1 − clip(Σ w_i · s_i, 0, 1))

    i=thermal     s = clip((site_temp - 20) / 30, 0, 1) + hot_time_fraction   w=0.35
    i=voltage     s = clip(|V - 12| / 12, 0, 1)                                w=0.25
    i=mode        s = MODE_STRESS[eco=0 | balanced=0.1 | turbo=0.3]            w=0.15
    i=instability s = σ_h / μ_h (hashrate coefficient of variation, clipped)   w=0.25

    HSI = 100 → fully healthy;  HSI = 0 → fully stressed.
    During warmup (empty rolling window) HSI = 50.0.
"""

from __future__ import annotations

import math
from typing import Optional

from ingest.features import MinerWindow
from ingest.thresholds import (
    HOT_TEMP_THRESHOLD_C,
    HSI_W_INSTABILITY,
    HSI_W_MODE,
    HSI_W_THERMAL,
    HSI_W_VOLTAGE,
    IMM_HSI,
    IMM_TE,
    MODE_STRESS,
    SITE_TEMP_STRESS_COLD,
    SITE_TEMP_STRESS_RANGE_C,
    TE_WARMUP_DEFAULT,
    VOLTAGE_NOMINAL_V,
    WARN_HSI,
    WARN_TE,
)


_HSI_WARMUP_DEFAULT = 50.0


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# True Efficiency (v2 — pure economic)
# ---------------------------------------------------------------------------


def compute_te(
    telemetry: dict,
    fleet_r_p5: float = math.nan,
    fleet_r_p95: float = math.nan,
    # The following args are retained for backwards-compat with older
    # callers (runner.py passes `alpha` and `sigma_hash`) but are ignored.
    alpha: float = 0.0,
    sigma_hash: float = 0.0,
) -> tuple[float, dict]:
    """Compute True Efficiency (v2) — a purely economic fleet-relative ratio.

    Args:
        telemetry: TelemetryTick.data payload dict.
        fleet_r_p5: 5th percentile of r(t) across the fleet. NaN → warmup.
        fleet_r_p95: 95th percentile of r(t) across the fleet. NaN → warmup.
        alpha, sigma_hash: **ignored** — kept in the signature so existing
            runner plumbing doesn't need a synchronous breaking change.
            Stability penalty lives in HSI now.

    Returns:
        (te_value, components_dict)
    """
    del alpha, sigma_hash  # explicitly discarded — see docstring

    h_eff: float = telemetry["hashrate_th"]
    p_asic_w: float = telemetry["power_w"]
    env: dict = telemetry["env"]
    rho_elec: float = env["elec_price_usd_kwh"]
    p_hashprice: float = env["hashprice_usd_per_th_day"]

    # V(t): revenue in USD/day
    value_usd_day: float = h_eff * p_hashprice

    # C(t): raw ASIC-power cost in USD/day — no cooling multiplier, no sigma
    cost_usd_day: float = (p_asic_w / 1000.0) * rho_elec * 24.0

    # r(t) = V(t) / C(t)
    if cost_usd_day <= 0 or h_eff <= 0 or p_asic_w <= 0 or rho_elec <= 0:
        r_t = math.nan
    else:
        r_t = value_usd_day / cost_usd_day

    # Fleet-relative percentile scaling of log r(t)
    if (
        math.isnan(fleet_r_p5)
        or math.isnan(fleet_r_p95)
        or math.isnan(r_t)
        or fleet_r_p5 <= 0
        or fleet_r_p95 <= fleet_r_p5
    ):
        te = TE_WARMUP_DEFAULT
    else:
        log_r = math.log(r_t)
        log_p5 = math.log(fleet_r_p5)
        log_p95 = math.log(fleet_r_p95)
        denom = log_p95 - log_p5
        if denom < 1e-9:
            te = 50.0
        else:
            score = (log_r - log_p5) / denom
            te = 100.0 * _clip(score, 0.0, 1.0)

    components: dict = {
        "value_usd_day": round(value_usd_day, 4),
        "cost_usd_day": round(cost_usd_day, 4),
        "h_eff_th": h_eff,
        "p_hashprice": p_hashprice,
        "p_asic_w": p_asic_w,
        "rho_elec": rho_elec,
    }
    return round(te, 2), components


# ---------------------------------------------------------------------------
# Health State Index (v2 — 4 stress components)
# ---------------------------------------------------------------------------


def compute_hsi(
    telemetry: dict,
    window: Optional[MinerWindow] = None,
) -> tuple[float, dict]:
    """Compute Health State Index (v2) — operational stress + stability.

    Args:
        telemetry: TelemetryTick.data payload dict. Must carry
            `voltage_v`, `operating_mode`, `env.site_temp_c`, and
            `temp_chip_c`.
        window: rolling state for this miner. None / empty → warmup (50.0).
    """
    env: dict = telemetry.get("env", {}) or {}

    # Warmup: when we don't have enough telemetry history, return neutral 50
    # (instead of a misleading 100) — otherwise a miner with zero ticks would
    # look perfectly healthy.
    if window is None or len(window.short) == 0:
        components: dict = {
            "thermal_stress": 0.0,
            "voltage_stress": 0.0,
            "mode_stress": 0.0,
            "instability_stress": 0.0,
            "hot_time_frac": 0.0,
        }
        return _HSI_WARMUP_DEFAULT, components

    # --- 1. Thermal stress: site-temp + hot-time exposure ---
    site_temp_c: float = float(env.get("site_temp_c", SITE_TEMP_STRESS_COLD))
    site_temp_norm = _clip(
        (site_temp_c - SITE_TEMP_STRESS_COLD) / SITE_TEMP_STRESS_RANGE_C, 0.0, 1.0
    )
    hot_time_frac = window.hot_time_fraction(HOT_TEMP_THRESHOLD_C)
    s_thermal = _clip(site_temp_norm + hot_time_frac, 0.0, 1.0)

    # --- 2. Voltage stress: deviation from 12 V nominal ---
    voltage_v: float = float(telemetry.get("voltage_v", VOLTAGE_NOMINAL_V))
    s_voltage = _clip(abs(voltage_v - VOLTAGE_NOMINAL_V) / VOLTAGE_NOMINAL_V, 0.0, 1.0)

    # --- 3. Mode stress: eco / balanced / turbo lookup ---
    operating_mode: str = str(telemetry.get("operating_mode", "balanced"))
    s_mode = MODE_STRESS.get(operating_mode, MODE_STRESS["balanced"])

    # --- 4. Instability stress: hashrate coefficient of variation ---
    # MinerWindow.rolling_hashrate_cv() already returns σ_h / μ_h clipped to [0,1].
    s_instability = _clip(window.rolling_hashrate_cv(), 0.0, 1.0)

    total_stress = (
        HSI_W_THERMAL * s_thermal
        + HSI_W_VOLTAGE * s_voltage
        + HSI_W_MODE * s_mode
        + HSI_W_INSTABILITY * s_instability
    )
    total_stress = _clip(total_stress, 0.0, 1.0)
    hsi = 100.0 * (1.0 - total_stress)

    components = {
        "thermal_stress": round(s_thermal, 4),
        "voltage_stress": round(s_voltage, 4),
        "mode_stress": round(s_mode, 4),
        "instability_stress": round(s_instability, 4),
        "hot_time_frac": round(hot_time_frac, 4),
    }
    return round(hsi, 2), components


# ---------------------------------------------------------------------------
# Miner status
# ---------------------------------------------------------------------------


def compute_miner_status(te: float, hsi: float) -> str:
    """Classify a miner into one of 'ok' | 'warn' | 'imm' | 'shut'.

    Note: 'shut' (offline/stale) is NOT determined here — it requires
    wall-clock comparison and is handled by the runner before calling this
    function.

    Thresholds are defined in `ingest/thresholds.py` for operator tuning.

    Args:
        te:  True Efficiency value in [0, 100].
        hsi: Health State Index value in [0, 100].

    Returns:
        Status string: 'ok', 'warn', or 'imm'.
    """
    if hsi <= IMM_HSI or te <= IMM_TE:
        return "imm"
    if hsi <= WARN_HSI or te <= WARN_TE:
        return "warn"
    return "ok"
