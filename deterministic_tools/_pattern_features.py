"""Feature extractors for the two XGBoost pattern predictors.

These functions mirror the feature set used to build features.parquet:
  - chip_instability: rolling_variance_spike pattern
  - hashboard_failure: thermal_electrical_decoupling pattern

IMPORTANT: ``fault_injected`` is NEVER read here. All features come from
observable telemetry and rolling history only.
"""

from __future__ import annotations

import math

import numpy as np

from deterministic_tools.base import MinerHistory
from shared.schemas.events import TelemetryTick

# At 5 s/tick, these tick counts match the feature windows:
#   1m  →  12 ticks
#   5m  →  60 ticks
#   30m → 360 ticks

_TICKS_1M = 12
_TICKS_5M = 60
_TICKS_30M = 360

# Minimum ticks required before computing any features.
# Must have at least 30m of history for rolling-variance features.
_MIN_TICKS_CHIP = _TICKS_30M
# Hashboard features need 30m too (power_w_30m_std)
_MIN_TICKS_HB = _TICKS_30M


# ---------------------------------------------------------------------------
# Chip instability features (rolling_variance_spike pattern)
# ---------------------------------------------------------------------------

CHIP_INSTABILITY_FEATURE_NAMES = [
    "hashrate_th_30m_std",
    "power_w_30m_std",
    "hashrate_th_5m_std",
    "temp_chip_c_30m_std",
    "power_w_5m_std",
    "hashrate_th_1m_std",
    "power_w_1m_std",
    "temp_chip_c_5m_std",
    "fan_rpm_mean_30m_std",
    "hashrate_th_1m_mean",
    "hashrate_th_30m_mean",
    "temp_chip_c_1m_mean",
    "power_w_1m_mean",
    "fan_rpm_mean_1m_std",
]


def extract_chip_instability_features(
    miner_telemetry: TelemetryTick,
    miner_history: MinerHistory,
) -> list[float] | None:
    """Extract chip_instability feature vector from rolling history.

    IMPORTANT: Does NOT read ``fault_injected``.

    Args:
        miner_telemetry: the latest tick (not yet pushed to history).
        miner_history: rolling per-miner history (read-only).

    Returns:
        14-element float list (matches CHIP_INSTABILITY_FEATURE_NAMES),
        or None if the window is too short to compute 30m features.
    """
    ticks = miner_history.recent_telemetry(_TICKS_30M)
    if len(ticks) < _MIN_TICKS_CHIP:
        return None

    # Include the latest tick for the current window
    all_ticks = list(ticks) + [miner_telemetry]
    last_30m = all_ticks[-_TICKS_30M:]
    last_5m = all_ticks[-_TICKS_5M:]
    last_1m = all_ticks[-_TICKS_1M:]

    hr_30m = [t.hashrate_th for t in last_30m]
    hr_5m = [t.hashrate_th for t in last_5m]
    hr_1m = [t.hashrate_th for t in last_1m]

    pw_30m = [t.power_w for t in last_30m]
    pw_5m = [t.power_w for t in last_5m]
    pw_1m = [t.power_w for t in last_1m]

    temp_30m = [t.temp_chip_c for t in last_30m]
    temp_5m = [t.temp_chip_c for t in last_5m]
    temp_1m = [t.temp_chip_c for t in last_1m]

    fan_30m = [float(np.mean(t.fan_rpm)) if t.fan_rpm else 5800.0 for t in last_30m]
    fan_1m = [float(np.mean(t.fan_rpm)) if t.fan_rpm else 5800.0 for t in last_1m]

    def _std(xs: list[float]) -> float:
        return float(np.std(xs)) if len(xs) > 1 else 0.0

    def _mean(xs: list[float]) -> float:
        return float(np.mean(xs))

    return [
        _std(hr_30m),       # hashrate_th_30m_std
        _std(pw_30m),       # power_w_30m_std
        _std(hr_5m),        # hashrate_th_5m_std
        _std(temp_30m),     # temp_chip_c_30m_std
        _std(pw_5m),        # power_w_5m_std
        _std(hr_1m),        # hashrate_th_1m_std
        _std(pw_1m),        # power_w_1m_std
        _std(temp_5m),      # temp_chip_c_5m_std
        _std(fan_30m),      # fan_rpm_mean_30m_std
        _mean(hr_1m),       # hashrate_th_1m_mean
        _mean(hr_30m),      # hashrate_th_30m_mean
        _mean(temp_1m),     # temp_chip_c_1m_mean
        _mean(pw_1m),       # power_w_1m_mean
        _std(fan_1m),       # fan_rpm_mean_1m_std
    ]


# ---------------------------------------------------------------------------
# Hashboard failure features (thermal_electrical_decoupling pattern)
# ---------------------------------------------------------------------------

HASHBOARD_FAILURE_FEATURE_NAMES = [
    "temp_per_power",
    "voltage_per_power",
    "temp_amb_c",
    "power_w_1m_mean",
    "power_w_5m_mean",
    "voltage_v_1m_mean",
    "hashrate_th",
    "power_per_hashrate",
    "power_w_30m_std",
    "voltage_v_5m_mean",
    "hashrate_th_1m_mean",
]


def extract_hashboard_failure_features(
    miner_telemetry: TelemetryTick,
    miner_history: MinerHistory,
) -> list[float] | None:
    """Extract hashboard_failure feature vector from rolling history.

    IMPORTANT: Does NOT read ``fault_injected``.

    Args:
        miner_telemetry: the latest tick (not yet pushed to history).
        miner_history: rolling per-miner history (read-only).

    Returns:
        11-element float list (matches HASHBOARD_FAILURE_FEATURE_NAMES),
        or None if the window is too short to compute 30m features.
    """
    ticks = miner_history.recent_telemetry(_TICKS_30M)
    if len(ticks) < _MIN_TICKS_HB:
        return None

    all_ticks = list(ticks) + [miner_telemetry]
    last_30m = all_ticks[-_TICKS_30M:]
    last_5m = all_ticks[-_TICKS_5M:]
    last_1m = all_ticks[-_TICKS_1M:]

    latest = miner_telemetry

    # Derived instantaneous features (matching parquet computation)
    power_safe = max(latest.power_w, 1e-6)
    hr_safe = max(latest.hashrate_th, 1e-6)

    temp_per_power = latest.temp_chip_c / power_safe
    voltage_per_power = latest.voltage_v / power_safe
    power_per_hashrate = latest.power_w / hr_safe

    # Rolling means
    pw_1m_mean = float(np.mean([t.power_w for t in last_1m]))
    pw_5m_mean = float(np.mean([t.power_w for t in last_5m]))
    pw_30m = [t.power_w for t in last_30m]
    pw_30m_std = float(np.std(pw_30m)) if len(pw_30m) > 1 else 0.0

    v_1m_mean = float(np.mean([t.voltage_v for t in last_1m]))
    v_5m_mean = float(np.mean([t.voltage_v for t in last_5m]))

    hr_1m_mean = float(np.mean([t.hashrate_th for t in last_1m]))

    return [
        temp_per_power,             # temp_per_power
        voltage_per_power,          # voltage_per_power
        latest.temp_amb_c,          # temp_amb_c
        pw_1m_mean,                 # power_w_1m_mean
        pw_5m_mean,                 # power_w_5m_mean
        v_1m_mean,                  # voltage_v_1m_mean
        latest.hashrate_th,         # hashrate_th
        power_per_hashrate,         # power_per_hashrate
        pw_30m_std,                 # power_w_30m_std
        v_5m_mean,                  # voltage_v_5m_mean
        hr_1m_mean,                 # hashrate_th_1m_mean
    ]
