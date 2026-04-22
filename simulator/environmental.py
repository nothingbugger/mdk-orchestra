"""Environmental feed simulator.

Models ambient and grid conditions that affect all miners:
- Site temperature: day/night sinusoidal cycle
- Site humidity: slow random walk within realistic bounds
- Electricity price: flat baseline with peak-window surcharges
- Hashprice: slow random walk (reflects network difficulty + BTC price drift)

All state is deterministic given the starting timestamp and seed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

# ---------------------------------------------------------------------------
# Constants — calibrated from real-world ranges
# ---------------------------------------------------------------------------

# Temperature
_SITE_TEMP_BASE_C: float = 22.0  # annual mean indoor temp
_SITE_TEMP_DIURNAL_AMP_C: float = 4.0  # ±4 °C day/night swing
_SITE_TEMP_NOISE_STD_C: float = 0.15  # sensor jitter

# Humidity
_HUMIDITY_BASE_PCT: float = 45.0
_HUMIDITY_DRIFT_SIGMA: float = 0.05  # per-tick drift
_HUMIDITY_MEAN_REVERT_STRENGTH: float = 0.02
_HUMIDITY_MIN_PCT: float = 20.0
_HUMIDITY_MAX_PCT: float = 75.0

# Electricity price (USD/kWh)
_ELEC_BASE_USD_KWH: float = 0.065
_ELEC_PEAK_SURCHARGE: float = 0.018  # +1.8¢ during peak hours
_ELEC_PEAK_START_HOUR: int = 15  # 15:00 UTC
_ELEC_PEAK_END_HOUR: int = 19  # 19:00 UTC
_ELEC_NOISE_STD: float = 0.0008

# Hashprice (USD / TH / day)
_HASHPRICE_INIT: float = 0.058
_HASHPRICE_DRIFT_SIGMA: float = 0.0003  # per-tick random walk
_HASHPRICE_MEAN_REVERT_TARGET: float = 0.055
_HASHPRICE_MEAN_REVERT_STRENGTH: float = 0.0005
_HASHPRICE_MIN: float = 0.020
_HASHPRICE_MAX: float = 0.150


@dataclass
class EnvState:
    """Mutable environmental state, advanced one tick at a time."""

    site_temp_c: float = _SITE_TEMP_BASE_C
    site_humidity_pct: float = _HUMIDITY_BASE_PCT
    elec_price_usd_kwh: float = _ELEC_BASE_USD_KWH
    hashprice_usd_per_th_day: float = _HASHPRICE_INIT

    # RNG — seeded externally, never recreated
    _rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())

    def tick(self, tick_time: datetime) -> "EnvState":
        """Advance environmental state by one tick. Mutates in-place, returns self."""
        rng = self._rng

        # ---- site temperature: diurnal sine + noise ----
        hour_frac = tick_time.hour + tick_time.minute / 60.0 + tick_time.second / 3600.0
        # Peak warmth at 14:00, coolest at 02:00 (phase shift -4h from noon)
        phase = (hour_frac - 14.0) * (2 * math.pi / 24.0)
        diurnal = _SITE_TEMP_DIURNAL_AMP_C * math.cos(phase)
        noise = float(rng.normal(0.0, _SITE_TEMP_NOISE_STD_C))
        self.site_temp_c = _SITE_TEMP_BASE_C + diurnal + noise

        # ---- humidity: mean-reverting random walk ----
        drift = float(rng.normal(0.0, _HUMIDITY_DRIFT_SIGMA))
        revert = _HUMIDITY_MEAN_REVERT_STRENGTH * (_HUMIDITY_BASE_PCT - self.site_humidity_pct)
        self.site_humidity_pct = float(
            np.clip(self.site_humidity_pct + drift + revert, _HUMIDITY_MIN_PCT, _HUMIDITY_MAX_PCT)
        )

        # ---- electricity price: base + peak window + noise ----
        in_peak = _ELEC_PEAK_START_HOUR <= tick_time.hour < _ELEC_PEAK_END_HOUR
        peak_component = _ELEC_PEAK_SURCHARGE if in_peak else 0.0
        elec_noise = float(rng.normal(0.0, _ELEC_NOISE_STD))
        self.elec_price_usd_kwh = max(0.01, _ELEC_BASE_USD_KWH + peak_component + elec_noise)

        # ---- hashprice: slow random walk with mean reversion ----
        hp_drift = float(rng.normal(0.0, _HASHPRICE_DRIFT_SIGMA))
        hp_revert = _HASHPRICE_MEAN_REVERT_STRENGTH * (
            _HASHPRICE_MEAN_REVERT_TARGET - self.hashprice_usd_per_th_day
        )
        self.hashprice_usd_per_th_day = float(
            np.clip(
                self.hashprice_usd_per_th_day + hp_drift + hp_revert,
                _HASHPRICE_MIN,
                _HASHPRICE_MAX,
            )
        )

        return self

    def as_dict(self) -> dict:
        """Return env snapshot as a plain dict (matches EnvBlock schema)."""
        return {
            "site_temp_c": round(self.site_temp_c, 2),
            "site_humidity_pct": round(self.site_humidity_pct, 1),
            "elec_price_usd_kwh": round(self.elec_price_usd_kwh, 4),
            "hashprice_usd_per_th_day": round(self.hashprice_usd_per_th_day, 5),
        }


def make_env_state(seed: int | None = None) -> EnvState:
    """Create a freshly seeded EnvState.

    Args:
        seed: RNG seed for reproducibility. None = non-deterministic.

    Returns:
        Initialized EnvState ready to tick.
    """
    rng = np.random.default_rng(seed)
    state = EnvState(_rng=rng)
    return state
