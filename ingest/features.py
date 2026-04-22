"""Per-miner rolling feature state for the ingest module.

Uses `collections.deque` for rolling windows — pure stdlib, low overhead,
O(1) append/pop, no dataframe allocation per tick.

Each `MinerWindow` instance stores the last N telemetry ticks for one miner
and exposes rolling statistics used by `kpi.py`.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from ingest.thresholds import (
    HOT_TEMP_THRESHOLD_C,
    ROLLING_WINDOW_30MIN_TICKS,
    ROLLING_WINDOW_6H_TICKS,
    STALE_TIMEOUT_S,
)


@dataclass
class TickSnapshot:
    """Minimal extract from a TelemetryTick stored in the rolling window.

    We keep only what rolling computations need — avoids holding large raw
    telemetry objects in memory for all miners × all window depths.
    """

    wall_ts: float  # time.monotonic() when tick was received
    hashrate_th: float
    temp_chip_c: float
    power_w: float  # P_asic proxy


@dataclass
class MinerWindow:
    """Rolling state for a single miner.

    Two separate deques:
      - `short` — last ROLLING_WINDOW_30MIN_TICKS ticks (30 min at 5s cadence)
      - `long`  — last ROLLING_WINDOW_6H_TICKS ticks (6 h at 5s cadence)

    The `long` window feeds fleet-level r(t) percentile computation inside
    `FleetState`; the `short` window feeds per-miner HSI components.

    Note: all rolling statistics are computed lazily on each tick — simple
    enough at 10 events/s total fleet rate.
    """

    miner_id: str
    short: deque[TickSnapshot] = field(default_factory=lambda: deque(maxlen=ROLLING_WINDOW_30MIN_TICKS))
    long: deque[TickSnapshot] = field(default_factory=lambda: deque(maxlen=ROLLING_WINDOW_6H_TICKS))
    last_tick_wall_ts: Optional[float] = field(default=None)
    tick_count: int = field(default=0)

    def record(self, snap: TickSnapshot) -> None:
        """Append a new snapshot to both windows."""
        self.short.append(snap)
        self.long.append(snap)
        self.last_tick_wall_ts = snap.wall_ts
        self.tick_count += 1

    # ------------------------------------------------------------------
    # Rolling statistics
    # ------------------------------------------------------------------

    def rolling_hashrate_mean(self) -> float:
        """Mean hashrate over the 30-min window. Returns NaN if empty."""
        if not self.short:
            return math.nan
        vals = [s.hashrate_th for s in self.short]
        return sum(vals) / len(vals)

    def rolling_hashrate_std(self) -> float:
        """Population std-dev of hashrate over the 30-min window.

        Returns 0.0 when there is only one sample (std is undefined).
        σ_hash is used in the TE cost term: C(t) = … + α · σ_hash(t).
        """
        n = len(self.short)
        if n < 2:
            return 0.0
        vals = [s.hashrate_th for s in self.short]
        mean = sum(vals) / n
        variance = sum((v - mean) ** 2 for v in vals) / n
        return math.sqrt(variance)

    def rolling_hashrate_cv(self) -> float:
        """Coefficient of variation of hashrate (σ / μ), clipped to [0, 1].

        Used as `hashrate_variability` component in HSI.
        Returns 0 when mean is 0 (avoid division by zero).
        """
        mean = self.rolling_hashrate_mean()
        if math.isnan(mean) or mean <= 0:
            return 0.0
        cv = self.rolling_hashrate_std() / mean
        return min(1.0, max(0.0, cv))

    def rolling_temp_mean(self) -> float:
        """Mean chip temperature over the 30-min window. Returns NaN if empty."""
        if not self.short:
            return math.nan
        vals = [s.temp_chip_c for s in self.short]
        return sum(vals) / len(vals)

    def rolling_temp_std(self) -> float:
        """Std-dev of chip temperature over the 30-min window."""
        n = len(self.short)
        if n < 2:
            return 0.0
        vals = [s.temp_chip_c for s in self.short]
        mean = sum(vals) / n
        variance = sum((v - mean) ** 2 for v in vals) / n
        return math.sqrt(variance)

    def hot_time_fraction(self, hot_threshold_c: float = HOT_TEMP_THRESHOLD_C) -> float:
        """Fraction of 30-min ticks with chip temp above `hot_threshold_c`.

        Returns 0.0 if no ticks recorded yet.
        """
        if not self.short:
            return 0.0
        hot = sum(1 for s in self.short if s.temp_chip_c > hot_threshold_c)
        return hot / len(self.short)

    def thermal_stress(self) -> float:
        """Normalised thermal stress relative to miner's own rolling baseline.

        Formula:
            stress = clip( (T_now - T_baseline) / T_scale, 0, 1 )

        Where:
            T_baseline = rolling mean of chip temp (30-min window)
            T_scale    = max(rolling std * 3, 5°C) — 3-sigma range,
                         floor at 5°C to avoid near-zero divisor
            T_now      = most recent chip temp

        A miner running exactly at its own baseline scores 0.
        A miner 3σ above its own baseline scores 1.
        """
        if not self.short:
            return 0.0
        t_now = self.short[-1].temp_chip_c
        t_mean = self.rolling_temp_mean()
        t_std = self.rolling_temp_std()
        t_scale = max(t_std * 3.0, 5.0)
        stress = (t_now - t_mean) / t_scale
        return min(1.0, max(0.0, stress))

    def is_stale(self, now_wall: Optional[float] = None) -> bool:
        """True if the miner has not emitted a tick recently enough.

        Args:
            now_wall: current wall clock (time.monotonic()). Defaults to now.
        """
        if self.last_tick_wall_ts is None:
            return True
        if now_wall is None:
            now_wall = time.monotonic()
        return (now_wall - self.last_tick_wall_ts) > STALE_TIMEOUT_S


@dataclass
class FleetState:
    """Aggregate rolling state across all miners.

    Maintains a dict of per-miner windows and computes fleet-level r(t)
    percentiles for the TE formula.

    Also tracks warmup: TE is only meaningful once we have data from at least
    `WARMUP_MIN_MINERS` miners with `WARMUP_MIN_TICKS_PER_MINER` ticks each.
    """

    windows: dict[str, MinerWindow] = field(default_factory=dict)
    # Cache of latest r(t) per miner, used to compute fleet percentiles
    _latest_r: dict[str, float] = field(default_factory=dict)

    def get_or_create(self, miner_id: str) -> MinerWindow:
        """Return existing window or create a fresh one."""
        if miner_id not in self.windows:
            self.windows[miner_id] = MinerWindow(miner_id=miner_id)
        return self.windows[miner_id]

    def update_r(self, miner_id: str, r: float) -> None:
        """Update the cached r(t) for percentile computation."""
        if math.isfinite(r) and r > 0:
            self._latest_r[miner_id] = r

    def fleet_r_percentiles(self) -> tuple[float, float]:
        """Return (r_p5, r_p95) from the latest r(t) values across the fleet.

        Returns (NaN, NaN) when fewer miners than WARMUP_MIN_MINERS have data.
        Falls back to the single-miner range if only a few miners have valid r.
        """
        from ingest.thresholds import WARMUP_MIN_MINERS

        values = sorted(v for v in self._latest_r.values() if math.isfinite(v) and v > 0)
        if len(values) < WARMUP_MIN_MINERS:
            return math.nan, math.nan

        n = len(values)
        p5_idx = max(0, int(math.floor(0.05 * (n - 1))))
        p95_idx = min(n - 1, int(math.ceil(0.95 * (n - 1))))
        return values[p5_idx], values[p95_idx]

    def is_warmed_up(self) -> bool:
        """True when the fleet has enough data for stable TE computation."""
        from ingest.thresholds import WARMUP_MIN_MINERS, WARMUP_MIN_TICKS_PER_MINER

        eligible = sum(
            1
            for w in self.windows.values()
            if w.tick_count >= WARMUP_MIN_TICKS_PER_MINER
        )
        return eligible >= WARMUP_MIN_MINERS

    def latest_kpis(self) -> dict[str, dict]:
        """Return most recently computed KPI dict per miner_id.

        Populated by the runner after each kpi_update event is emitted.
        """
        return getattr(self, "_latest_kpis", {})

    def set_latest_kpi(self, miner_id: str, kpi: dict) -> None:
        """Store the latest KPI dict for a miner (called by runner)."""
        if not hasattr(self, "_latest_kpis"):
            self._latest_kpis: dict[str, dict] = {}
        self._latest_kpis[miner_id] = kpi
