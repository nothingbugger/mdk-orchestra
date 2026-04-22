"""Rule-engine flagger — hard-threshold, no training required.

Rules (from the assignment spec):

  thermal_runaway:
    temp_chip_c > 85 °C sustained for > temp_sustained_s seconds.
    Severity: crit.

  voltage_drift (warn):
    voltage_v outside [11.4, 12.6] V sustained for > voltage_warn_sustained_s.
    Severity: warn.
    Elevated to crit if voltage also outside [voltage_crit_band_lo,
    voltage_crit_band_hi] (default [11.0, 13.0]).

  hashrate_degradation:
    hashrate_th < 0.80 * hashrate_expected_th sustained for >
    hashrate_drop_sustained_s.
    Severity: warn.

  fan_anomaly:
    any of the 4 fan_rpm values < 3000 RPM sustained for > fan_sustained_s.
    Severity: warn.

Per (miner_id, flag_type) cooldown prevents flag storms.  The cooldown window
is read from the sensitivity profile (default 10 min = 600 s).

The rule engine is always active — it does not require a trained model.
"""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime
from typing import Any

import structlog

from deterministic_tools.base import FlagResult, MinerHistory
from deterministic_tools.config import rule_engine_cfg
from shared.schemas.events import TelemetryTick

log = structlog.get_logger(__name__)

# Hard physics thresholds — not tunable by sensitivity profile.
_TEMP_CRIT_C: float = 85.0
_VOLTAGE_WARN_LO: float = 11.4
_VOLTAGE_WARN_HI: float = 12.6
_HASHRATE_DROP_RATIO: float = 0.80
_FAN_WARN_RPM: int = 3000


# ---------------------------------------------------------------------------
# Per-rule timing tracker (internal helper)
# ---------------------------------------------------------------------------


class _SustainedTracker:
    """Track whether a boolean condition has been *continuously* True for at
    least ``required_s`` seconds.

    A single False resets the streak. This implements the "sustained for N
    minutes" semantics from the spec.
    """

    def __init__(self, required_s: float) -> None:
        self._required_s = required_s
        self._streak_start: datetime | None = None

    def update(self, condition: bool, now: datetime) -> bool:
        """Feed the latest condition value and return whether the requirement
        has been met (condition True continuously for >= required_s).

        Args:
            condition: True if the condition is currently breached.
            now: current event timestamp.

        Returns:
            True iff the condition has been sustained long enough.
        """
        if not condition:
            self._streak_start = None
            return False
        if self._streak_start is None:
            self._streak_start = now
        elapsed = (now - self._streak_start).total_seconds()
        return elapsed >= self._required_s

    @property
    def streak_s(self) -> float:
        """Seconds since the current streak started, or 0 if no streak."""
        return 0.0 if self._streak_start is None else 0.0  # placeholder — not exposed externally


# ---------------------------------------------------------------------------
# RuleEngineFlagger
# ---------------------------------------------------------------------------


class RuleEngineFlagger:
    """Deterministic threshold-based flagger.

    Implements the ``Flagger`` Protocol. No training needed; always active.

    Each instance holds per-miner sustained-condition trackers and delegates
    cooldown management to ``MinerHistory``.
    """

    name: str = "rule_engine"

    def __init__(self, sensitivity: str = "medium") -> None:
        """Initialise the rule engine.

        Args:
            sensitivity: one of 'low', 'medium', 'high'. Controls timing
                windows and cooldown duration via ``sensitivity.yaml``.
        """
        self._sensitivity = sensitivity
        self._cfg = rule_engine_cfg(sensitivity)
        self._cooldown_s: float = float(self._cfg["cooldown_s"])

        # Per-miner sustained trackers.
        # Keys: miner_id → _SustainedTracker
        self._thermal_trackers: dict[str, _SustainedTracker] = defaultdict(
            lambda: _SustainedTracker(self._cfg["temp_sustained_s"])
        )
        self._voltage_trackers: dict[str, _SustainedTracker] = defaultdict(
            lambda: _SustainedTracker(self._cfg["voltage_warn_sustained_s"])
        )
        self._hashrate_trackers: dict[str, _SustainedTracker] = defaultdict(
            lambda: _SustainedTracker(self._cfg["hashrate_drop_sustained_s"])
        )
        # Fan: one tracker per fan index, keyed by (miner_id, fan_idx).
        self._fan_trackers: dict[tuple[str, int], _SustainedTracker] = defaultdict(
            lambda: _SustainedTracker(self._cfg["fan_sustained_s"])
        )

        log.info(
            "rule_engine_flagger.init",
            sensitivity=sensitivity,
            temp_sustained_s=self._cfg["temp_sustained_s"],
            voltage_warn_sustained_s=self._cfg["voltage_warn_sustained_s"],
            hashrate_drop_sustained_s=self._cfg["hashrate_drop_sustained_s"],
            fan_sustained_s=self._cfg["fan_sustained_s"],
            cooldown_s=self._cooldown_s,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        miner_telemetry: TelemetryTick,
        miner_history: MinerHistory,
        now: datetime | None = None,
    ) -> FlagResult | None:
        """Evaluate one telemetry tick against all rules.

        Returns the first triggered rule's FlagResult, or None.  Only one
        flag is returned per call; priority order: thermal > voltage > fan >
        hashrate.  Subsequent flags on the same tick will be emitted on the
        next call once cooldown clears.

        NOTE: ``fault_injected`` is NOT read. Detection is based only on
        observable telemetry fields.

        Args:
            miner_telemetry: the latest tick (TelemetryTick model).
            miner_history: rolling per-miner history (read-only).
            now: explicit timestamp override (tests). Defaults to current UTC.

        Returns:
            A FlagResult or None.
        """
        from datetime import timezone

        if now is None:
            now = datetime.now(tz=timezone.utc)

        mid = miner_telemetry.miner_id

        # Run each rule and collect any triggered ones.
        candidates: list[FlagResult] = []

        r = self._check_thermal(miner_telemetry, mid, now)
        if r is not None:
            candidates.append(r)

        r = self._check_voltage(miner_telemetry, mid, now)
        if r is not None:
            candidates.append(r)

        r = self._check_fan(miner_telemetry, mid, now)
        if r is not None:
            candidates.append(r)

        r = self._check_hashrate(miner_telemetry, mid, now)
        if r is not None:
            candidates.append(r)

        if not candidates:
            return None

        # Apply confidence filter.
        min_conf = float(self._cfg["min_confidence"])
        ok = [c for c in candidates if c.confidence >= min_conf]
        if not ok:
            return None

        # Priority: crit > warn > info; within same severity, first triggered.
        priority = {"crit": 0, "warn": 1, "info": 2}
        ok.sort(key=lambda x: priority.get(x.severity, 99))
        chosen = ok[0]

        # Cooldown check.
        if miner_history.is_on_cooldown(chosen.flag_type, self._cooldown_s, now):
            log.debug(
                "rule_engine_flagger.cooldown_suppressed",
                miner_id=mid,
                flag_type=chosen.flag_type,
            )
            return None

        miner_history.record_emission(chosen.flag_type, now)
        log.info(
            "rule_engine_flagger.flag_raised",
            miner_id=mid,
            flag_type=chosen.flag_type,
            severity=chosen.severity,
            confidence=chosen.confidence,
        )
        return chosen

    # ------------------------------------------------------------------
    # Individual rule checks
    # ------------------------------------------------------------------

    def _check_thermal(
        self, tick: TelemetryTick, mid: str, now: datetime
    ) -> FlagResult | None:
        """Rule: chip temp > 85 °C sustained."""
        over_temp = tick.temp_chip_c > _TEMP_CRIT_C
        sustained = self._thermal_trackers[mid].update(over_temp, now)
        if not sustained:
            return None
        return FlagResult(
            flag_type="thermal_runaway",
            severity="crit",
            confidence=0.95,
            source_tool="rule_engine",
            evidence={
                "metric": "temp_chip_c",
                "window_min": self._cfg["temp_sustained_s"] / 60.0,
                "current_value": tick.temp_chip_c,
                "threshold_c": _TEMP_CRIT_C,
            },
            raw_score=float(tick.temp_chip_c - _TEMP_CRIT_C),
        )

    def _check_voltage(
        self, tick: TelemetryTick, mid: str, now: datetime
    ) -> FlagResult | None:
        """Rule: voltage outside warn band, optionally elevated to crit."""
        v = tick.voltage_v
        outside_warn = v < _VOLTAGE_WARN_LO or v > _VOLTAGE_WARN_HI
        sustained = self._voltage_trackers[mid].update(outside_warn, now)
        if not sustained:
            return None

        crit_lo = float(self._cfg["voltage_crit_band_lo"])
        crit_hi = float(self._cfg["voltage_crit_band_hi"])
        outside_crit = v < crit_lo or v > crit_hi

        severity = "crit" if outside_crit else "warn"
        confidence = 0.92 if outside_crit else 0.80

        deviation = min(abs(v - _VOLTAGE_WARN_LO), abs(v - _VOLTAGE_WARN_HI))
        return FlagResult(
            flag_type="voltage_drift",
            severity=severity,
            confidence=confidence,
            source_tool="rule_engine",
            evidence={
                "metric": "voltage_v",
                "window_min": self._cfg["voltage_warn_sustained_s"] / 60.0,
                "current_value": v,
                "warn_band_lo": _VOLTAGE_WARN_LO,
                "warn_band_hi": _VOLTAGE_WARN_HI,
                "crit_band_lo": crit_lo,
                "crit_band_hi": crit_hi,
                "outside_crit": outside_crit,
            },
            raw_score=float(deviation),
        )

    def _check_hashrate(
        self, tick: TelemetryTick, mid: str, now: datetime
    ) -> FlagResult | None:
        """Rule: hashrate < 80% of expected, sustained."""
        expected = tick.hashrate_expected_th
        if expected <= 0:
            return None
        ratio = tick.hashrate_th / expected
        below_threshold = ratio < _HASHRATE_DROP_RATIO
        sustained = self._hashrate_trackers[mid].update(below_threshold, now)
        if not sustained:
            return None
        return FlagResult(
            flag_type="hashrate_degradation",
            severity="warn",
            confidence=0.85,
            source_tool="rule_engine",
            evidence={
                "metric": "hashrate_th",
                "window_min": self._cfg["hashrate_drop_sustained_s"] / 60.0,
                "current_hashrate_th": tick.hashrate_th,
                "expected_hashrate_th": expected,
                "ratio": round(ratio, 4),
                "threshold_ratio": _HASHRATE_DROP_RATIO,
            },
            raw_score=float(_HASHRATE_DROP_RATIO - ratio),
        )

    def _check_fan(
        self, tick: TelemetryTick, mid: str, now: datetime
    ) -> FlagResult | None:
        """Rule: any of the 4 fans < 3000 RPM, sustained."""
        failed_fans: list[dict[str, Any]] = []
        for idx, rpm in enumerate(tick.fan_rpm):
            below_threshold = rpm < _FAN_WARN_RPM
            tracker_key = (mid, idx)
            sustained = self._fan_trackers[tracker_key].update(below_threshold, now)
            if sustained:
                failed_fans.append({"fan_idx": idx, "rpm": rpm})

        if not failed_fans:
            return None

        return FlagResult(
            flag_type="fan_anomaly",
            severity="warn",
            confidence=0.88,
            source_tool="rule_engine",
            evidence={
                "metric": "fan_rpm",
                "window_min": self._cfg["fan_sustained_s"] / 60.0,
                "failed_fans": failed_fans,
                "threshold_rpm": _FAN_WARN_RPM,
            },
            raw_score=float(len(failed_fans)),
        )
