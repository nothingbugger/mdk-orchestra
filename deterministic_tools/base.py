"""Base types for the deterministic_tools module.

Defines the Flagger Protocol, FlagResult dataclass, and MinerHistory rolling-state
container that all flaggers and the main detector loop depend on.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable

from shared.schemas.events import KpiUpdate, TelemetryTick


# ---------------------------------------------------------------------------
# FlagResult — return type of every flagger
# ---------------------------------------------------------------------------


@dataclass
class FlagResult:
    """A pre-failure pattern detected by a flagger.

    Matches the ``flag_raised`` event schema (minus envelope fields that are
    added by the detector loop: flag_id, miner_id, ts).
    """

    flag_type: str
    """Closed set — see FlagType in shared/schemas/events.py."""

    severity: str
    """One of 'info', 'warn', 'crit'."""

    confidence: float
    """[0.0, 1.0] — estimated probability this is a real signal."""

    source_tool: str
    """One of 'xgboost_predictor', 'isolation_forest_v2', 'rule_engine'."""

    evidence: dict
    """Free-form evidence block; must contain 'metric' and 'window_min' keys."""

    raw_score: float
    """Raw numeric score from the detector (e.g. anomaly score, probability)."""


# ---------------------------------------------------------------------------
# MinerHistory — bounded rolling per-miner state
# ---------------------------------------------------------------------------

# Number of ticks to keep in memory per miner.
# At 5 s per tick: 720 ticks ≈ 1 hour, 360 ticks ≈ 30 min.
_HISTORY_MAXLEN: int = 720


@dataclass
class MinerHistory:
    """Rolling state for a single miner, bounded by ``maxlen`` ticks.

    The detector loop maintains one ``MinerHistory`` per miner_id and passes it
    into each flagger on every tick. Flaggers read but do not mutate history.
    """

    miner_id: str
    maxlen: int = _HISTORY_MAXLEN

    telemetry: deque[TelemetryTick] = field(default_factory=deque)
    kpis: deque[KpiUpdate] = field(default_factory=deque)
    timestamps: deque[datetime] = field(default_factory=deque)

    # Cooldown tracking: (flag_type) → last-emitted datetime
    _cooldowns: dict[str, datetime] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.telemetry = deque(maxlen=self.maxlen)
        self.kpis = deque(maxlen=self.maxlen)
        self.timestamps = deque(maxlen=self.maxlen)

    def push_telemetry(self, tick: TelemetryTick, ts: datetime) -> None:
        """Append a new telemetry tick (oldest is auto-evicted at maxlen)."""
        self.telemetry.append(tick)
        self.timestamps.append(ts)

    def push_kpi(self, kpi: KpiUpdate) -> None:
        """Append a new KPI update."""
        self.kpis.append(kpi)

    def last_telemetry(self) -> TelemetryTick | None:
        """Most recent telemetry tick, or None if history is empty."""
        return self.telemetry[-1] if self.telemetry else None

    def last_kpi(self) -> KpiUpdate | None:
        """Most recent KPI update, or None if no KPIs have been pushed."""
        return self.kpis[-1] if self.kpis else None

    def recent_telemetry(self, n: int) -> list[TelemetryTick]:
        """Return the last ``n`` telemetry ticks (most-recent last)."""
        items = list(self.telemetry)
        return items[-n:] if n < len(items) else items

    def recent_ticks_count(self) -> int:
        return len(self.telemetry)

    # -- cooldown helpers -------------------------------------------------

    def is_on_cooldown(self, flag_type: str, cooldown_s: float, now: datetime) -> bool:
        """Return True if this (miner, flag_type) pair is in cooldown.

        Args:
            flag_type: the flag type string.
            cooldown_s: minimum seconds between repeated emissions.
            now: current timestamp.

        Returns:
            True if a flag of this type was recently emitted and the cooldown
            has not yet elapsed.
        """
        last = self._cooldowns.get(flag_type)
        if last is None:
            return False
        elapsed = (now - last).total_seconds()
        return elapsed < cooldown_s

    def record_emission(self, flag_type: str, ts: datetime) -> None:
        """Register that a flag was emitted now (starts / resets cooldown)."""
        self._cooldowns[flag_type] = ts


# ---------------------------------------------------------------------------
# Flagger Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Flagger(Protocol):
    """Duck-typed protocol every flagger must satisfy.

    Matches the interface definition in ``shared/specs/interfaces.md`` §3.
    """

    name: str

    def evaluate(
        self,
        miner_telemetry: TelemetryTick,
        miner_history: MinerHistory,
    ) -> FlagResult | None:
        """Evaluate one telemetry tick and return a FlagResult or None.

        Args:
            miner_telemetry: the latest (just-received) telemetry tick for this
                miner. Implementations must NOT read ``fault_injected`` for
                detection logic — that field is ground-truth only.
            miner_history: rolling history for this miner including past
                telemetry and KPI updates. Read-only; do not mutate.

        Returns:
            A ``FlagResult`` if a pre-failure pattern was detected, else ``None``.
        """
        ...
