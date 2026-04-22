"""Rolling telemetry/KPI buffers used by Maestro to build ReasoningRequest context.

The specialists expect `context.miner_recent_telemetry_30min` and related
slices. Someone has to extract those from the live stream; doing it here
keeps Maestro focused on dispatch/synthesis and lets each specialist ask
for exactly the slice it wants.

The buffer is a background tailer that populates an in-memory dict
`miner_id -> deque[Envelope]`. It is intentionally bounded (the deque
maxlen caps memory) and lossy — the dashboard is the source of truth for
long-horizon history; this is just enough to give the specialists a
recent window.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import structlog

from shared.event_bus import tail_events
from shared.paths import stream_paths

_LOG = structlog.get_logger(__name__)


class FleetHistoryBuffer:
    """In-memory rolling buffer of recent telemetry + KPIs per miner.

    Started once at Maestro boot; runs a daemon thread per channel that
    tails the stream and appends envelopes into per-miner deques. The
    specialists read via `recent_telemetry(miner_id, minutes)` etc.
    """

    def __init__(
        self,
        telemetry_path: Path | None = None,
        kpi_path: Path | None = None,
        telemetry_maxlen: int = 2400,  # ~3.3h at 5s cadence, enough for 30min + buffer
        kpi_maxlen: int = 2400,
    ) -> None:
        paths = stream_paths()
        self._telemetry_path = telemetry_path or paths.telemetry
        self._kpi_path = kpi_path or paths.kpis
        self._telemetry_maxlen = telemetry_maxlen
        self._kpi_maxlen = kpi_maxlen

        self._telemetry: dict[str, deque[dict[str, Any]]] = {}
        self._kpi: dict[str, deque[dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        """Spawn background tailer threads. Idempotent."""
        if self._threads:
            return
        t1 = threading.Thread(target=self._tail_telemetry, name="hist-telemetry", daemon=True)
        t2 = threading.Thread(target=self._tail_kpis, name="hist-kpis", daemon=True)
        t1.start()
        t2.start()
        self._threads = [t1, t2]

    def stop(self) -> None:
        """Signal tailer threads to stop. They're daemons so this is best-effort."""
        self._stopped.set()

    def _tail_telemetry(self) -> None:
        try:
            for env in tail_events(self._telemetry_path, from_start=False, stop_when=self._stopped.is_set):
                if env.event != "telemetry_tick":
                    continue
                miner_id = env.data.get("miner_id")
                if not miner_id:
                    continue
                payload = {"ts": env.ts.isoformat(), **env.data}
                with self._lock:
                    dq = self._telemetry.setdefault(miner_id, deque(maxlen=self._telemetry_maxlen))
                    dq.append(payload)
        except Exception as exc:  # pragma: no cover — daemon fails should not crash main
            _LOG.error("telemetry_tailer_crashed", exc=str(exc))

    def _tail_kpis(self) -> None:
        try:
            for env in tail_events(self._kpi_path, from_start=False, stop_when=self._stopped.is_set):
                if env.event != "kpi_update":
                    continue
                miner_id = env.data.get("miner_id")
                if not miner_id:
                    continue
                payload = {"ts": env.ts.isoformat(), **env.data}
                with self._lock:
                    dq = self._kpi.setdefault(miner_id, deque(maxlen=self._kpi_maxlen))
                    dq.append(payload)
        except Exception as exc:  # pragma: no cover
            _LOG.error("kpi_tailer_crashed", exc=str(exc))

    def recent_telemetry(self, miner_id: str, minutes: float = 30.0) -> list[dict[str, Any]]:
        """Return telemetry samples for `miner_id` within the last `minutes`."""
        cutoff = datetime.now(tz=timezone.utc) - timedelta(minutes=minutes)
        with self._lock:
            dq = self._telemetry.get(miner_id)
            if not dq:
                return []
            snapshot = list(dq)
        return _filter_since(snapshot, cutoff)

    def recent_kpis(self, miner_id: str, minutes: float = 30.0) -> list[dict[str, Any]]:
        cutoff = datetime.now(tz=timezone.utc) - timedelta(minutes=minutes)
        with self._lock:
            dq = self._kpi.get(miner_id)
            if not dq:
                return []
            snapshot = list(dq)
        return _filter_since(snapshot, cutoff)

    def zone_peers(
        self, miner_id: str, minutes: float = 30.0, limit: int = 12
    ) -> dict[str, list[dict[str, Any]]]:
        """Return recent telemetry for other miners, used by env/power agents.

        Prototype: since we don't yet model explicit zone assignment in the
        simulator, returns a best-effort sample of up-to-`limit` other
        miners. Environment/power agents use this to check cross-miner
        correlation. Replace with a proper zone map once the simulator
        exposes rack/PDU topology.
        """
        cutoff = datetime.now(tz=timezone.utc) - timedelta(minutes=minutes)
        with self._lock:
            peer_ids = [mid for mid in self._telemetry.keys() if mid != miner_id][:limit]
            snapshots = {mid: list(self._telemetry[mid]) for mid in peer_ids}
        return {mid: _filter_since(samples, cutoff) for mid, samples in snapshots.items()}


def _filter_since(samples: Iterable[dict[str, Any]], cutoff: datetime) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in samples:
        ts_raw = s.get("ts")
        if not isinstance(ts_raw, str):
            continue
        try:
            ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        except ValueError:
            continue
        if ts >= cutoff:
            out.append(s)
    return out
