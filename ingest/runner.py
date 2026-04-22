"""Ingest runner: tails telemetry_tick events and emits kpi_update + fleet_snapshot.

Public API (matches shared/specs/interfaces.md §2):

    run_ingest(input_stream, kpi_output, snapshot_output, snapshot_interval_s)

Architecture
------------
One single-threaded loop tails the telemetry stream.

On each `telemetry_tick`:
  1. Update per-miner rolling window (features.MinerWindow)
  2. Compute TE and HSI via kpi.compute_te / kpi.compute_hsi
  3. Emit kpi_update event to kpi_output stream
  4. Cache latest KPI for fleet snapshot

A background-compatible timer triggers fleet_snapshot emission every
`snapshot_interval_s` — checked on each loop iteration (so snapshot cadence is
bounded by telemetry cadence, acceptable at 10 events/s fleet rate).

Stream paths default to the canonical layout from shared/paths.py.
Override via env vars MDK_STREAM_DIR (see shared/paths.py).
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import structlog

from ingest.features import FleetState, TickSnapshot
from ingest.kpi import compute_hsi, compute_miner_status, compute_te
from ingest.thresholds import STALE_TIMEOUT_S
from shared.event_bus import tail_events, write_event
from shared.paths import stream_paths
from shared.schemas.events import (
    EnvBlock,
    FleetSnapshot,
    FleetSnapshotMiner,
    HsiComponents,
    KpiUpdate,
    TeComponents,
    TelemetryTick,
)

log = structlog.get_logger("ingest.runner")

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_ingest(
    input_stream: str | None = None,
    kpi_output: str | None = None,
    snapshot_output: str | None = None,
    snapshot_interval_s: float = 1.0,
) -> None:
    """Tail the telemetry stream, compute KPIs, emit kpi_update + fleet_snapshot.

    Runs forever (until process is killed or input stream raises StopIteration).

    Args:
        input_stream:       path to telemetry.jsonl.  None → canonical path.
        kpi_output:         path for kpi_update events.  None → canonical path.
        snapshot_output:    path for fleet_snapshot events.  None → canonical path.
        snapshot_interval_s: minimum seconds between fleet_snapshot emissions.
    """
    sp = stream_paths()
    input_path = Path(input_stream) if input_stream else sp.telemetry
    kpi_path = Path(kpi_output) if kpi_output else sp.kpis
    snap_path = Path(snapshot_output) if snapshot_output else sp.snapshots

    log.info(
        "ingest.start",
        input=str(input_path),
        kpi_output=str(kpi_path),
        snapshot_output=str(snap_path),
        snapshot_interval_s=snapshot_interval_s,
    )

    fleet = FleetState()
    last_snapshot_wall = time.monotonic()

    for env in tail_events(input_path, from_start=False):
        if env.event != "telemetry_tick":
            continue

        # --- parse payload ---
        try:
            tick: TelemetryTick = env.typed_data()  # type: ignore[assignment]
        except Exception as exc:
            log.warning("ingest.bad_tick", error=str(exc))
            continue

        miner_id = tick.miner_id

        # --- update rolling window ---
        window = fleet.get_or_create(miner_id)
        snap = TickSnapshot(
            wall_ts=time.monotonic(),
            hashrate_th=tick.hashrate_th,
            temp_chip_c=tick.temp_chip_c,
            power_w=tick.power_w,
        )
        window.record(snap)

        # --- compute r(t) and update fleet percentile cache ---
        te_val_temp, te_comps_temp = compute_te(
            telemetry=tick.model_dump(),
            fleet_r_p5=math.nan,
            fleet_r_p95=math.nan,
        )
        # Extract r(t) to update fleet cache before percentile computation
        value_usd_day = te_comps_temp["value_usd_day"]
        cost_usd_day = te_comps_temp["cost_usd_day"]
        if cost_usd_day > 0:
            r_t = value_usd_day / cost_usd_day
            fleet.update_r(miner_id, r_t)

        # --- compute TE with live fleet percentiles ---
        r_p5, r_p95 = fleet.fleet_r_percentiles()
        te, te_comps = compute_te(
            telemetry=tick.model_dump(),
            fleet_r_p5=r_p5,
            fleet_r_p95=r_p95,
        )

        # --- compute HSI ---
        hsi, hsi_comps = compute_hsi(telemetry=tick.model_dump(), window=window)

        # --- build and emit kpi_update ---
        kpi_payload = KpiUpdate(
            miner_id=miner_id,
            te=te,
            hsi=hsi,
            te_components=TeComponents(**te_comps),
            hsi_components=HsiComponents(**hsi_comps),
        )
        write_event("kpi_update", "ingest", kpi_payload, stream_path=kpi_path)

        log.debug(
            "ingest.kpi_emitted",
            miner_id=miner_id,
            te=te,
            hsi=hsi,
            warmed_up=fleet.is_warmed_up(),
        )

        # --- cache latest KPI for snapshot ---
        fleet.set_latest_kpi(
            miner_id,
            {
                "te": te,
                "hsi": hsi,
                "status": compute_miner_status(te, hsi),
                "hashrate_th": tick.hashrate_th,
                "temp_chip_c": tick.temp_chip_c,
                "env": tick.env.model_dump(),
            },
        )

        # --- maybe emit fleet snapshot ---
        now = time.monotonic()
        if (now - last_snapshot_wall) >= snapshot_interval_s:
            _emit_snapshot(fleet, snap_path)
            last_snapshot_wall = now


# ---------------------------------------------------------------------------
# Fleet snapshot helper
# ---------------------------------------------------------------------------


def _emit_snapshot(fleet: FleetState, snap_path: Path) -> None:
    """Build and emit a fleet_snapshot event from latest per-miner KPIs."""
    now_wall = time.monotonic()
    cached = fleet.latest_kpis()

    if not cached:
        return

    miners_out: dict[str, FleetSnapshotMiner] = {}
    te_vals: list[float] = []
    hsi_vals: list[float] = []
    last_env: Optional[dict] = None

    for mid, kpi in cached.items():
        window = fleet.windows.get(mid)
        # Determine status — check staleness first
        if window is not None and window.is_stale(now_wall):
            status = "shut"
        else:
            status = kpi["status"]

        miners_out[mid] = FleetSnapshotMiner(
            te=kpi["te"],
            hsi=kpi["hsi"],
            status=status,
            hashrate_th=kpi["hashrate_th"],
            temp_chip_c=kpi["temp_chip_c"],
        )
        te_vals.append(kpi["te"])
        hsi_vals.append(kpi["hsi"])
        last_env = kpi.get("env")

    fleet_te = sum(te_vals) / len(te_vals) if te_vals else 0.0
    fleet_hsi = sum(hsi_vals) / len(hsi_vals) if hsi_vals else 0.0

    # Fall back to empty env block if no env seen yet
    if last_env is None:
        last_env = {
            "site_temp_c": 0.0,
            "site_humidity_pct": 0.0,
            "elec_price_usd_kwh": 0.0,
            "hashprice_usd_per_th_day": 0.0,
        }

    snapshot_payload = FleetSnapshot(
        miner_count=len(miners_out),
        miners=miners_out,
        fleet_te=round(fleet_te, 2),
        fleet_hsi=round(fleet_hsi, 2),
        env=EnvBlock(**last_env),
    )

    write_event("fleet_snapshot", "ingest", snapshot_payload, stream_path=snap_path)
    log.debug(
        "ingest.snapshot_emitted",
        miner_count=len(miners_out),
        fleet_te=round(fleet_te, 2),
        fleet_hsi=round(fleet_hsi, 2),
    )
