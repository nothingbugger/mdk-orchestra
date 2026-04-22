"""Generate a simulated telemetry dataset for XGBoost training.

Runs the fleet simulator at accelerated speed and writes all telemetry ticks
to a dedicated JSONL file. This is consumed by train.py via --stream.

Default settings (from the training pipeline spec):
  - 50 miners
  - 4 hours simulated time (at 5s/tick = 2880 ticks/miner)
  - Speed multiplier: 20x (wall ~12 min for 4h sim)
  - Fault mix: balanced (round-robin over all fault types)
  - Seed: 7

Usage:
    python3 scripts/generate_training_data.py
    python3 scripts/generate_training_data.py --hours 8 --out /tmp/train_8h.jsonl
    python3 scripts/generate_training_data.py --hours 4 --seed 7 --miners 50 --out /tmp/train.jsonl
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structlog

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)

log = structlog.get_logger(__name__)

TICK_INTERVAL_S: float = 5.0  # simulated seconds per tick


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate simulated telemetry for XGB training")
    p.add_argument("--hours", type=float, default=4.0, help="Simulated hours to generate (default: 4)")
    p.add_argument("--miners", type=int, default=50, help="Number of miners (default: 50)")
    p.add_argument("--seed", type=int, default=7, help="RNG seed (default: 7)")
    p.add_argument(
        "--fault-mix",
        default="balanced",
        choices=["balanced", "random"],
        help="Fault type distribution (default: balanced)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output JSONL path. Defaults to /tmp/mdk_train_data/telemetry.jsonl",
    )
    p.add_argument(
        "--no-speed-limit",
        action="store_true",
        default=False,
        help="Run as fast as possible (no wall-clock throttling)",
    )
    return p.parse_args(argv)


def generate(
    hours: float = 4.0,
    n_miners: int = 50,
    seed: int = 7,
    fault_mix: str = "balanced",
    out_path: Path | None = None,
    no_speed_limit: bool = True,
) -> Path:
    """Run the simulator and write telemetry to a JSONL file.

    Args:
        hours: simulated hours to produce.
        n_miners: fleet size.
        seed: RNG master seed.
        fault_mix: 'balanced' or 'random'.
        out_path: output file. Created (with parents) if missing.
        no_speed_limit: if True, run at max CPU speed. If False, pace at 20x.

    Returns:
        Path to the written JSONL file.
    """
    from simulator.fleet_sim import make_fleet, tick_fleet

    if out_path is None:
        out_path = Path("/tmp/mdk_train_data/telemetry.jsonl")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Truncate to fresh start
    out_path.write_text("")

    n_ticks = int(hours * 3600 / TICK_INTERVAL_S)
    log.info(
        "generate.starting",
        hours=hours,
        n_miners=n_miners,
        seed=seed,
        fault_mix=fault_mix,
        n_ticks=n_ticks,
        out_path=str(out_path),
    )

    fleet = make_fleet(n_miners=n_miners, seed=seed, fault_injection_enabled=True, fault_mix=fault_mix)

    sim_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    wall_start = time.monotonic()

    # Speed multiplier for optional pacing (20x real-time)
    speed_mult = 20.0
    sim_dt_per_wall_s = TICK_INTERVAL_S * speed_mult  # sim-s of progress per wall-s

    ticks_written = 0
    for tick_idx in range(n_ticks):
        events = tick_fleet(fleet, sim_time, tick_interval_s=TICK_INTERVAL_S, output_stream_path=out_path)
        ticks_written += len(events)
        sim_time = datetime.fromtimestamp(
            sim_time.timestamp() + TICK_INTERVAL_S, tz=timezone.utc
        )

        if not no_speed_limit:
            # Pace to 20x real-time
            expected_wall = (tick_idx + 1) * TICK_INTERVAL_S / speed_mult
            actual_wall = time.monotonic() - wall_start
            if actual_wall < expected_wall:
                time.sleep(expected_wall - actual_wall)

        if (tick_idx + 1) % 500 == 0:
            elapsed = time.monotonic() - wall_start
            pct = (tick_idx + 1) / n_ticks * 100
            log.info(
                "generate.progress",
                tick=tick_idx + 1,
                of=n_ticks,
                pct=round(pct, 1),
                wall_elapsed_s=round(elapsed, 1),
                events_written=ticks_written,
            )

    elapsed_total = time.monotonic() - wall_start
    log.info(
        "generate.done",
        ticks=n_ticks,
        total_events=ticks_written,
        wall_elapsed_s=round(elapsed_total, 1),
        out_path=str(out_path),
    )
    return out_path


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    out_path = Path(args.out) if args.out else None
    result = generate(
        hours=args.hours,
        n_miners=args.miners,
        seed=args.seed,
        fault_mix=args.fault_mix,
        out_path=out_path,
        no_speed_limit=True,  # always max speed for CI / batch runs
    )
    print(str(result))


if __name__ == "__main__":
    main()
