"""Public API for the simulator module.

Implements the interface defined in `shared/specs/interfaces.md §1`:
  - run_simulator(...)  — blocking loop
  - simulate_one_tick(...) — single-tick helper for tests / A/B runs

The simulator module is runnable in isolation:
  python -m simulator.main [--n-miners N] [--seed S] [--speed F]
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import structlog

from simulator.fleet_sim import FleetState, make_fleet, tick_fleet

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# SimulatorState — opaque carrier passed to simulate_one_tick
# ---------------------------------------------------------------------------


@dataclass
class SimulatorState:
    """Wrapper around FleetState exposed as part of the public API.

    Callers should create this via `make_simulator_state(...)` rather than
    constructing FleetState directly.
    """

    fleet: FleetState
    tick_interval_s: float
    simulated_time: datetime
    output_stream: str | None  # None = canonical default from shared.paths


def make_simulator_state(
    n_miners: int = 50,
    tick_interval_s: float = 5.0,
    fault_injection_enabled: bool = True,
    output_stream: str | None = None,
    seed: int | None = None,
    sim_start_time: datetime | None = None,
    fault_mix: str = "random",
) -> SimulatorState:
    """Create an initialized SimulatorState ready for simulate_one_tick calls.

    Args:
        n_miners: number of miners in the fleet.
        tick_interval_s: simulated seconds per tick.
        fault_injection_enabled: whether to schedule faults.
        output_stream: path to JSONL output file. None = auto-route via paths.
        seed: RNG seed for reproducibility.
        sim_start_time: initial simulated time. Defaults to now (UTC).

    Returns:
        SimulatorState ready for ticking.
    """
    fleet = make_fleet(
        n_miners=n_miners,
        seed=seed,
        fault_injection_enabled=fault_injection_enabled,
        fault_mix=fault_mix,
    )
    start = sim_start_time or datetime.now(tz=timezone.utc)
    return SimulatorState(
        fleet=fleet,
        tick_interval_s=tick_interval_s,
        simulated_time=start,
        output_stream=output_stream,
    )


# ---------------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------------


def simulate_one_tick(
    state: SimulatorState,
    tick_time: datetime,
) -> list[dict]:
    """Compute one tick for all miners, return list of telemetry_tick event dicts.

    This is the low-level building block. `run_simulator` calls this in a loop.
    Also useful for unit tests and A/B runs that want to drive time explicitly.

    Args:
        state: current SimulatorState (mutated — tick count and miner states advance).
        tick_time: the simulated wall-clock datetime for this tick.

    Returns:
        List of event envelope dicts (one per miner), each conforming to the
        telemetry_tick envelope schema.
    """
    return tick_fleet(
        fleet=state.fleet,
        tick_time=tick_time,
        tick_interval_s=state.tick_interval_s,
        output_stream_path=state.output_stream,
    )


def run_simulator(
    n_miners: int = 50,
    tick_interval_s: float = 5.0,
    duration_s: float | None = None,
    fault_injection_enabled: bool = True,
    output_stream: str | None = None,
    seed: int | None = None,
    wall_speed_factor: float = 1.0,
    sim_start_time: datetime | None = None,
    fault_mix: str = "random",
) -> None:
    """Run the simulator loop, emitting telemetry_tick events to output_stream.

    This function blocks until `duration_s` elapses (or forever if None).

    Args:
        n_miners: number of miners to simulate.
        tick_interval_s: simulated seconds between ticks. Default 5 s per spec.
        duration_s: stop after this many simulated seconds. None = run forever.
        fault_injection_enabled: whether to inject pre-failure patterns.
        output_stream: path to output JSONL. None = canonical default.
        seed: RNG seed. None = non-deterministic.
        wall_speed_factor: 1.0 = real time; 10.0 = 10x faster (tick every
            tick_interval_s / wall_speed_factor real seconds).
        sim_start_time: initial simulated datetime. None = now (UTC).
    """
    state = make_simulator_state(
        n_miners=n_miners,
        tick_interval_s=tick_interval_s,
        fault_injection_enabled=fault_injection_enabled,
        output_stream=output_stream,
        seed=seed,
        sim_start_time=sim_start_time,
        fault_mix=fault_mix,
    )

    wall_tick_s = tick_interval_s / wall_speed_factor
    simulated_elapsed_s = 0.0
    tick_dt = timedelta(seconds=tick_interval_s)

    logger.info(
        "simulator_started",
        n_miners=n_miners,
        tick_interval_s=tick_interval_s,
        wall_tick_s=wall_tick_s,
        duration_s=duration_s,
        fault_injection=fault_injection_enabled,
        seed=seed,
    )

    try:
        while True:
            tick_start_wall = time.monotonic()
            tick_time = state.simulated_time

            events = simulate_one_tick(state, tick_time)

            state.simulated_time += tick_dt
            simulated_elapsed_s += tick_interval_s

            if state.fleet.tick_count % 60 == 0:  # log every ~5 min sim time
                logger.info(
                    "simulator_tick",
                    tick=state.fleet.tick_count,
                    sim_elapsed_min=round(simulated_elapsed_s / 60, 1),
                    n_events=len(events),
                )

            if duration_s is not None and simulated_elapsed_s >= duration_s:
                logger.info(
                    "simulator_done",
                    ticks=state.fleet.tick_count,
                    sim_elapsed_s=simulated_elapsed_s,
                )
                break

            # Wall-clock pacing
            elapsed_wall = time.monotonic() - tick_start_wall
            sleep_s = max(0.0, wall_tick_s - elapsed_wall)
            if sleep_s > 0:
                time.sleep(sleep_s)

    except KeyboardInterrupt:
        logger.info(
            "simulator_interrupted",
            ticks=state.fleet.tick_count,
            sim_elapsed_s=simulated_elapsed_s,
        )
