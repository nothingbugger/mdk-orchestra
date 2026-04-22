"""Fleet-level simulation orchestrator.

Manages:
- A pool of MinerState objects (one per miner)
- Per-tick state advancement calling tick_miner()
- Fault injection scheduler: pre-schedules faults so each miner can have
  a realistic pre-onset window before the fault fully activates
- Emission of TelemetryTick pydantic events via write_event()

Fault injection design:
  - At simulator start, a random subset of miners is assigned a future fault
    (onset_tick = some tick in the future, active_tick = onset_tick + window).
  - The pre-onset window is fault-type specific (chip_instability: long;
    cooling_degradation: medium; power_sag: long; hashboard_failure: short).
  - After a fault fully activates, it clears after another window to simulate
    recovery (or the miner just keeps degrading — depends on fault type).
  - `fault_injected` in the event is non-null only from fault_onset_tick onwards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import NamedTuple

import numpy as np
import structlog

from shared.event_bus import write_event
from shared.paths import stream_paths
from shared.schemas.events import EnvBlock, TelemetryTick

from simulator.environmental import EnvState, make_env_state
from simulator.miner_sim import MinerState, OperatingMode, make_miner, tick_miner

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Fault configuration
# ---------------------------------------------------------------------------

FAULT_TYPES: list[str] = [
    "chip_instability",
    "cooling_degradation",
    "power_sag",
    "hashboard_failure",
]

# (pre_onset_ticks_min, pre_onset_ticks_max): how many ticks before full activation
_FAULT_ONSET_WINDOWS: dict[str, tuple[int, int]] = {
    "chip_instability": (240, 720),  # 20–60 min of pre-signs at 5s ticks
    "cooling_degradation": (180, 480),  # 15–40 min
    "power_sag": (360, 900),  # 30–75 min
    "hashboard_failure": (60, 240),  # 5–20 min (sudden)
}

# Probability that any given miner gets a fault scheduled at sim start
_FAULT_INJECTION_PROBABILITY: float = 0.10  # ~5 miners out of 50 (paced for live API/LLM runs)

# Earliest tick a fault can start (give the system some warm-up time)
_FAULT_EARLIEST_TICK: int = 60  # 5 min at 5s ticks
_FAULT_LATEST_TICK: int = 720  # 60 min

# Operating mode distribution across fleet
_MODE_DISTRIBUTION: dict[str, float] = {
    "turbo": 0.20,
    "balanced": 0.65,
    "eco": 0.15,
}


# ---------------------------------------------------------------------------
# Fleet state container
# ---------------------------------------------------------------------------


@dataclass
class FleetState:
    """Holds all per-miner states and environmental state. Passed to tick()."""

    miners: list[MinerState]
    env: EnvState
    n_miners: int
    tick_count: int = 0
    _rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())


class FaultSchedule(NamedTuple):
    miner_id: str
    fault_type: str
    onset_tick: int  # tick when pre-onset signals begin
    active_tick: int  # tick when fault fully activates


# ---------------------------------------------------------------------------
# Fleet construction
# ---------------------------------------------------------------------------


def make_fleet(
    n_miners: int = 50,
    seed: int | None = None,
    fault_injection_enabled: bool = True,
    fault_mix: str = "random",
) -> FleetState:
    """Instantiate the full fleet with optional pre-scheduled fault injection.

    Args:
        n_miners: number of miners to simulate.
        seed: master RNG seed. Each miner gets a deterministic sub-seed
            derived from this master seed + miner index.
        fault_injection_enabled: if False, no faults are scheduled.
        fault_mix: "random" (default, backward compatible) or "balanced"
            (round-robin over FAULT_TYPES). Use "balanced" to force a wider
            variety of flag types in a single run — useful for pitch / demo
            runs where the default RNG can cluster on one type.

    Returns:
        FleetState ready for ticking.
    """
    master_rng = np.random.default_rng(seed)

    # --- build miners ---
    miners: list[MinerState] = []
    mode_choices = list(_MODE_DISTRIBUTION.keys())
    mode_probs = [_MODE_DISTRIBUTION[m] for m in mode_choices]

    for i in range(n_miners):
        miner_id = f"m{(i + 1):03d}"
        mode = str(master_rng.choice(mode_choices, p=mode_probs))  # type: ignore[arg-type]
        # Derive a unique seed per miner from the master seed
        miner_seed = int(master_rng.integers(0, 2**31))
        m = make_miner(miner_id, operating_mode=mode, seed=miner_seed)  # type: ignore[arg-type]
        miners.append(m)

    # --- schedule faults ---
    if fault_injection_enabled:
        _schedule_faults(miners, master_rng, fault_mix=fault_mix)

    env_seed = int(master_rng.integers(0, 2**31))
    env = make_env_state(env_seed)
    fleet_rng = np.random.default_rng(int(master_rng.integers(0, 2**31)))

    fleet = FleetState(miners=miners, env=env, n_miners=n_miners, _rng=fleet_rng)
    logger.info(
        "fleet_created",
        n_miners=n_miners,
        faults_scheduled=sum(1 for m in miners if m.fault_type is not None),
        seed=seed,
    )
    return fleet


def _schedule_faults(
    miners: list[MinerState],
    rng: np.random.Generator,
    fault_mix: str = "random",
) -> None:
    """Assign faults to a random subset of miners in-place.

    Each chosen miner gets one fault type with a randomly chosen onset window.
    The fault is pre-onset from onset_tick; fully active at active_tick.

    Args:
        miners: fleet to mutate.
        rng: master RNG (controls selection + timing).
        fault_mix: "random" (per-miner independent type draw) or "balanced"
            (round-robin through FAULT_TYPES, so each type receives ≈ an
            equal share of the faulted-miner budget). Timing stays RNG-driven
            under either mode — only the TYPE assignment differs.
    """
    rr_idx = 0
    for miner in miners:
        if float(rng.random()) > _FAULT_INJECTION_PROBABILITY:
            continue
        if fault_mix == "balanced":
            fault_type = FAULT_TYPES[rr_idx % len(FAULT_TYPES)]
            rr_idx += 1
        else:
            fault_type = str(rng.choice(FAULT_TYPES))
        onset_tick = int(rng.integers(_FAULT_EARLIEST_TICK, _FAULT_LATEST_TICK + 1))
        window_min, window_max = _FAULT_ONSET_WINDOWS[fault_type]
        onset_window = int(rng.integers(window_min, window_max + 1))
        active_tick = onset_tick + onset_window

        miner.fault_type = fault_type
        miner.fault_onset_tick = onset_tick
        miner.fault_active_tick = active_tick
        logger.debug(
            "fault_scheduled",
            miner_id=miner.miner_id,
            fault_type=fault_type,
            onset_tick=onset_tick,
            active_tick=active_tick,
        )


# ---------------------------------------------------------------------------
# Per-tick emission
# ---------------------------------------------------------------------------


def tick_fleet(
    fleet: FleetState,
    tick_time: datetime,
    tick_interval_s: float = 5.0,
    output_stream_path=None,
) -> list[dict]:
    """Advance fleet by one tick, emit TelemetryTick events, return event dicts.

    Args:
        fleet: FleetState to advance (mutated).
        tick_time: simulated wall-clock for this tick.
        tick_interval_s: simulated seconds per tick.
        output_stream_path: override stream path. None = canonical default.

    Returns:
        List of raw event dicts (one per miner).
    """
    fleet.tick_count += 1
    fleet.env.tick(tick_time)
    env_dict = fleet.env.as_dict()
    env_block = EnvBlock(**env_dict)

    emitted: list[dict] = []
    for miner in fleet.miners:
        # Compute telemetry physics
        payload_raw = tick_miner(
            state=miner,
            ambient_temp_c=fleet.env.site_temp_c,
            tick_interval_s=tick_interval_s,
        )

        # Determine fault_injected tag
        fault_tag: str | None = None
        if (
            miner.fault_type is not None
            and miner.current_tick >= miner.fault_onset_tick
        ):
            fault_tag = miner.fault_type

        # Build pydantic model for schema validation
        tick_payload = TelemetryTick(
            miner_id=payload_raw["miner_id"],
            miner_model=payload_raw["miner_model"],
            hashrate_th=payload_raw["hashrate_th"],
            hashrate_expected_th=payload_raw["hashrate_expected_th"],
            temp_chip_c=payload_raw["temp_chip_c"],
            temp_amb_c=payload_raw["temp_amb_c"],
            power_w=payload_raw["power_w"],
            voltage_v=payload_raw["voltage_v"],
            fan_rpm=payload_raw["fan_rpm"],
            operating_mode=payload_raw["operating_mode"],
            uptime_s=payload_raw["uptime_s"],
            env=env_block,
            fault_injected=fault_tag,
        )

        env_kwargs = {}
        if output_stream_path is not None:
            from pathlib import Path

            env_kwargs["stream_path"] = Path(output_stream_path)

        envelope = write_event(
            event="telemetry_tick",
            source="simulator",
            payload=tick_payload,
            **env_kwargs,
        )

        emitted.append(envelope.model_dump(mode="json"))

    return emitted
