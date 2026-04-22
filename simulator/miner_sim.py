"""Single-miner physics model.

Calibration target: Antminer S19j Pro class ASIC.
  - Nominal hashrate: ~104 TH/s (balanced mode)
  - Nominal power: ~3250 W (balanced mode)
  - Chip voltage (12V rail): ~12.0 V nominal
  - Chip temp: ~75-80 °C at nominal ambient and good cooling
  - Fan RPM: 4 fans, 5000-7000 RPM range

Three operating modes:
  turbo:    115 TH/s target, 3600 W, higher chip temp
  balanced: 104 TH/s target, 3250 W  (default / reference)
  eco:       88 TH/s target, 2700 W, lower chip temp

Per-miner bias: small random offset on hashrate, voltage, and temp — simulates
unit-to-unit manufacturing variance. Seeded per miner_id for reproducibility.

Fault signatures (pre-onset gradual degradation):
  chip_instability: hashrate variance increases, occasional dips; chip temp spikes
  cooling_degradation: fan RPM fails (one fan slows), chip temp rises steadily
  power_sag: voltage drops slowly toward threshold, hashrate becomes unstable
  hashboard_failure: one of the 3 hashboards drops out, hashrate falls ~33%
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# ---------------------------------------------------------------------------
# Operating-mode profiles — calibrated from S19j Pro datasheet
# ---------------------------------------------------------------------------

OperatingMode = Literal["turbo", "balanced", "eco"]

_MODE_HASHRATE_TH: dict[str, float] = {
    "turbo": 115.0,
    "balanced": 104.0,
    "eco": 88.0,
}
_MODE_POWER_W: dict[str, float] = {
    "turbo": 3600.0,
    "balanced": 3250.0,
    "eco": 2700.0,
}
# Relative thermal load vs balanced (chip temp scaling)
_MODE_THERMAL_FACTOR: dict[str, float] = {
    "turbo": 1.10,
    "balanced": 1.00,
    "eco": 0.83,
}

_MINER_MODEL: str = "S19j Pro"
_NOMINAL_VOLTAGE_V: float = 12.0
_VOLTAGE_NOISE_STD_V: float = 0.03  # normal operational jitter

# Thermal model parameters
_CHIP_TEMP_AMBIENT_OFFSET: float = 52.0  # °C above ambient at balanced, nominal fans
_CHIP_TEMP_NOISE_STD: float = 0.4
_THERMAL_INERTIA: float = 0.12  # fraction of previous-tick state retained

# Fan parameters
_FAN_NOMINAL_RPM: float = 5800.0
_FAN_MAX_RPM: float = 7200.0
_FAN_MIN_RPM: float = 4500.0
_FAN_TEMP_SLOPE: float = 35.0  # RPM increase per °C above 70 °C chip temp
_FAN_NOISE_STD: float = 40.0  # per-fan jitter (RPM)

# Uptime: start from a random value simulating miners already running
_UPTIME_INIT_MIN_S: float = 3600.0  # at least 1 hour uptime at sim start
_UPTIME_INIT_MAX_S: float = 1_296_000.0  # up to ~15 days


@dataclass
class MinerState:
    """Internal mutable state of a single miner. Not serialized directly."""

    miner_id: str
    operating_mode: OperatingMode = "balanced"

    # Physics state (carry forward between ticks)
    chip_temp_c: float = 75.0
    voltage_v: float = _NOMINAL_VOLTAGE_V
    hashrate_th: float = 104.0
    fan_rpm: list[float] = field(default_factory=lambda: [5800.0, 5800.0, 5800.0, 5800.0])
    uptime_s: float = 0.0

    # Per-miner bias (set once, stays constant through life)
    hashrate_bias: float = 0.0  # TH/s offset
    voltage_bias: float = 0.0  # V offset
    temp_bias: float = 0.0  # °C offset

    # Fault state
    fault_type: str | None = None
    fault_onset_tick: int = 0  # tick number when fault was scheduled
    fault_active_tick: int = 0  # tick number when fault becomes "active"
    current_tick: int = 0

    # Per-miner RNG
    _rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())

    # Thermal inertia: actual temp lags the "target" temp
    _target_chip_temp: float = 75.0

    def __post_init__(self) -> None:
        # Set plausible initial chip temp
        self._target_chip_temp = self.chip_temp_c


def make_miner(
    miner_id: str,
    operating_mode: OperatingMode = "balanced",
    seed: int | None = None,
) -> MinerState:
    """Create a MinerState with per-miner bias and random initial uptime.

    Args:
        miner_id: canonical id like 'm042'.
        operating_mode: initial operating mode.
        seed: RNG seed. If None, derived from miner_id hash so each miner is
            reproducible given the same fleet seed.

    Returns:
        Initialized MinerState.
    """
    # Derive a deterministic seed from the miner index if not provided
    effective_seed = seed if seed is not None else (abs(hash(miner_id)) % (2**32))
    rng = np.random.default_rng(effective_seed)

    # Per-miner variance calibrated from empirical S19j Pro unit spread
    hashrate_bias = float(rng.normal(0.0, 1.8))  # ±1.8 TH/s 1-sigma
    voltage_bias = float(rng.normal(0.0, 0.04))  # ±40 mV 1-sigma
    temp_bias = float(rng.normal(0.0, 1.5))  # ±1.5°C 1-sigma

    nominal_hr = _MODE_HASHRATE_TH[operating_mode]
    nominal_temp = (
        25.0  # some ambient
        + _CHIP_TEMP_AMBIENT_OFFSET * _MODE_THERMAL_FACTOR[operating_mode]
        + temp_bias
    )
    uptime_s = float(rng.uniform(_UPTIME_INIT_MIN_S, _UPTIME_INIT_MAX_S))

    state = MinerState(
        miner_id=miner_id,
        operating_mode=operating_mode,
        chip_temp_c=nominal_temp,
        voltage_v=_NOMINAL_VOLTAGE_V + voltage_bias,
        hashrate_th=nominal_hr + hashrate_bias,
        uptime_s=uptime_s,
        hashrate_bias=hashrate_bias,
        voltage_bias=voltage_bias,
        temp_bias=temp_bias,
        _rng=rng,
        _target_chip_temp=nominal_temp,
    )
    return state


# ---------------------------------------------------------------------------
# Physics tick
# ---------------------------------------------------------------------------


def tick_miner(
    state: MinerState,
    ambient_temp_c: float,
    tick_interval_s: float = 5.0,
    aging_factor: float = 1.0,
) -> dict:
    """Advance one miner by one tick and return the telemetry payload dict.

    Args:
        state: current MinerState (mutated in place).
        ambient_temp_c: current room temperature feeding the cooling system.
        tick_interval_s: simulated seconds per tick (used for uptime increment).
        aging_factor: 1.0 = no aging; < 1.0 = degraded performance from long
            run. Slowly drifts hashrate down by this factor.

    Returns:
        Dict matching TelemetryTick.data schema (without env / fault_injected —
        those are added by the fleet layer).
    """
    rng = state._rng
    mode = state.operating_mode
    state.current_tick += 1
    state.uptime_s += tick_interval_s

    # --- nominal values for current mode ---
    nominal_hr = _MODE_HASHRATE_TH[mode]
    nominal_power = _MODE_POWER_W[mode]
    thermal_factor = _MODE_THERMAL_FACTOR[mode]

    # --- fault modifiers (pre-onset + active) ---
    fault_tag = state.fault_type
    fault_hr_modifier = 1.0
    fault_voltage_modifier = 0.0
    fault_temp_add = 0.0
    fault_fan_fail_idx: int | None = None  # which fan is degrading, if any

    if fault_tag is not None:
        # Pre-onset ramp: normalised 0→1 as tick approaches active
        onset_window = max(1, state.fault_active_tick - state.fault_onset_tick)
        ticks_since_onset = state.current_tick - state.fault_onset_tick
        # pre-onset ratio: 0 at fault_onset_tick, 1 at fault_active_tick
        pre_ratio = float(np.clip(ticks_since_onset / onset_window, 0.0, 1.0))
        # post-active: fully active beyond fault_active_tick
        is_active = state.current_tick >= state.fault_active_tick

        if fault_tag == "chip_instability":
            # Hashrate becomes increasingly variable; chip temp spikes
            noise_amp = 3.0 + pre_ratio * 8.0  # grows from 3 → 11 TH/s noise
            fault_hr_modifier = 1.0 - pre_ratio * 0.05  # small mean drift down
            if is_active:
                fault_hr_modifier = 0.75  # 25% effective drop when active
                noise_amp = 15.0
            # Apply extra noise inline (in addition to normal noise below)
            fault_chip_instability_noise = float(rng.normal(0.0, noise_amp))
        else:
            fault_chip_instability_noise = 0.0

        if fault_tag == "cooling_degradation":
            fault_fan_fail_idx = 2  # fan index 2 degrades
            fault_temp_add = pre_ratio * 8.0  # chip temp rises gradually
            if is_active:
                fault_temp_add = 12.0 + float(rng.uniform(0.0, 3.0))

        if fault_tag == "power_sag":
            # Voltage droops; hashrate becomes unstable at low V
            fault_voltage_modifier = -pre_ratio * 0.35  # up to -350 mV drop
            if is_active:
                fault_voltage_modifier = -0.45 + float(rng.normal(0.0, 0.04))
            fault_hr_modifier = max(0.80, 1.0 - pre_ratio * 0.12)

        if fault_tag == "hashboard_failure":
            # One of 3 hashboards dies → hashrate drops ~33%
            fault_hr_modifier = 1.0 - pre_ratio * 0.30  # gradual drop to 70%
            if is_active:
                fault_hr_modifier = 0.67  # exactly 2/3 hashboards working
    else:
        fault_chip_instability_noise = 0.0

    # --- compute hashrate ---
    hr_noise = float(rng.normal(0.0, 0.4))  # normal operational noise
    # Aging: very slow drift (~0.01% per long-run tick, negligible short-term)
    effective_nominal_hr = nominal_hr * aging_factor + state.hashrate_bias
    hashrate = (
        effective_nominal_hr * fault_hr_modifier
        + hr_noise
        + fault_chip_instability_noise
    )
    hashrate = float(np.clip(hashrate, 1.0, nominal_hr * 1.05))
    state.hashrate_th = hashrate

    # --- compute voltage ---
    v_noise = float(rng.normal(0.0, _VOLTAGE_NOISE_STD_V))
    voltage = _NOMINAL_VOLTAGE_V + state.voltage_bias + fault_voltage_modifier + v_noise
    voltage = float(np.clip(voltage, 10.5, 13.5))
    state.voltage_v = voltage

    # --- compute power ---
    # Power scales slightly with actual vs nominal hashrate (chip load)
    power_ratio = hashrate / max(effective_nominal_hr, 1.0)
    power = nominal_power * power_ratio + float(rng.normal(0.0, 15.0))
    power = float(np.clip(power, 500.0, nominal_power * 1.12))

    # --- compute chip temperature ---
    # Target temp: ambient + fixed rise due to power load + mode factor + bias
    power_rise = (power / nominal_power) * _CHIP_TEMP_AMBIENT_OFFSET * thermal_factor
    target_temp = ambient_temp_c + power_rise + state.temp_bias + fault_temp_add
    target_temp = float(np.clip(target_temp, ambient_temp_c + 5.0, 105.0))

    # Thermal inertia: blend current toward target
    state._target_chip_temp = target_temp
    blended = (1.0 - _THERMAL_INERTIA) * target_temp + _THERMAL_INERTIA * state.chip_temp_c
    temp_noise = float(rng.normal(0.0, _CHIP_TEMP_NOISE_STD))
    chip_temp = blended + temp_noise
    chip_temp = float(np.clip(chip_temp, ambient_temp_c + 2.0, 105.0))
    state.chip_temp_c = chip_temp

    # --- compute fan RPM ---
    fan_rpms: list[int] = []
    for i in range(4):
        base_rpm = _FAN_NOMINAL_RPM
        # Fan responds to chip temp
        temp_delta = max(0.0, chip_temp - 70.0)
        target_rpm = base_rpm + _FAN_TEMP_SLOPE * temp_delta
        target_rpm = float(np.clip(target_rpm, _FAN_MIN_RPM, _FAN_MAX_RPM))
        fan_noise = float(rng.normal(0.0, _FAN_NOISE_STD))

        if fault_fan_fail_idx is not None and i == fault_fan_fail_idx:
            # Degrading fan: RPM drops toward 0 as fault progresses
            fan_ticks = state.current_tick - state.fault_onset_tick
            onset_window = max(1, state.fault_active_tick - state.fault_onset_tick)
            ratio = float(np.clip(fan_ticks / onset_window, 0.0, 1.0))
            is_active = state.current_tick >= state.fault_active_tick
            if is_active:
                target_rpm = float(rng.uniform(500.0, 1200.0))  # failed fan stutters
            else:
                target_rpm = target_rpm * (1.0 - ratio * 0.65)

        rpm_val = int(np.clip(target_rpm + fan_noise, 0.0, _FAN_MAX_RPM))
        fan_rpms.append(rpm_val)

    state.fan_rpm = [float(r) for r in fan_rpms]

    return {
        "miner_id": state.miner_id,
        "miner_model": _MINER_MODEL,
        "hashrate_th": round(hashrate, 2),
        "hashrate_expected_th": round(_MODE_HASHRATE_TH[mode], 1),
        "temp_chip_c": round(chip_temp, 2),
        "temp_amb_c": round(ambient_temp_c, 2),
        "power_w": round(power, 1),
        "voltage_v": round(voltage, 3),
        "fan_rpm": fan_rpms,
        "operating_mode": mode,
        "uptime_s": round(state.uptime_s, 0),
    }
