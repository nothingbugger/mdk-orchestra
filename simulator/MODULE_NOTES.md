# simulator — MODULE_NOTES

**Written by**: Claude Code Sonnet 4.6 (feat/simulator session)
**Date**: 2026-04-20
**Status**: complete, syntax-verified, pending full dep-install run

---

## What was built

| File | Purpose |
|------|---------|
| `simulator/environmental.py` | Ambient + grid price feed: diurnal temp, humidity, elec price, hashprice |
| `simulator/miner_sim.py` | Single-miner physics model: S19j Pro calibrated hashrate/power/temp/fans |
| `simulator/fleet_sim.py` | Fleet orchestrator: 50-miner pool, fault scheduler, tick_fleet() |
| `simulator/runner.py` | Public API: `run_simulator()`, `simulate_one_tick()`, `make_simulator_state()` |
| `simulator/main.py` | CLI entry point: `python -m simulator.main` |
| `simulator/__init__.py` | Package marker + usage docstring |
| `simulator/README.md` | User-facing documentation |
| `tests/test_simulator_smoke.py` | Unit tests (happy path + edge cases) |

---

## Interface conformance

### interfaces.md §1 compliance

| Spec requirement | Status | Notes |
|-----------------|--------|-------|
| `run_simulator(n_miners, tick_interval_s, duration_s, fault_injection_enabled, output_stream, seed)` | COMPLIANT | Adds `wall_speed_factor` and `sim_start_time` (non-breaking additions) |
| `simulate_one_tick(state: SimulatorState, tick_time: datetime) -> list[dict]` | COMPLIANT | Exact signature |
| Output: `telemetry_tick` schema events | COMPLIANT | Pydantic TelemetryTick validated at emit time |

### event_schemas.md compliance

| Field | Status |
|-------|--------|
| `miner_id: m\d{3}` | COMPLIANT |
| `miner_model` | COMPLIANT ("S19j Pro") |
| `hashrate_th`, `hashrate_expected_th` | COMPLIANT |
| `temp_chip_c`, `temp_amb_c` | COMPLIANT |
| `power_w`, `voltage_v` | COMPLIANT |
| `fan_rpm: list[int], len=4` | COMPLIANT |
| `operating_mode: turbo/balanced/eco` | COMPLIANT |
| `uptime_s` | COMPLIANT |
| `env: EnvBlock` | COMPLIANT (site_temp_c, site_humidity_pct, elec_price_usd_kwh, hashprice_usd_per_th_day) |
| `fault_injected: str | null` | COMPLIANT (null before onset, tag string after) |

---

## Design decisions

### 1. Dataclass for internal state, Pydantic only at emit boundary
`MinerState` and `FleetState` are `@dataclass` for speed. The `TelemetryTick`
pydantic model is constructed exactly once per tick per miner at emit time.
This keeps the hot physics path free from pydantic overhead.

### 2. Per-miner RNG derived from master seed
Each miner's RNG is seeded by `master_rng.integers(0, 2^31)` during fleet
construction. This means the same `seed` → identical per-miner variance →
identical telemetry sequences, even if the number of miners changes (since the
master RNG is consumed in a fixed order: miners 1..N, then env, then fleet).

### 3. Fault scheduling is pre-computed at fleet creation
All fault onset/active ticks are fixed at `make_fleet()` time. This allows
the A/B runner to use the same `FleetState` for both tracks (just don't read
`fault_injected` in the baseline track).

### 4. `write_event` routes to canonical stream + live.jsonl
The event bus writes `telemetry.jsonl` + `live.jsonl` by default (per bus
implementation). The `output_stream` override only overrides the primary file;
`live.jsonl` continues to receive a copy unless the override path IS `live`.
This matches the expected dashboard behavior.

---

## Deviations from spec

1. **`run_simulator` signature extended**: Added `wall_speed_factor` and
   `sim_start_time` parameters (both optional, default to spec values).
   Non-breaking; other modules that call `run_simulator` can ignore them.
   Spec says "internals not constrained" so this is acceptable.

2. **`runner.py` not `simulator/runner.py` at module root**: `interfaces.md`
   says `# simulator/runner.py` but does not mandate a specific sub-path.
   The file is at `simulator/runner.py`, which is `simulator/runner.py` from
   the repo root. Import: `from simulator.runner import run_simulator`. FINE.

3. **Aging factor unimplemented**: `tick_miner()` accepts `aging_factor` but
   `fleet_sim.py` does not yet derive it from uptime. All miners run at 1.0.
   Future enhancement.

4. **Fan RPM list type**: spec says `list[int]`, pydantic model says `list[int]`.
   Internal computation uses floats then casts. Compliant at emit boundary.

---

## Open questions

1. **Operating mode changes**: should the orchestrator be able to command mode
   changes at runtime (e.g., via an action feed)? Currently modes are fixed.
   If yes, `MinerState.operating_mode` just needs to be mutated before the
   next tick — the physics model already handles it. Left as future work.

2. **Fault persistence**: after a fault activates fully, it never "heals".
   The spec doesn't define fault lifecycle beyond the pre-onset window. For the
   A/B experiment this is fine (fault window is the detection window). If a
   restart/repair simulation is needed, add a `fault_clear_tick` field.

3. **Multi-tick gap between miner emissions**: all miners in a tick are emitted
   in a tight loop with the same simulated `tick_time`. In a real system each
   miner would emit at slightly different wall-clock moments. For dashboard
   purposes this is acceptable (50 × 5s ticks in ~50ms real time at 10×
   speed). If strict per-miner jitter is needed, add a sub-tick offset.

---

## What would need to change for production

- Replace the flat JSONL bus with Kafka / NATS for multi-process scaling
- Add a state persistence layer (Parquet snapshot) so the sim can resume
- Add dynamic mode commands from the action module
- Model per-miner chip wear (aging_factor derived from uptime + thermal stress)
- Model cluster-level thermal coupling (hot miner raises neighbor ambient temp)
