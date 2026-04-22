"""Smoke test: one tick → valid TelemetryTick envelope per miner.

Run with:
    pytest tests/test_simulator_smoke.py -v

Requires pydantic (and all project deps) to be installed:
    pip install -e ".[dev]"
    # or
    uv pip install -e ".[dev]"
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from shared.schemas.events import EnvBlock, TelemetryTick, parse_event
from simulator.runner import make_simulator_state, simulate_one_tick


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_fleet_state(tmp_path):
    """5-miner fleet with fixed seed, output redirected to a temp file."""
    out_file = tmp_path / "telemetry.jsonl"
    state = make_simulator_state(
        n_miners=5,
        tick_interval_s=5.0,
        fault_injection_enabled=True,
        output_stream=str(out_file),
        seed=42,
    )
    return state, out_file


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestOneTick:
    """One tick produces N valid TelemetryTick envelope per miner."""

    def test_returns_n_events(self, small_fleet_state):
        state, _ = small_fleet_state
        tick_time = datetime(2026, 4, 20, 15, 30, 45, tzinfo=timezone.utc)
        events = simulate_one_tick(state, tick_time)
        assert len(events) == 5

    def test_each_event_is_valid_envelope(self, small_fleet_state):
        state, _ = small_fleet_state
        tick_time = datetime(2026, 4, 20, 15, 30, 45, tzinfo=timezone.utc)
        events = simulate_one_tick(state, tick_time)
        for ev in events:
            # Should parse without raising
            envelope, payload = parse_event(ev)
            assert envelope.event == "telemetry_tick"
            assert envelope.source == "simulator"

    def test_payload_fields_match_schema(self, small_fleet_state):
        state, _ = small_fleet_state
        tick_time = datetime(2026, 4, 20, 15, 30, 45, tzinfo=timezone.utc)
        events = simulate_one_tick(state, tick_time)
        for ev in events:
            _, payload = parse_event(ev)
            tick: TelemetryTick = payload  # type: ignore[assignment]
            assert tick.miner_model == "S19j Pro"
            assert len(tick.fan_rpm) == 4
            assert tick.operating_mode in ("turbo", "balanced", "eco")

    def test_miner_id_format(self, small_fleet_state):
        """miner_id must match m\\d{3} pattern."""
        import re

        state, _ = small_fleet_state
        tick_time = datetime(2026, 4, 20, 15, 30, 45, tzinfo=timezone.utc)
        events = simulate_one_tick(state, tick_time)
        pattern = re.compile(r"^m\d{3}$")
        for ev in events:
            _, payload = parse_event(ev)
            assert pattern.match(payload.miner_id), f"Bad miner_id: {payload.miner_id}"  # type: ignore[union-attr]

    def test_events_written_to_file(self, small_fleet_state):
        state, out_file = small_fleet_state
        tick_time = datetime(2026, 4, 20, 15, 30, 45, tzinfo=timezone.utc)
        simulate_one_tick(state, tick_time)
        assert out_file.exists()
        lines = out_file.read_text().splitlines()
        # 5 miners → 5 telemetry lines (live.jsonl gets another copy but not in out_file)
        assert len(lines) == 5

    def test_each_file_line_is_valid_json(self, small_fleet_state):
        state, out_file = small_fleet_state
        tick_time = datetime(2026, 4, 20, 15, 30, 45, tzinfo=timezone.utc)
        simulate_one_tick(state, tick_time)
        for line in out_file.read_text().splitlines():
            obj = json.loads(line)
            assert "event" in obj
            assert "ts" in obj
            assert "source" in obj
            assert "data" in obj


# ---------------------------------------------------------------------------
# Fault injection
# ---------------------------------------------------------------------------


class TestFaultInjection:
    """Fault injection smoke tests."""

    def test_fault_injected_null_before_onset(self):
        """Before the fault onset tick, fault_injected must be null."""
        state = make_simulator_state(
            n_miners=1,
            tick_interval_s=5.0,
            fault_injection_enabled=True,
            seed=42,
        )
        # Find the miner with a fault scheduled
        miner = state.fleet.miners[0]
        if miner.fault_type is None:
            pytest.skip("No fault scheduled for this seed/miner combo")

        # Run ticks up to onset - 1; fault_injected should be null the whole time
        tick_time = datetime(2026, 4, 20, 10, 0, 0, tzinfo=timezone.utc)
        from datetime import timedelta

        for _ in range(miner.fault_onset_tick - 1):
            events = simulate_one_tick(state, tick_time)
            tick_time += timedelta(seconds=5)
            _, payload = parse_event(events[0])
            assert payload.fault_injected is None, (  # type: ignore[union-attr]
                f"fault_injected non-null at tick {miner.current_tick} before onset {miner.fault_onset_tick}"
            )

    def test_fault_injected_non_null_at_onset(self):
        """At and after onset, fault_injected must carry the fault tag."""
        # Use a fixed fleet where we know m001 has a fault via known seed
        state = make_simulator_state(
            n_miners=3,
            tick_interval_s=5.0,
            fault_injection_enabled=True,
            seed=100,
        )
        # Force a fault on miner 0 for test predictability
        miner = state.fleet.miners[0]
        miner.fault_type = "power_sag"
        miner.fault_onset_tick = 3
        miner.fault_active_tick = 20

        tick_time = datetime(2026, 4, 20, 10, 0, 0, tzinfo=timezone.utc)
        from datetime import timedelta

        # Tick 1, 2: before onset → null
        for _ in range(2):
            events = simulate_one_tick(state, tick_time)
            tick_time += timedelta(seconds=5)
        _, payload = parse_event(events[0])
        assert payload.fault_injected is None  # type: ignore[union-attr]

        # Tick 3: onset → non-null
        events = simulate_one_tick(state, tick_time)
        _, payload = parse_event(events[0])
        assert payload.fault_injected == "power_sag"  # type: ignore[union-attr]

    def test_no_faults_when_disabled(self):
        """With fault_injection_enabled=False, fault_injected is always null."""
        state = make_simulator_state(
            n_miners=10,
            tick_interval_s=5.0,
            fault_injection_enabled=False,
            seed=42,
        )
        tick_time = datetime(2026, 4, 20, 10, 0, 0, tzinfo=timezone.utc)
        from datetime import timedelta

        for i in range(50):
            events = simulate_one_tick(state, tick_time)
            tick_time += timedelta(seconds=5)
            for ev in events:
                _, payload = parse_event(ev)
                assert payload.fault_injected is None, (  # type: ignore[union-attr]
                    f"Unexpected fault at tick {i}: {payload.fault_injected}"
                )


# ---------------------------------------------------------------------------
# Seed reproducibility
# ---------------------------------------------------------------------------


class TestSeedReproducibility:
    """Same seed → identical telemetry sequences."""

    def _run_n_ticks(self, seed: int, n_ticks: int, n_miners: int = 3) -> list[float]:
        """Run n_ticks, collect all hashrate values, return as flat list."""
        state = make_simulator_state(
            n_miners=n_miners,
            tick_interval_s=5.0,
            fault_injection_enabled=False,
            seed=seed,
        )
        tick_time = datetime(2026, 4, 20, 10, 0, 0, tzinfo=timezone.utc)
        from datetime import timedelta

        hashrates = []
        for _ in range(n_ticks):
            events = simulate_one_tick(state, tick_time)
            tick_time += timedelta(seconds=5)
            for ev in events:
                _, payload = parse_event(ev)
                hashrates.append(payload.hashrate_th)  # type: ignore[union-attr]
        return hashrates

    def test_same_seed_same_sequence(self):
        seq_a = self._run_n_ticks(seed=42, n_ticks=10)
        seq_b = self._run_n_ticks(seed=42, n_ticks=10)
        assert seq_a == seq_b, "Same seed must produce identical sequences"

    def test_different_seeds_different_sequence(self):
        seq_a = self._run_n_ticks(seed=42, n_ticks=10)
        seq_b = self._run_n_ticks(seed=99, n_ticks=10)
        # Extremely unlikely to be equal; if so, seeds happen to collide — acceptable
        assert seq_a != seq_b, "Different seeds should (almost certainly) differ"


# ---------------------------------------------------------------------------
# Environmental drift
# ---------------------------------------------------------------------------


class TestEnvironmentalDrift:
    """EnvBlock fields change over time and stay in bounds."""

    def test_env_evolves(self):
        """After many ticks, env values should have drifted from initial."""
        state = make_simulator_state(
            n_miners=1,
            tick_interval_s=5.0,
            seed=7,
        )
        tick_time = datetime(2026, 4, 20, 0, 0, 0, tzinfo=timezone.utc)
        from datetime import timedelta

        # First tick values
        events = simulate_one_tick(state, tick_time)
        _, p0 = parse_event(events[0])
        initial_temp = p0.env.site_temp_c  # type: ignore[union-attr]

        # Run 720 ticks (~1 hour)
        for _ in range(720):
            tick_time += timedelta(seconds=5)
            events = simulate_one_tick(state, tick_time)

        _, p_last = parse_event(events[0])
        final_temp = p_last.env.site_temp_c  # type: ignore[union-attr]

        # Temperature should have changed (diurnal cycle)
        assert initial_temp != final_temp

    def test_env_bounds(self):
        """EnvBlock fields stay within physically plausible bounds."""
        state = make_simulator_state(
            n_miners=2,
            tick_interval_s=5.0,
            seed=13,
        )
        tick_time = datetime(2026, 4, 20, 0, 0, 0, tzinfo=timezone.utc)
        from datetime import timedelta

        for _ in range(500):
            events = simulate_one_tick(state, tick_time)
            tick_time += timedelta(seconds=5)
            for ev in events:
                _, p = parse_event(ev)
                env = p.env  # type: ignore[union-attr]
                assert 10.0 <= env.site_temp_c <= 40.0, f"temp out of range: {env.site_temp_c}"
                assert 0.0 <= env.site_humidity_pct <= 100.0
                assert env.elec_price_usd_kwh > 0
                assert env.hashprice_usd_per_th_day > 0

    def test_physics_bounds(self):
        """Per-miner physics values stay within physically plausible ranges."""
        state = make_simulator_state(
            n_miners=5,
            tick_interval_s=5.0,
            seed=21,
            fault_injection_enabled=False,
        )
        tick_time = datetime(2026, 4, 20, 12, 0, 0, tzinfo=timezone.utc)
        from datetime import timedelta

        for _ in range(100):
            events = simulate_one_tick(state, tick_time)
            tick_time += timedelta(seconds=5)
            for ev in events:
                _, p = parse_event(ev)
                tick: TelemetryTick = p  # type: ignore[assignment]
                assert 50.0 <= tick.hashrate_th <= 130.0, f"hashrate: {tick.hashrate_th}"
                assert 55.0 <= tick.temp_chip_c <= 105.0, f"chip_temp: {tick.temp_chip_c}"
                assert 10.5 <= tick.voltage_v <= 13.5, f"voltage: {tick.voltage_v}"
                assert all(0 <= r <= 7500 for r in tick.fan_rpm), f"fans: {tick.fan_rpm}"
