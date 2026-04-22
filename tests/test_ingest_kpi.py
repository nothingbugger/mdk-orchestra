"""Unit tests for ingest.kpi: TE computation, HSI monotonicity, status transitions.

Run with:  pytest tests/test_ingest_kpi.py -v
"""

from __future__ import annotations

import math

import pytest

from ingest.features import FleetState, MinerWindow, TickSnapshot
from ingest.kpi import compute_hsi, compute_miner_status, compute_te
from ingest.thresholds import (
    IMM_HSI,
    IMM_TE,
    TE_WARMUP_DEFAULT,
    WARN_HSI,
    WARN_TE,
)


# ---------------------------------------------------------------------------
# Fixtures — canonical telemetry payload
# ---------------------------------------------------------------------------


def _make_telemetry(
    hashrate_th: float = 95.3,
    power_w: float = 3250.0,
    elec_price: float = 0.072,
    hashprice: float = 0.058,
    temp_chip_c: float = 78.5,
) -> dict:
    return {
        "miner_id": "m042",
        "miner_model": "S19j Pro",
        "hashrate_th": hashrate_th,
        "hashrate_expected_th": 104.0,
        "temp_chip_c": temp_chip_c,
        "temp_amb_c": 24.1,
        "power_w": power_w,
        "voltage_v": 12.08,
        "fan_rpm": [5800, 5820, 5790, 5810],
        "operating_mode": "balanced",
        "uptime_s": 432150.0,
        "env": {
            "site_temp_c": 23.5,
            "site_humidity_pct": 42,
            "elec_price_usd_kwh": elec_price,
            "hashprice_usd_per_th_day": hashprice,
        },
        "fault_injected": None,
    }


# ---------------------------------------------------------------------------
# TE — warmup behaviour
# ---------------------------------------------------------------------------


class TestTeWarmup:
    def test_te_warmup_when_percentiles_nan(self) -> None:
        """With NaN percentiles (warmup), TE must equal TE_WARMUP_DEFAULT."""
        tel = _make_telemetry()
        te, comps = compute_te(tel, fleet_r_p5=math.nan, fleet_r_p95=math.nan)
        assert te == TE_WARMUP_DEFAULT

    def test_te_warmup_components_still_computed(self) -> None:
        """Even in warmup, cost/value components should be non-zero."""
        tel = _make_telemetry()
        _, comps = compute_te(tel, fleet_r_p5=math.nan, fleet_r_p95=math.nan)
        assert comps["value_usd_day"] > 0
        assert comps["cost_usd_day"] > 0

    def test_te_warmup_when_p5_equals_p95(self) -> None:
        """Degenerate fleet (all miners identical) → TE = 50."""
        tel = _make_telemetry()
        # p5 == p95 means denom ~0 → should return 50
        te, _ = compute_te(tel, fleet_r_p5=0.5, fleet_r_p95=0.5)
        assert te == 50.0

    def test_fleet_state_not_warmed_up_initially(self) -> None:
        """Fresh FleetState should not report warmed up."""
        fs = FleetState()
        assert not fs.is_warmed_up()


# ---------------------------------------------------------------------------
# TE — golden set (known inputs → expected output range)
# ---------------------------------------------------------------------------


class TestTeGolden:
    """Golden-set tests.  We check output is in [0, 100] and directional."""

    def _r_for_tel(self, tel: dict, sigma: float = 0.0) -> float:
        """Helper: extract r(t) = value / cost from components."""
        _, comps = compute_te(tel, sigma_hash=sigma, fleet_r_p5=1.0, fleet_r_p95=2.0)
        v = comps["value_usd_day"]
        c = comps["cost_usd_day"]
        return v / c if c > 0 else 0.0

    def test_te_in_range(self) -> None:
        tel = _make_telemetry()
        r = self._r_for_tel(tel)
        # Use r as both percentile boundaries to test interior
        te, _ = compute_te(tel, fleet_r_p5=r * 0.5, fleet_r_p95=r * 2.0)
        assert 0.0 <= te <= 100.0

    def test_te_at_p5_is_zero(self) -> None:
        """Miner at fleet p5 → TE = 0."""
        tel = _make_telemetry()
        r = self._r_for_tel(tel)
        te, _ = compute_te(tel, fleet_r_p5=r, fleet_r_p95=r * 4.0)
        assert te == pytest.approx(0.0, abs=0.1)

    def test_te_at_p95_is_100(self) -> None:
        """Miner at fleet p95 → TE = 100."""
        tel = _make_telemetry()
        r = self._r_for_tel(tel)
        te, _ = compute_te(tel, fleet_r_p5=r * 0.25, fleet_r_p95=r)
        assert te == pytest.approx(100.0, abs=0.1)

    def test_te_above_p95_clipped_to_100(self) -> None:
        """Miner above p95 → TE clipped at 100."""
        tel = _make_telemetry()
        r = self._r_for_tel(tel)
        te, _ = compute_te(tel, fleet_r_p5=r * 0.1, fleet_r_p95=r * 0.5)
        assert te == 100.0

    def test_te_below_p5_clipped_to_zero(self) -> None:
        """Miner below p5 → TE clipped at 0."""
        tel = _make_telemetry()
        r = self._r_for_tel(tel)
        te, _ = compute_te(tel, fleet_r_p5=r * 2.0, fleet_r_p95=r * 8.0)
        assert te == 0.0

    def test_higher_hashprice_increases_te(self) -> None:
        """Higher hashprice (more revenue) should increase TE relative to same fleet."""
        tel_low = _make_telemetry(hashprice=0.03)
        tel_high = _make_telemetry(hashprice=0.08)
        r_low = self._r_for_tel(tel_low)
        r_high = self._r_for_tel(tel_high)
        # Fix fleet percentiles using low hashprice as reference
        te_low, _ = compute_te(tel_low, fleet_r_p5=r_low * 0.5, fleet_r_p95=r_low * 2.0)
        te_high, _ = compute_te(tel_high, fleet_r_p5=r_low * 0.5, fleet_r_p95=r_low * 2.0)
        assert te_high > te_low

    def test_higher_elec_price_decreases_te(self) -> None:
        """Higher electricity price (more cost) should decrease r and thus TE."""
        tel_cheap = _make_telemetry(elec_price=0.04)
        tel_exp = _make_telemetry(elec_price=0.12)
        r_cheap = self._r_for_tel(tel_cheap)
        r_exp = self._r_for_tel(tel_exp)
        # Both against same fleet percentiles
        p5 = r_exp * 0.9
        p95 = r_cheap * 1.1
        te_cheap, _ = compute_te(tel_cheap, fleet_r_p5=p5, fleet_r_p95=p95)
        te_exp, _ = compute_te(tel_exp, fleet_r_p5=p5, fleet_r_p95=p95)
        assert te_cheap > te_exp

    def test_higher_sigma_increases_cost_decreases_te(self) -> None:
        """Higher hashrate std (σ_hash) adds to cost → lower TE."""
        tel = _make_telemetry()
        r_stable = self._r_for_tel(tel, sigma=0.0)
        r_noisy = self._r_for_tel(tel, sigma=5.0)
        # p5 / p95 built around stable r
        p5 = r_stable * 0.5
        p95 = r_stable * 2.0
        te_stable, _ = compute_te(tel, sigma_hash=0.0, fleet_r_p5=p5, fleet_r_p95=p95)
        te_noisy, _ = compute_te(tel, sigma_hash=5.0, fleet_r_p5=p5, fleet_r_p95=p95)
        assert te_stable >= te_noisy

    def test_te_components_no_cooling_multiplier(self) -> None:
        """TE v2: p_asic_w is logged as-is, with no cooling multiplier.
        Cooling overhead now enters HSI via the thermal-stress term."""
        tel = _make_telemetry(power_w=3000.0)
        _, comps = compute_te(tel)
        assert "p_cool_w" not in comps
        assert comps["p_asic_w"] == pytest.approx(3000.0, rel=1e-4)
        # Cost = (3000/1000) * rho_elec * 24. With default rho_elec = 0.07
        # (matches _make_telemetry default), cost = 5.04 USD/day.
        assert comps["cost_usd_day"] == pytest.approx(
            (3000.0 / 1000.0) * tel["env"]["elec_price_usd_kwh"] * 24.0,
            rel=1e-4,
        )

    def test_te_ignores_voltage_and_mode(self) -> None:
        """TE v2 depends only on hashrate, hashprice, power, elec_price.
        Voltage and operating_mode must not move the needle.
        """
        tel_a = _make_telemetry()
        tel_a["voltage_v"] = 12.0
        tel_a["operating_mode"] = "balanced"
        tel_b = _make_telemetry()
        tel_b["voltage_v"] = 10.8
        tel_b["operating_mode"] = "turbo"
        te_a, _ = compute_te(tel_a, fleet_r_p5=0.5, fleet_r_p95=2.0)
        te_b, _ = compute_te(tel_b, fleet_r_p5=0.5, fleet_r_p95=2.0)
        assert te_a == te_b


# ---------------------------------------------------------------------------
# HSI — monotonicity
# ---------------------------------------------------------------------------


class TestHsiMonotonicity:
    def _window_at_temp(self, base_temp: float, current_temp: float, n: int = 30) -> MinerWindow:
        """Build a MinerWindow with n ticks at base_temp then one at current_temp."""
        w = MinerWindow(miner_id="m001")
        import time

        t0 = time.monotonic()
        for i in range(n):
            w.record(TickSnapshot(wall_ts=t0 + i * 5, hashrate_th=100.0, temp_chip_c=base_temp, power_w=3250.0))
        # Record current (hot) tick
        w.record(TickSnapshot(wall_ts=t0 + n * 5, hashrate_th=100.0, temp_chip_c=current_temp, power_w=3250.0))
        return w

    def test_hsi_decreases_with_higher_chip_temp(self) -> None:
        """Higher chip temp (above own baseline) → lower HSI."""
        tel = _make_telemetry(temp_chip_c=90.0)  # current temp
        w_low = self._window_at_temp(base_temp=70.0, current_temp=75.0)
        w_high = self._window_at_temp(base_temp=70.0, current_temp=90.0)
        hsi_low, _ = compute_hsi(tel, window=w_low)
        hsi_high, _ = compute_hsi(tel, window=w_high)
        assert hsi_low > hsi_high, f"hsi_low={hsi_low} should > hsi_high={hsi_high}"

    def test_hsi_decreases_with_more_hot_time(self) -> None:
        """More ticks above hot threshold → lower HSI (hot_time_frac driven)."""
        from ingest.thresholds import HOT_TEMP_THRESHOLD_C

        tel = _make_telemetry()
        # Window with no hot ticks
        w_cool = MinerWindow(miner_id="m002")
        # Window with many hot ticks
        w_hot = MinerWindow(miner_id="m003")
        import time

        t0 = time.monotonic()
        for i in range(30):
            cool_temp = HOT_TEMP_THRESHOLD_C - 5.0
            hot_temp = HOT_TEMP_THRESHOLD_C + 5.0
            w_cool.record(TickSnapshot(wall_ts=t0 + i * 5, hashrate_th=100.0, temp_chip_c=cool_temp, power_w=3000.0))
            w_hot.record(TickSnapshot(wall_ts=t0 + i * 5, hashrate_th=100.0, temp_chip_c=hot_temp, power_w=3000.0))

        hsi_cool, _ = compute_hsi(tel, window=w_cool)
        hsi_hot, _ = compute_hsi(tel, window=w_hot)
        assert hsi_cool > hsi_hot

    def test_hsi_decreases_with_higher_hashrate_variability(self) -> None:
        """More hashrate variation → lower HSI (hashrate_variability driven)."""
        tel = _make_telemetry()
        w_stable = MinerWindow(miner_id="m004")
        w_noisy = MinerWindow(miner_id="m005")
        import time

        t0 = time.monotonic()
        import random

        rng = random.Random(42)
        for i in range(30):
            stable_hr = 100.0  # rock-solid
            noisy_hr = 100.0 + rng.uniform(-20, 20)  # ±20%
            w_stable.record(TickSnapshot(wall_ts=t0 + i * 5, hashrate_th=stable_hr, temp_chip_c=70.0, power_w=3000.0))
            w_noisy.record(TickSnapshot(wall_ts=t0 + i * 5, hashrate_th=noisy_hr, temp_chip_c=70.0, power_w=3000.0))

        hsi_stable, _ = compute_hsi(tel, window=w_stable)
        hsi_noisy, _ = compute_hsi(tel, window=w_noisy)
        assert hsi_stable > hsi_noisy

    def test_hsi_in_range(self) -> None:
        """HSI must always be in [0, 100]."""
        import time

        w = MinerWindow(miner_id="m006")
        t0 = time.monotonic()
        # Simulate an extreme-stress miner
        for i in range(100):
            w.record(TickSnapshot(wall_ts=t0 + i, hashrate_th=50.0 + (i % 30), temp_chip_c=90.0, power_w=3600.0))
        tel = _make_telemetry(temp_chip_c=95.0)
        hsi, _ = compute_hsi(tel, window=w)
        assert 0.0 <= hsi <= 100.0

    def test_hsi_default_when_no_window(self) -> None:
        """No window data → HSI = 50 (neutral warmup placeholder).

        v1 used 100 (fully healthy) which was misleading: a miner with
        zero telemetry cannot be "healthy". v2 returns 50 during warmup
        so callers don't confuse cold-start with a confirmed healthy state.
        """
        tel = _make_telemetry()
        hsi, comps = compute_hsi(tel, window=None)
        assert hsi == 50.0
        assert comps["thermal_stress"] == 0.0


# ---------------------------------------------------------------------------
# HSI v2 — per-component behaviour
# ---------------------------------------------------------------------------


def _warm_window(n: int = 40, hashrate_th: float = 100.0,
                 temp_chip_c: float = 70.0) -> MinerWindow:
    """Build a MinerWindow with `n` warm ticks at constant hashrate + temp."""
    import time
    w = MinerWindow(miner_id="m001")
    t0 = time.monotonic()
    for i in range(n):
        w.record(TickSnapshot(
            wall_ts=t0 + i * 5, hashrate_th=hashrate_th,
            temp_chip_c=temp_chip_c, power_w=3250.0,
        ))
    return w


class TestHsiV2Components:
    def test_hsi_responds_to_voltage_deviation(self) -> None:
        """Off-nominal voltage (10.8V) lowers HSI vs nominal (12V)."""
        window = _warm_window()
        tel_nominal = _make_telemetry()
        tel_nominal["voltage_v"] = 12.0
        tel_off = _make_telemetry()
        tel_off["voltage_v"] = 10.8
        hsi_nom, _ = compute_hsi(tel_nominal, window=window)
        hsi_off, _ = compute_hsi(tel_off, window=window)
        assert hsi_off < hsi_nom

    def test_hsi_responds_to_operating_mode(self) -> None:
        """turbo lowers HSI vs eco."""
        window = _warm_window()
        tel_eco = _make_telemetry()
        tel_eco["operating_mode"] = "eco"
        tel_turbo = _make_telemetry()
        tel_turbo["operating_mode"] = "turbo"
        hsi_eco, _ = compute_hsi(tel_eco, window=window)
        hsi_turbo, _ = compute_hsi(tel_turbo, window=window)
        assert hsi_turbo < hsi_eco

    def test_hsi_responds_to_site_temperature(self) -> None:
        """Hotter site temperature → lower HSI (thermal-stress term)."""
        window = _warm_window()
        tel_cool = _make_telemetry()
        tel_cool["env"]["site_temp_c"] = 20.0
        tel_hot = _make_telemetry()
        tel_hot["env"]["site_temp_c"] = 40.0
        hsi_cool, _ = compute_hsi(tel_cool, window=window)
        hsi_hot, _ = compute_hsi(tel_hot, window=window)
        assert hsi_hot < hsi_cool

    def test_hsi_components_are_all_present(self) -> None:
        """The components dict must expose all four stress terms v2 emits."""
        window = _warm_window()
        tel = _make_telemetry()
        _, comps = compute_hsi(tel, window=window)
        for key in ("thermal_stress", "voltage_stress",
                    "mode_stress", "instability_stress", "hot_time_frac"):
            assert key in comps, f"missing HSI v2 component: {key}"

    def test_hsi_in_range_for_varied_inputs(self) -> None:
        """HSI must stay in [0, 100] across a range of inputs."""
        window = _warm_window()
        for voltage in (10.0, 11.5, 12.0, 12.5, 13.5):
            for mode in ("eco", "balanced", "turbo"):
                for site_temp in (15.0, 25.0, 35.0, 50.0):
                    tel = _make_telemetry()
                    tel["voltage_v"] = voltage
                    tel["operating_mode"] = mode
                    tel["env"]["site_temp_c"] = site_temp
                    hsi, _ = compute_hsi(tel, window=window)
                    assert 0.0 <= hsi <= 100.0, (
                        f"HSI out of range: V={voltage} mode={mode} "
                        f"site_temp={site_temp} hsi={hsi}"
                    )


# ---------------------------------------------------------------------------
# Status transitions
# ---------------------------------------------------------------------------


class TestMinerStatus:
    def test_ok_when_healthy(self) -> None:
        assert compute_miner_status(te=70.0, hsi=80.0) == "ok"

    def test_warn_when_te_low(self) -> None:
        assert compute_miner_status(te=WARN_TE - 0.1, hsi=90.0) == "warn"

    def test_warn_when_hsi_low(self) -> None:
        assert compute_miner_status(te=70.0, hsi=WARN_HSI - 0.1) == "warn"

    def test_imm_when_te_very_low(self) -> None:
        assert compute_miner_status(te=IMM_TE - 0.1, hsi=90.0) == "imm"

    def test_imm_when_hsi_very_low(self) -> None:
        assert compute_miner_status(te=70.0, hsi=IMM_HSI - 0.1) == "imm"

    def test_imm_supersedes_warn(self) -> None:
        """If both IMM thresholds are breached, result is still 'imm' not 'warn'."""
        assert compute_miner_status(te=IMM_TE - 1, hsi=IMM_HSI - 1) == "imm"

    def test_boundary_warn_te_exact(self) -> None:
        """Exactly at warn threshold should be 'warn'."""
        assert compute_miner_status(te=WARN_TE, hsi=90.0) == "warn"

    def test_boundary_imm_te_exact(self) -> None:
        """Exactly at imm threshold should be 'imm'."""
        assert compute_miner_status(te=IMM_TE, hsi=90.0) == "imm"

    def test_ok_boundary_just_above_warn(self) -> None:
        """Just above warn threshold → ok."""
        te_ok = WARN_TE + 0.01
        hsi_ok = WARN_HSI + 0.01
        assert compute_miner_status(te=te_ok, hsi=hsi_ok) == "ok"


# ---------------------------------------------------------------------------
# Fleet percentile warmup
# ---------------------------------------------------------------------------


class TestFleetWarmup:
    def test_not_warmed_up_with_few_ticks(self) -> None:
        """Fleet with only 2 ticks per miner should not be warmed up."""
        fs = FleetState()
        import time

        t0 = time.monotonic()
        for mid in ["m001", "m002", "m003", "m004", "m005", "m006"]:
            w = fs.get_or_create(mid)
            for i in range(2):  # only 2 ticks each
                w.record(TickSnapshot(wall_ts=t0 + i, hashrate_th=100.0, temp_chip_c=70.0, power_w=3000.0))
        assert not fs.is_warmed_up()

    def test_warmed_up_after_enough_ticks(self) -> None:
        """Fleet with 5 miners × 60 ticks each should be warmed up."""
        from ingest.thresholds import WARMUP_MIN_MINERS, WARMUP_MIN_TICKS_PER_MINER

        fs = FleetState()
        import time

        t0 = time.monotonic()
        for i in range(WARMUP_MIN_MINERS):
            mid = f"m{i+1:03d}"
            w = fs.get_or_create(mid)
            for j in range(WARMUP_MIN_TICKS_PER_MINER):
                w.record(TickSnapshot(wall_ts=t0 + j, hashrate_th=100.0, temp_chip_c=70.0, power_w=3000.0))
        assert fs.is_warmed_up()

    def test_fleet_r_percentiles_nan_when_insufficient(self) -> None:
        """With fewer than WARMUP_MIN_MINERS, r percentiles are NaN."""
        fs = FleetState()
        fs.update_r("m001", 0.5)
        p5, p95 = fs.fleet_r_percentiles()
        assert math.isnan(p5)
        assert math.isnan(p95)

    def test_fleet_r_percentiles_valid_with_enough_miners(self) -> None:
        """With 10 miners, p5 < p95 and both finite."""
        fs = FleetState()
        import random

        rng = random.Random(0)
        for i in range(10):
            fs.update_r(f"m{i+1:03d}", 0.3 + rng.random() * 0.7)
        p5, p95 = fs.fleet_r_percentiles()
        assert math.isfinite(p5)
        assert math.isfinite(p95)
        assert p5 < p95
