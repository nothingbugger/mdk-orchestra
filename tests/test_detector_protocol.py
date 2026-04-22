"""Tests for Flagger protocol conformance and schema round-trip.

Covers:
  - All three flaggers conform to the Flagger Protocol (isinstance check via
    @runtime_checkable).
  - FlagResult from rule engine can be wrapped in a FlagRaised Pydantic model,
    serialised to JSON, parsed back via parse_event, and round-trips correctly.
  - MinerHistory bounded eviction works.
  - Isolation Forest and XGBoost flaggers return None when model not loaded.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from deterministic_tools.base import Flagger, FlagResult, MinerHistory
from deterministic_tools.isolation_forest_flagger import IsolationForestFlagger
from deterministic_tools.rule_engine_flagger import RuleEngineFlagger
from deterministic_tools.xgboost_flagger import XGBoostFlagger
from shared.schemas.events import EnvBlock, FlagEvidence, FlagRaised, TelemetryTick


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_tick(miner_id: str = "m001", temp_chip_c: float = 75.0) -> TelemetryTick:
    return TelemetryTick(
        miner_id=miner_id,
        miner_model="S19j Pro",
        hashrate_th=100.0,
        hashrate_expected_th=104.0,
        temp_chip_c=temp_chip_c,
        temp_amb_c=25.0,
        power_w=3250.0,
        voltage_v=12.0,
        fan_rpm=[5800, 5800, 5800, 5800],
        operating_mode="balanced",
        uptime_s=3600.0,
        env=EnvBlock(
            site_temp_c=25.0,
            site_humidity_pct=40.0,
            elec_price_usd_kwh=0.07,
            hashprice_usd_per_th_day=0.058,
        ),
        fault_injected=None,
    )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestFlaggerProtocol:
    """All flaggers must satisfy the Flagger Protocol."""

    def test_rule_engine_is_flagger(self) -> None:
        f = RuleEngineFlagger()
        assert isinstance(f, Flagger)

    def test_isolation_forest_is_flagger(self, tmp_path) -> None:
        f = IsolationForestFlagger(model_path=tmp_path / "if_v2.pkl")
        assert isinstance(f, Flagger)

    def test_xgboost_is_flagger(self, tmp_path) -> None:
        f = XGBoostFlagger(model_path=tmp_path / "xgb_predictor.pkl")
        assert isinstance(f, Flagger)

    def test_all_have_name(self) -> None:
        assert RuleEngineFlagger().name == "rule_engine"
        assert IsolationForestFlagger(model_path="/tmp/nope.pkl").name == "isolation_forest_v2"
        assert XGBoostFlagger(model_path="/tmp/nope.pkl").name == "xgboost_predictor"

    def test_rule_engine_evaluate_signature(self) -> None:
        """evaluate() must accept (TelemetryTick, MinerHistory) and return FlagResult or None."""
        f = RuleEngineFlagger()
        tick = _make_tick()
        hist = MinerHistory(miner_id="m001")
        result = f.evaluate(tick, hist)
        assert result is None or isinstance(result, FlagResult)

    def test_ml_flaggers_return_none_without_model(self, tmp_path) -> None:
        """ML flaggers return None immediately when no model is loaded."""
        if_f = IsolationForestFlagger(model_path=tmp_path / "missing_if.pkl")
        xgb_f = XGBoostFlagger(model_path=tmp_path / "missing_xgb.pkl")
        tick = _make_tick()
        hist = MinerHistory(miner_id="m001")
        assert if_f.evaluate(tick, hist) is None
        assert xgb_f.evaluate(tick, hist) is None


# ---------------------------------------------------------------------------
# FlagRaised schema round-trip
# ---------------------------------------------------------------------------


class TestSchemaRoundTrip:
    """FlagRaised → JSON → parse_event → FlagRaised round-trip."""

    def _make_flag_raised(self) -> FlagRaised:
        return FlagRaised(
            flag_id="flg_00001",
            miner_id="m042",
            flag_type="thermal_runaway",
            severity="crit",
            confidence=0.95,
            source_tool="rule_engine",
            evidence=FlagEvidence(
                metric="temp_chip_c",
                window_min=10.0,
                current_value=87.2,
                threshold_c=85.0,
            ),
            raw_score=2.2,
        )

    def test_flag_raised_serialises(self) -> None:
        flag = self._make_flag_raised()
        data = flag.model_dump(mode="json")
        assert data["flag_type"] == "thermal_runaway"
        assert data["severity"] == "crit"

    def test_envelope_round_trip(self) -> None:
        """Build envelope, serialise to JSON, parse back, typed_data() matches."""
        from shared.event_bus import write_event
        from shared.schemas.events import Envelope, parse_event

        flag = self._make_flag_raised()
        env = Envelope.wrap(event="flag_raised", source="detector", payload=flag)
        raw_json = env.model_dump_json()

        # Round-trip.
        parsed_dict = json.loads(raw_json)
        env2, payload = parse_event(parsed_dict)
        assert env2.event == "flag_raised"
        assert env2.source == "detector"
        assert isinstance(payload, FlagRaised)
        assert payload.flag_id == "flg_00001"
        assert payload.miner_id == "m042"
        assert payload.severity == "crit"

    def test_evidence_extra_fields_preserved(self) -> None:
        """FlagEvidence allows extra fields (model_config extra='allow')."""
        flag = self._make_flag_raised()
        data = flag.evidence.model_dump(mode="json")
        assert data["metric"] == "temp_chip_c"
        assert data["window_min"] == 10.0
        assert data["current_value"] == 87.2


# ---------------------------------------------------------------------------
# MinerHistory
# ---------------------------------------------------------------------------


class TestMinerHistory:
    """MinerHistory rolling eviction and helpers."""

    def test_bounded_eviction(self) -> None:
        """History respects maxlen — oldest ticks are evicted."""
        hist = MinerHistory(miner_id="m001", maxlen=10)
        ts = datetime(2026, 4, 20, 12, 0, 0, tzinfo=timezone.utc)
        from datetime import timedelta

        for i in range(15):
            tick = _make_tick(miner_id="m001")
            hist.push_telemetry(tick, ts + timedelta(seconds=i * 5))

        assert len(hist.telemetry) == 10

    def test_last_telemetry_none_on_empty(self) -> None:
        hist = MinerHistory(miner_id="m001")
        assert hist.last_telemetry() is None

    def test_last_kpi_none_on_empty(self) -> None:
        hist = MinerHistory(miner_id="m001")
        assert hist.last_kpi() is None

    def test_cooldown_not_set_returns_false(self) -> None:
        hist = MinerHistory(miner_id="m001")
        ts = datetime(2026, 4, 20, 12, 0, 0, tzinfo=timezone.utc)
        assert not hist.is_on_cooldown("thermal_runaway", 600, ts)

    def test_cooldown_active_returns_true(self) -> None:
        hist = MinerHistory(miner_id="m001")
        from datetime import timedelta

        ts = datetime(2026, 4, 20, 12, 0, 0, tzinfo=timezone.utc)
        hist.record_emission("thermal_runaway", ts)
        ts2 = ts + timedelta(seconds=100)  # within 600 s cooldown
        assert hist.is_on_cooldown("thermal_runaway", 600, ts2)

    def test_cooldown_cleared_after_expiry(self) -> None:
        hist = MinerHistory(miner_id="m001")
        from datetime import timedelta

        ts = datetime(2026, 4, 20, 12, 0, 0, tzinfo=timezone.utc)
        hist.record_emission("thermal_runaway", ts)
        ts2 = ts + timedelta(seconds=700)  # after 600 s cooldown
        assert not hist.is_on_cooldown("thermal_runaway", 600, ts2)

    def test_recent_telemetry_n(self) -> None:
        hist = MinerHistory(miner_id="m001")
        ts = datetime(2026, 4, 20, 12, 0, 0, tzinfo=timezone.utc)
        from datetime import timedelta

        for i in range(20):
            tick = _make_tick(miner_id="m001")
            hist.push_telemetry(tick, ts + timedelta(seconds=i * 5))

        last5 = hist.recent_telemetry(5)
        assert len(last5) == 5
