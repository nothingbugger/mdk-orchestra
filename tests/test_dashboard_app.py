"""Tests for the MDK Fleet dashboard module.

Covers:
- create_app() returns a Flask app
- / renders 200 on fixture data
- /miner/<id> renders 200 / 404
- /decisions renders 200
- SSE generator yields proper format
- build_css produces CSS with no inline hex outside :root
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

import pytest

# Make sure the project root is on the path for module imports
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from flask import Flask
from flask.testing import FlaskClient

from dashboard.app import create_app


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _write_fixture_streams(stream_dir: Path) -> None:
    """Write minimal fixture JSONL files for the dashboard to render."""

    # 5 telemetry_tick events for m001
    telemetry = stream_dir / "telemetry.jsonl"
    for i in range(5):
        ev = {
            "event": "telemetry_tick",
            "ts": f"2026-04-20T15:30:{i:02d}.000+00:00",
            "source": "simulator",
            "data": {
                "miner_id": "m001",
                "miner_model": "S19j Pro",
                "hashrate_th": 95.0 + i,
                "hashrate_expected_th": 104.0,
                "temp_chip_c": 75.0 + i * 0.5,
                "temp_amb_c": 24.0,
                "power_w": 3250,
                "voltage_v": 12.08,
                "fan_rpm": [5800, 5820, 5790, 5810],
                "operating_mode": "balanced",
                "uptime_s": 10000 + i,
                "env": {
                    "site_temp_c": 23.5,
                    "site_humidity_pct": 42,
                    "elec_price_usd_kwh": 0.072,
                    "hashprice_usd_per_th_day": 0.058,
                },
                "fault_injected": None,
            },
        }
        with telemetry.open("a") as f:
            f.write(json.dumps(ev) + "\n")

    # 2 flag_raised events
    flags = stream_dir / "flags.jsonl"
    for i, sev in enumerate(["warn", "crit"]):
        ev = {
            "event": "flag_raised",
            "ts": f"2026-04-20T15:31:{i:02d}.000+00:00",
            "source": "detector",
            "data": {
                "flag_id": f"flg_{i:05d}",
                "miner_id": "m001",
                "flag_type": "voltage_drift",
                "severity": sev,
                "confidence": 0.71 + i * 0.1,
                "source_tool": "rule_engine",
                "evidence": {"metric": "voltage_v", "window_min": 30},
                "raw_score": 0.82,
            },
        }
        with flags.open("a") as f:
            f.write(json.dumps(ev) + "\n")

    # 1 orchestrator_decision
    decisions = stream_dir / "decisions.jsonl"
    ev = {
        "event": "orchestrator_decision",
        "ts": "2026-04-20T15:32:00.000+00:00",
        "source": "orchestrator",
        "data": {
            "decision_id": "dec_00001",
            "flag_id": "flg_00000",
            "miner_id": "m001",
            "action": "throttle",
            "action_params": {"target_hashrate_pct": 0.80},
            "autonomy_level": "L3_bounded_auto",
            "confidence": 0.78,
            "reasoning_trace": "Voltage drift confirmed by voltage_agent. Throttle to 80%.",
            "consulted_agents": ["voltage_agent", "power_agent"],
            "total_cost_usd": 0.018,
            "total_latency_ms": 4820,
            "pending_human_approval": False,
        },
    }
    with decisions.open("a") as f:
        f.write(json.dumps(ev) + "\n")

    # 1 fleet_snapshot
    snapshots = stream_dir / "snapshots.jsonl"
    miners_data = {f"m{i:03d}": {
        "te": 45.0 + i * 0.3,
        "hsi": 80.0 - i * 0.2,
        "status": "ok",
        "hashrate_th": 95.0,
        "temp_chip_c": 76.0,
    } for i in range(1, 51)}
    # Force one miner to warn
    miners_data["m003"]["status"] = "warn"
    ev_snap = {
        "event": "fleet_snapshot",
        "ts": "2026-04-20T15:32:05.000+00:00",
        "source": "ingest",
        "data": {
            "miner_count": 50,
            "miners": miners_data,
            "fleet_te": 49.8,
            "fleet_hsi": 81.6,
            "env": {
                "site_temp_c": 23.5,
                "site_humidity_pct": 42,
                "elec_price_usd_kwh": 0.072,
                "hashprice_usd_per_th_day": 0.058,
            },
        },
    }
    with snapshots.open("a") as f:
        f.write(json.dumps(ev_snap) + "\n")

    # kpis
    kpis = stream_dir / "kpis.jsonl"
    ev_kpi = {
        "event": "kpi_update",
        "ts": "2026-04-20T15:31:30.000+00:00",
        "source": "ingest",
        "data": {
            "miner_id": "m001",
            "te": 47.3,
            "hsi": 82.1,
            "te_components": {
                "value_usd_day": 5.53,
                "cost_usd_day": 11.70,
                "h_eff_th": 95.3,
                "p_hashprice": 0.058,
                "p_asic_w": 3250,
                "p_cool_w": 520,
                "rho_elec": 0.072,
                "sigma_hash": 0.21,
                "alpha": 0.5,
            },
            "hsi_components": {
                "thermal_stress": 0.18,
                "hashrate_variability": 0.11,
                "hot_time_frac": 0.05,
            },
        },
    }
    with kpis.open("a") as f:
        f.write(json.dumps(ev_kpi) + "\n")


@pytest.fixture()
def stream_dir(tmp_path: Path) -> Path:
    """Create a temp stream dir with fixture data and return it."""
    _write_fixture_streams(tmp_path)
    return tmp_path


@pytest.fixture()
def app(stream_dir: Path) -> Flask:
    """Return a configured Flask app pointed at fixture stream dir."""
    flask_app = create_app(stream_dir=str(stream_dir))
    flask_app.config["TESTING"] = True
    return flask_app


@pytest.fixture()
def client(app: Flask) -> FlaskClient:
    return app.test_client()


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestCreateApp:
    def test_returns_flask_instance(self, app: Flask) -> None:
        assert isinstance(app, Flask)

    def test_stream_dir_stored(self, app: Flask, stream_dir: Path) -> None:
        assert app.config["MDK_STREAM_DIR"] == stream_dir

    def test_tokens_loaded(self, app: Flask) -> None:
        """Tokens dict should be non-empty (tokens.json was found)."""
        assert isinstance(app.config["MDK_TOKENS"], dict)
        assert app.config["MDK_TOKENS"]  # non-empty


class TestMainRoute:
    def test_index_returns_200(self, client: FlaskClient) -> None:
        resp = client.get("/")
        assert resp.status_code == 200

    def test_index_contains_fleet_grid(self, client: FlaskClient) -> None:
        resp = client.get("/")
        html = resp.get_data(as_text=True)
        assert "fleet-grid" in html

    def test_index_contains_miner_cells(self, client: FlaskClient) -> None:
        resp = client.get("/")
        html = resp.get_data(as_text=True)
        assert "miner-cell" in html
        assert "m001" in html

    def test_index_contains_flags_feed(self, client: FlaskClient) -> None:
        resp = client.get("/")
        html = resp.get_data(as_text=True)
        assert "flags-feed" in html

    def test_index_contains_trace_feed(self, client: FlaskClient) -> None:
        resp = client.get("/")
        html = resp.get_data(as_text=True)
        assert "trace-feed" in html

    def test_index_shows_fixture_flag(self, client: FlaskClient) -> None:
        resp = client.get("/")
        html = resp.get_data(as_text=True)
        assert "voltage_drift" in html

    def test_index_shows_fixture_decision(self, client: FlaskClient) -> None:
        resp = client.get("/")
        html = resp.get_data(as_text=True)
        assert "throttle" in html


class TestMinerDetailRoute:
    def test_miner_detail_returns_200(self, client: FlaskClient) -> None:
        resp = client.get("/miner/m001")
        assert resp.status_code == 200

    def test_miner_detail_shows_miner_id(self, client: FlaskClient) -> None:
        resp = client.get("/miner/m001")
        html = resp.get_data(as_text=True)
        assert "m001" in html

    def test_miner_detail_shows_telemetry_table(self, client: FlaskClient) -> None:
        resp = client.get("/miner/m001")
        html = resp.get_data(as_text=True)
        assert "telemetry-table" in html

    def test_miner_detail_shows_flags(self, client: FlaskClient) -> None:
        resp = client.get("/miner/m001")
        html = resp.get_data(as_text=True)
        assert "voltage_drift" in html

    def test_miner_detail_shows_decision(self, client: FlaskClient) -> None:
        resp = client.get("/miner/m001")
        html = resp.get_data(as_text=True)
        assert "throttle" in html

    def test_miner_detail_invalid_id_returns_404(self, client: FlaskClient) -> None:
        resp = client.get("/miner/invalid")
        assert resp.status_code == 404

    def test_miner_detail_unknown_miner_returns_200(self, client: FlaskClient) -> None:
        """Unknown but valid-format miner id should render empty gracefully."""
        resp = client.get("/miner/m099")
        assert resp.status_code == 200


class TestDecisionsRoute:
    def test_decisions_returns_200(self, client: FlaskClient) -> None:
        resp = client.get("/decisions")
        assert resp.status_code == 200

    def test_decisions_contains_filter_bar(self, client: FlaskClient) -> None:
        resp = client.get("/decisions")
        html = resp.get_data(as_text=True)
        assert "decisions-filter-bar" in html

    def test_decisions_shows_fixture_decision(self, client: FlaskClient) -> None:
        resp = client.get("/decisions")
        html = resp.get_data(as_text=True)
        assert "throttle" in html
        assert "m001" in html

    def test_decisions_filter_by_miner(self, client: FlaskClient) -> None:
        resp = client.get("/decisions?miner=m001")
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)
        assert "m001" in html

    def test_decisions_filter_no_match(self, client: FlaskClient) -> None:
        resp = client.get("/decisions?miner=m099")
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)
        assert "No decisions match" in html


class TestSSERoutes:
    def test_unknown_channel_returns_404(self, client: FlaskClient) -> None:
        resp = client.get("/api/stream/unknown")
        assert resp.status_code == 404

    def test_known_channel_returns_event_stream(self, client: FlaskClient) -> None:
        """SSE endpoint should return text/event-stream content-type."""
        # We can't easily drain SSE in tests — just check headers + status
        resp = client.get("/api/stream/flag", buffered=False)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.content_type

    def test_sse_replay_yields_data_lines(self, client: FlaskClient) -> None:
        """Replay-only SSE with limit=2 should contain data: lines."""
        resp = client.get("/api/stream/flag?replay=2")
        # Read a small chunk
        chunk = resp.response.__next__()
        decoded = chunk.decode("utf-8")
        assert decoded.startswith("data: ")


class TestSSEGenerator:
    """Unit tests for sse.py generator functions."""

    def test_replay_jsonl_sse_empty_file(self, tmp_path: Path) -> None:
        from dashboard.sse import replay_jsonl_sse

        p = tmp_path / "empty.jsonl"
        p.touch()
        events = list(replay_jsonl_sse(p, limit=10))
        assert events == []

    def test_replay_jsonl_sse_yields_data_lines(self, tmp_path: Path) -> None:
        from dashboard.sse import replay_jsonl_sse

        p = tmp_path / "test.jsonl"
        p.write_text(
            json.dumps({"event": "flag_raised", "ts": "2026-01-01T00:00:00Z",
                        "source": "detector", "data": {}}) + "\n"
            + json.dumps({"event": "flag_raised", "ts": "2026-01-01T00:00:01Z",
                          "source": "detector", "data": {}}) + "\n"
        )
        events = list(replay_jsonl_sse(p, limit=10))
        assert len(events) == 2
        for e in events:
            assert e.startswith("data: ")
            assert e.endswith("\n\n")

    def test_replay_jsonl_sse_limit(self, tmp_path: Path) -> None:
        from dashboard.sse import replay_jsonl_sse

        p = tmp_path / "many.jsonl"
        for i in range(20):
            with p.open("a") as f:
                f.write(json.dumps({"event": "flag_raised",
                                    "ts": f"2026-01-01T00:00:{i:02d}Z",
                                    "source": "detector", "data": {}}) + "\n")
        events = list(replay_jsonl_sse(p, limit=5))
        assert len(events) == 5

    def test_replay_skips_malformed_json(self, tmp_path: Path) -> None:
        from dashboard.sse import replay_jsonl_sse

        p = tmp_path / "bad.jsonl"
        p.write_text("not-json\n" + json.dumps({"event": "e", "ts": "t",
                                                 "source": "s", "data": {}}) + "\n")
        events = list(replay_jsonl_sse(p, limit=10))
        assert len(events) == 1


class TestBuildCss:
    def test_build_css_no_error(self, tmp_path: Path) -> None:
        """build_css.py should run without errors when tokens.json is present."""
        from dashboard.build_css import build_css, _OUT_PATH

        build_css()
        assert _OUT_PATH.exists()

    def test_css_has_root_vars(self) -> None:
        """Generated CSS should contain :root { } with CSS variables."""
        from dashboard.build_css import _OUT_PATH

        if not _OUT_PATH.exists():
            pytest.skip("CSS not yet generated")
        css = _OUT_PATH.read_text(encoding="utf-8")
        assert ":root {" in css
        assert "--color-bg-base" in css

    def test_css_semantic_classes_use_vars_not_hex(self) -> None:
        """Semantic class rules outside :root should not contain bare hex codes."""
        from dashboard.build_css import _OUT_PATH

        if not _OUT_PATH.exists():
            pytest.skip("CSS not yet generated")

        css = _OUT_PATH.read_text(encoding="utf-8")

        # Split into :root block vs the rest
        root_end = css.find("}\n", css.find(":root {"))
        after_root = css[root_end + 1:]

        # In the semantic part, any hex color in a property value (not in a comment
        # and not inside a url(...) or a var(...)) is a violation.
        # Strategy: strip /* ... */ comments, then look for bare hex not in var().
        stripped = re.sub(r"/\*.*?\*/", "", after_root, flags=re.DOTALL)
        # Remove content of var(...) references — those are fine
        stripped = re.sub(r"var\([^)]+\)", "var(OK)", stripped)
        hex_matches = re.findall(r"#[0-9a-fA-F]{3,8}\b", stripped)
        # Allow rgba(...) — those come from tokens vars too, but if any bare #hex
        # appears it means we wrote it directly.
        real_violations = [m for m in hex_matches if m]
        assert real_violations == [], (
            f"Found hardcoded hex in semantic CSS (after stripping comments/vars): {real_violations}"
        )


class TestApiRoutes:
    def test_api_fleet_returns_json(self, client: FlaskClient) -> None:
        resp = client.get("/api/fleet")
        assert resp.status_code == 200
        assert resp.content_type == "application/json"
        data = json.loads(resp.data)
        assert "miners" in data

    def test_api_flags_returns_json(self, client: FlaskClient) -> None:
        resp = client.get("/api/flags?limit=5")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert isinstance(data, list)
