"""MDK Fleet Dashboard — Flask application factory.

Public API (matches shared/specs/interfaces.md §5):

    create_app(stream_dir, design_tokens_path, host, port) -> Flask
    run_dashboard(host, port) -> None

Routes:
    /                       main dashboard
    /miner/<id>             per-miner detail view
    /decisions              searchable reasoning trace log
    /api/stream/<channel>   SSE streams (telemetry|kpi|flag|decision|action)
"""

from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path
from typing import Any

import structlog
from flask import Flask, Response, abort, render_template, request, stream_with_context

from dashboard.sse import replay_jsonl_sse, tail_jsonl_sse

log = structlog.get_logger(__name__)

# Repo root — two levels up from this file
_REPO_ROOT = Path(__file__).resolve().parent.parent
_TOKENS_DEFAULT = _REPO_ROOT / "shared" / "design" / "tokens.json"

# Channel → JSONL filename mapping (relative to stream_dir)
_CHANNEL_FILES: dict[str, str] = {
    "telemetry": "telemetry.jsonl",
    "kpi": "kpis.jsonl",
    "flag": "flags.jsonl",
    "decision": "decisions.jsonl",
    "action": "actions.jsonl",
    "snapshot": "snapshots.jsonl",
    "live": "live.jsonl",
}

# How many recent events to load for initial page renders
_PAGE_INIT_LIMIT = 200


def _load_tokens(path: Path) -> dict[str, Any]:
    """Load tokens.json and return the dict. Returns empty dict on error."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        log.warning("dashboard.tokens_load_failed", path=str(path), error=str(exc))
        return {}


def _read_jsonl_tail(path: Path, limit: int = _PAGE_INIT_LIMIT) -> list[dict[str, Any]]:
    """Return the last `limit` parsed JSON objects from a JSONL file.

    Returns [] if the file does not exist or has no valid lines.
    """
    if not path.exists():
        return []

    lines: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            buf: deque[str] = deque(maxlen=limit)
            for raw in f:
                raw = raw.rstrip("\n")
                if raw:
                    buf.append(raw)
    except OSError:
        return []

    for raw in buf:
        try:
            lines.append(json.loads(raw))
        except json.JSONDecodeError:
            pass

    return lines


def _build_fleet_state(snapshot_events: list[dict[str, Any]]) -> dict[str, Any]:
    """Collapse recent fleet_snapshot events into a current fleet state dict."""
    state: dict[str, Any] = {
        "miners": {},
        "fleet_te": 0.0,
        "fleet_hsi": 0.0,
        "env": {},
    }
    for ev in snapshot_events:
        if ev.get("event") != "fleet_snapshot":
            continue
        data = ev.get("data", {})
        state["miners"].update(data.get("miners", {}))
        state["fleet_te"] = data.get("fleet_te", 0.0)
        state["fleet_hsi"] = data.get("fleet_hsi", 0.0)
        if data.get("env"):
            state["env"] = data["env"]
    return state


def _build_miner_state(
    miner_id: str,
    telemetry_events: list[dict[str, Any]],
    kpi_events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build per-miner detail state from recent telemetry + kpi events.

    Returns a dict with the most recent telemetry + KPI values and
    historical series for sparklines.
    """
    tel_series: list[dict[str, Any]] = []
    kpi_series: list[dict[str, Any]] = []

    for ev in telemetry_events:
        if ev.get("event") == "telemetry_tick":
            d = ev.get("data", {})
            if d.get("miner_id") == miner_id:
                tel_series.append({"ts": ev.get("ts"), **d})

    for ev in kpi_events:
        if ev.get("event") == "kpi_update":
            d = ev.get("data", {})
            if d.get("miner_id") == miner_id:
                kpi_series.append({"ts": ev.get("ts"), **d})

    latest_tel = tel_series[-1] if tel_series else {}
    latest_kpi = kpi_series[-1] if kpi_series else {}

    return {
        "miner_id": miner_id,
        "latest_telemetry": latest_tel,
        "latest_kpi": latest_kpi,
        "tel_series": tel_series[-60:],   # last 60 ticks for sparkline
        "kpi_series": kpi_series[-60:],
    }


def create_app(
    stream_dir: str = "/run/mdk_fleet/stream/",
    design_tokens_path: str = str(_TOKENS_DEFAULT),
    host: str = "127.0.0.1",
    port: int = 8000,
) -> Flask:
    """Build and return the Flask app.

    Args:
        stream_dir: directory containing JSONL stream files.
        design_tokens_path: path to shared/design/tokens.json.
        host: bind host (stored for run_dashboard convenience).
        port: bind port (stored for run_dashboard convenience).

    Returns:
        Configured Flask application instance.
    """
    # Resolve paths
    _stream_dir = Path(stream_dir).expanduser()
    _tokens_path = Path(design_tokens_path).expanduser()

    # Previous behaviour silently fell back to `_REPO_ROOT / "events"` when
    # the requested stream dir didn't exist. Under pipx that fallback landed
    # in site-packages and leaked stale events across sessions (see
    # dashboard-stream-isolation audit). Now: if the requested dir is
    # missing we CREATE it rather than pointing at a shared path. An
    # entirely fresh dir means the dashboard shows zero historical events
    # until the first real write arrives.
    _stream_dir.mkdir(parents=True, exist_ok=True)

    tokens = _load_tokens(_tokens_path)

    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )

    # Stash runtime config on app for access in routes
    app.config["MDK_STREAM_DIR"] = _stream_dir
    app.config["MDK_TOKENS"] = tokens
    app.config["MDK_HOST"] = host
    app.config["MDK_PORT"] = port

    # ------------------------------------------------------------------ #
    # Template context helpers                                             #
    # ------------------------------------------------------------------ #

    @app.context_processor
    def _inject_tokens() -> dict[str, Any]:
        """Inject token palette into every template context."""
        t = tokens.get("color", {})
        return {"tokens": tokens, "color": t}

    @app.context_processor
    def _inject_backend_badge() -> dict[str, Any]:
        """Read `backend_summary.json` from the stream dir to display the
        active LLM backend in the header of every page.

        Fallback: if the file is absent (e.g. dashboard launched
        standalone against an older run), show a neutral placeholder.
        """
        sd: Path = app.config["MDK_STREAM_DIR"]
        summary_path = sd / "backend_summary.json"
        if not summary_path.exists():
            return {"backend_badge": None}
        try:
            import json as _json
            data = _json.loads(summary_path.read_text())
            return {"backend_badge": data}
        except Exception:
            return {"backend_badge": None}

    # ------------------------------------------------------------------ #
    # Route: /  (main dashboard)                                          #
    # ------------------------------------------------------------------ #

    @app.route("/")
    def index() -> str:
        sd: Path = app.config["MDK_STREAM_DIR"]

        snapshot_events = _read_jsonl_tail(sd / "snapshots.jsonl")
        flag_events = _read_jsonl_tail(sd / "flags.jsonl")
        decision_events = _read_jsonl_tail(sd / "decisions.jsonl")

        fleet = _build_fleet_state(snapshot_events)

        # Derive per-miner status list for fleet map (50 miners)
        all_miners = [f"m{i:03d}" for i in range(1, 51)]
        miner_cells: list[dict[str, Any]] = []
        for mid in all_miners:
            info = fleet["miners"].get(mid, {})
            miner_cells.append(
                {
                    "id": mid,
                    "status": info.get("status", "shut"),
                    "te": info.get("te", 0.0),
                    "hsi": info.get("hsi", 0.0),
                    "hashrate_th": info.get("hashrate_th", 0.0),
                    "temp_chip_c": info.get("temp_chip_c", 0.0),
                }
            )

        # Recent flags (up to 30)
        recent_flags = [
            ev
            for ev in flag_events
            if ev.get("event") == "flag_raised"
        ][-30:][::-1]

        # Recent decisions (trace log, up to 20)
        recent_decisions = [
            ev
            for ev in decision_events
            if ev.get("event") == "orchestrator_decision"
        ][-20:][::-1]

        return render_template(
            "index.html",
            miner_cells=miner_cells,
            fleet=fleet,
            recent_flags=recent_flags,
            recent_decisions=recent_decisions,
        )

    # ------------------------------------------------------------------ #
    # Route: /miner/<id>                                                  #
    # ------------------------------------------------------------------ #

    @app.route("/miner/<string:miner_id>")
    def miner_detail(miner_id: str) -> str:
        # Validate format
        import re

        if not re.match(r"^m\d{3}$", miner_id):
            abort(404)

        sd: Path = app.config["MDK_STREAM_DIR"]

        tel_events = _read_jsonl_tail(sd / "telemetry.jsonl")
        kpi_events = _read_jsonl_tail(sd / "kpis.jsonl")
        flag_events = _read_jsonl_tail(sd / "flags.jsonl")
        decision_events = _read_jsonl_tail(sd / "decisions.jsonl")

        miner = _build_miner_state(miner_id, tel_events, kpi_events)

        # Flags for this miner
        miner_flags = [
            ev
            for ev in flag_events
            if ev.get("event") == "flag_raised"
            and ev.get("data", {}).get("miner_id") == miner_id
        ][-30:][::-1]

        # Decisions for this miner
        miner_decisions = [
            ev
            for ev in decision_events
            if ev.get("event") == "orchestrator_decision"
            and ev.get("data", {}).get("miner_id") == miner_id
        ][-20:][::-1]

        return render_template(
            "miner_detail.html",
            miner_id=miner_id,
            miner=miner,
            miner_flags=miner_flags,
            miner_decisions=miner_decisions,
        )

    # ------------------------------------------------------------------ #
    # Route: /decisions                                                    #
    # ------------------------------------------------------------------ #

    @app.route("/decisions")
    def decisions() -> str:
        sd: Path = app.config["MDK_STREAM_DIR"]
        decision_events = _read_jsonl_tail(sd / "decisions.jsonl", limit=500)

        # Parse filter params
        q_miner = request.args.get("miner", "").strip().lower()
        q_flag_type = request.args.get("flag_type", "").strip()
        q_autonomy = request.args.get("autonomy", "").strip()
        q_action = request.args.get("action", "").strip()

        rows = [ev for ev in decision_events if ev.get("event") == "orchestrator_decision"]

        # Apply filters
        if q_miner:
            rows = [r for r in rows if q_miner in r.get("data", {}).get("miner_id", "")]
        if q_action:
            rows = [r for r in rows if r.get("data", {}).get("action") == q_action]
        if q_autonomy:
            rows = [r for r in rows if r.get("data", {}).get("autonomy_level") == q_autonomy]

        rows = rows[::-1]  # newest first

        # Distinct action values for the filter dropdown
        all_actions = sorted({
            r.get("data", {}).get("action", "")
            for r in rows
            if r.get("data", {}).get("action")
        })
        all_autonomy = sorted({
            r.get("data", {}).get("autonomy_level", "")
            for r in rows
            if r.get("data", {}).get("autonomy_level")
        })

        return render_template(
            "decisions.html",
            decisions=rows,
            q_miner=q_miner,
            q_flag_type=q_flag_type,
            q_autonomy=q_autonomy,
            q_action=q_action,
            all_actions=all_actions,
            all_autonomy=all_autonomy,
            total=len(rows),
        )

    # ------------------------------------------------------------------ #
    # Route: /api/stream/<channel>  (SSE)                                 #
    # ------------------------------------------------------------------ #

    @app.route("/api/stream/<string:channel>")
    def stream(channel: str) -> Response:
        if channel not in _CHANNEL_FILES:
            abort(404)

        sd: Path = app.config["MDK_STREAM_DIR"]
        path = sd / _CHANNEL_FILES[channel]

        from_start_param = request.args.get("from_start", "false").lower()
        from_start = from_start_param == "true"

        replay_limit = int(request.args.get("replay", "50"))

        def _generator():
            # First, replay recent history so the client gets context
            yield from replay_jsonl_sse(path, limit=replay_limit)
            # Then tail-follow for new events
            yield from tail_jsonl_sse(path, from_start=False)

        return Response(
            stream_with_context(_generator()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    # ------------------------------------------------------------------ #
    # Route: /api/fleet  (JSON snapshot for polling clients)              #
    # ------------------------------------------------------------------ #

    @app.route("/api/fleet")
    def api_fleet() -> Response:
        sd: Path = app.config["MDK_STREAM_DIR"]
        snapshot_events = _read_jsonl_tail(sd / "snapshots.jsonl", limit=5)
        fleet = _build_fleet_state(snapshot_events)
        return Response(
            json.dumps(fleet, default=str),
            mimetype="application/json",
        )

    # ------------------------------------------------------------------ #
    # Route: /api/flags  (recent flags JSON)                              #
    # ------------------------------------------------------------------ #

    @app.route("/api/flags")
    def api_flags() -> Response:
        sd: Path = app.config["MDK_STREAM_DIR"]
        limit = int(request.args.get("limit", "30"))
        flag_events = _read_jsonl_tail(sd / "flags.jsonl", limit=limit)
        flags = [ev for ev in flag_events if ev.get("event") == "flag_raised"][-limit:][::-1]
        return Response(json.dumps(flags, default=str), mimetype="application/json")

    log.info(
        "dashboard.app_created",
        stream_dir=str(_stream_dir),
        tokens_loaded=bool(tokens),
    )
    return app


def run_dashboard(
    host: str = "127.0.0.1",
    port: int = 8000,
    stream_dir: str | None = None,
) -> None:
    """Start the dashboard server.

    Args:
        host: bind host.
        port: bind port.
        stream_dir: override for the JSONL stream dir. When None, reads
            `MDK_STREAM_DIR` from the environment; if that is unset, falls
            back to the spec default `/run/mdk_fleet/stream/` (which itself
            falls back to `events/` when not writable).
    """
    import os

    resolved_stream = stream_dir or os.environ.get("MDK_STREAM_DIR") or "/run/mdk_fleet/stream/"
    app = create_app(stream_dir=resolved_stream, host=host, port=port)
    log.info("dashboard.starting", host=host, port=port, stream_dir=resolved_stream)
    app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)
