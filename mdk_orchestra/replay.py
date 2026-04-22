"""Replay a recorded event-stream run at a configurable speed.

Reads JSONL files from a source directory (default: `examples/demo_replay/`),
merges all events into a single chronological timeline, then streams each
event into a fresh `runs/<id>/` directory spacing them by
`original_delta_s / speed` seconds of wall-clock time.

While the replay is running, the dashboard (pointed at `runs/<id>/` via
`MDK_STREAM_DIR`) sees events appear live, exactly as if they were being
produced by a real simulator + Orchestra pipeline.

When all events have been written, `replay_meta.json` is emitted into the
run dir — the CLI uses this marker to detect replay completion.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PYTHON = sys.executable

# Streams to replay — name → filename in the source dir.
_REPLAY_STREAMS: dict[str, str] = {
    "flags": "flags.jsonl",
    "decisions": "decisions.jsonl",
    "actions": "actions.jsonl",
    "snapshots": "snapshots.jsonl",
}


def _free_dashboard_port(port: int) -> None:
    """Kill any process bound to `port`. Mirrors the helper in cli.py; kept
    local to avoid a cross-module import cycle."""
    import signal as _signal
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True, timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return
    for raw in result.stdout.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            pid = int(raw)
        except ValueError:
            continue
        try:
            os.kill(pid, _signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    if result.stdout.strip():
        time.sleep(0.5)


def _parse_ts(raw: str) -> float:
    """Parse ISO-8601 timestamp (with optional trailing Z) to epoch seconds."""
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return _dt.datetime.fromisoformat(raw).timestamp()
    except ValueError:
        return 0.0


_MIN_FLAG_GAP_S = 8.0
"""Minimum simulated-time gap between two consecutive flag events.

The canonical replay dataset has ~10 flag events firing sub-second at run
start (detectors all light up on boot-up telemetry). That makes the demo
feel like "a wall of flags with no reasoning" — by the time the first
Orchestra decision lands 15 s later, the user has already seen 7 flags
and context is lost.

Enforcing an 8-second minimum gap spreads them so at 1× speed each flag
arrives at roughly the cadence of the decision arrivals behind it, and
the user sees the flag → reasoning → decision cadence the real system
is meant to show. Decisions, actions, and snapshots are unchanged —
only flag timestamps get pushed forward when they're too close together.
"""


def _load_timeline(source: Path) -> list[tuple[float, str, dict]]:
    """Return a list of `(ts_epoch, stream_name, event_dict)` sorted by ts.

    Applies a minimum gap (`_MIN_FLAG_GAP_S`) between consecutive flag
    events so the replay shows natural pacing rather than a burst of
    flags at the start.
    """
    events: list[tuple[float, str, dict]] = []
    for stream_name, fname in _REPLAY_STREAMS.items():
        path = source / fname
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = _parse_ts(ev.get("ts", ""))
                events.append((ts, stream_name, ev))

    # Space flags to at least _MIN_FLAG_GAP_S apart, walking in time order.
    # We do this before the global sort so decisions stay in their original
    # positions — flags shifted forward still appear before their decisions
    # because decisions lagged them by >=15s in the source.
    flag_events = sorted(
        (e for e in events if e[1] == "flags"), key=lambda t: t[0]
    )
    last_flag_ts: float | None = None
    reshaped: list[tuple[float, str, dict]] = []
    for ts, stream, ev in flag_events:
        if last_flag_ts is not None and ts - last_flag_ts < _MIN_FLAG_GAP_S:
            ts = last_flag_ts + _MIN_FLAG_GAP_S
        reshaped.append((ts, stream, ev))
        last_flag_ts = ts
    non_flags = [e for e in events if e[1] != "flags"]
    events = reshaped + non_flags
    events.sort(key=lambda t: t[0])
    return events


def _prepare_run_dir(run_id: str | None, override: str | None = None) -> Path:
    """Create a fresh `<runs-root>/demo_<ts>/` and return its path.

    Uses `shared.paths.get_runs_dir` to resolve the runs root so that
    pipx-installed users get their artifacts under `~/.mdk-orchestra/runs/`
    (and not inside the pipx venv's site-packages).
    """
    from shared.paths import get_runs_dir
    if run_id:
        name = run_id
    else:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"demo_{ts}"
    run_dir = get_runs_dir(override) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    return run_dir


def _write_backend_summary(run_dir: Path, speed: float, source: Path) -> None:
    # Format speed cleanly: "1×" for real-time, "4×" / "10×" etc. for accelerated.
    speed_str = f"{int(speed)}×" if speed == int(speed) else f"{speed:g}×"
    summary = {
        "label": "Demo Replay",
        "detail": f"Anthropic API run ({speed_str})",
        "profile": "demo_replay",
        "source": str(source.relative_to(_REPO_ROOT)) if source.is_relative_to(_REPO_ROOT) else str(source),
        "replay_speed": float(speed),
    }
    (run_dir / "backend_summary.json").write_text(json.dumps(summary, indent=2))


def _run_replay(
    source: Path,
    run_dir: Path,
    speed: float,
    open_files: dict[str, Any] | None = None,
    sleep_fn=time.sleep,
) -> dict:
    """Stream events from `source` into `run_dir` at the given speed.

    `sleep_fn` is injected so tests can substitute a no-op sleeper.

    Returns the meta dict (same contents that get written to replay_meta.json).
    """
    events = _load_timeline(source)
    if not events:
        raise RuntimeError(f"no events found in {source}")

    # Open one writer per stream
    managed = open_files is None
    writers: dict[str, Any] = open_files if open_files is not None else {}
    if managed:
        for stream in _REPLAY_STREAMS:
            writers[stream] = (run_dir / f"{stream}.jsonl").open("w", encoding="utf-8")

    start_epoch = events[0][0]
    wall_start = time.monotonic()
    started_at = _dt.datetime.utcnow().isoformat() + "Z"

    counts = {stream: 0 for stream in _REPLAY_STREAMS}
    try:
        for ts, stream_name, ev in events:
            sim_elapsed = ts - start_epoch
            wall_target = sim_elapsed / speed
            wall_elapsed = time.monotonic() - wall_start
            delay = wall_target - wall_elapsed
            if delay > 0:
                sleep_fn(delay)
            # Retime the event so downstream consumers see a "live" timestamp
            ev_out = dict(ev)
            ev_out["ts"] = (
                _dt.datetime.utcnow().isoformat(timespec="microseconds") + "Z"
            )
            writer = writers.get(stream_name)
            if writer is None:
                continue
            writer.write(json.dumps(ev_out) + "\n")
            writer.flush()
            counts[stream_name] += 1
    finally:
        if managed:
            for w in writers.values():
                try:
                    w.close()
                except Exception:
                    pass

    ended_at = _dt.datetime.utcnow().isoformat() + "Z"
    meta = {
        "replay_source": str(source.relative_to(_REPO_ROOT)) if source.is_relative_to(_REPO_ROOT) else str(source),
        "replay_speed": float(speed),
        "started_at": started_at,
        "ended_at": ended_at,
        "total_events": sum(counts.values()),
        "counts": counts,
    }
    return meta


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def replay_to_run(
    source: Path,
    speed: float = 4.0,
    run_id: str | None = None,
    output_dir: str | None = None,
) -> Path:
    """Replay events from `source` into a fresh `<runs-root>/<id>/` and return the path.

    Also writes `backend_summary.json` before the first event fires (so the
    dashboard's header badge shows "Demo Replay" immediately on open) and
    `replay_meta.json` after the last event lands.
    """
    run_dir = _prepare_run_dir(run_id, output_dir)
    _write_backend_summary(run_dir, speed, source)
    meta = _run_replay(source, run_dir, speed)
    (run_dir / "replay_meta.json").write_text(json.dumps(meta, indent=2))
    return run_dir


def prepare_and_serve_empty(
    *,
    source: Path,
    speed: float = 1.0,
    run_id: str | None = None,
    dashboard_port: int = 8000,
    output_dir: str | None = None,
) -> tuple[Path, subprocess.Popen]:
    """Set up a replay's run directory and dashboard WITHOUT streaming any
    events. Caller is responsible for driving the stream (e.g. on a
    background thread). Returns `(run_dir, dashboard_proc)`.

    Used by the CLI wizard so it can wrap the stream with a live progress
    panel on the main thread.
    """
    run_dir = _prepare_run_dir(run_id, output_dir)
    _write_backend_summary(run_dir, speed, source)
    for stream in _REPLAY_STREAMS:
        (run_dir / f"{stream}.jsonl").touch()

    env = os.environ.copy()
    env["MDK_STREAM_DIR"] = str(run_dir)
    log_path = run_dir / "logs" / "dashboard.log"
    _free_dashboard_port(dashboard_port)
    dashboard_proc = subprocess.Popen(
        [_PYTHON, "-m", "dashboard.main", f"--port={dashboard_port}"],
        env=env,
        cwd=str(_REPO_ROOT),
        stdout=log_path.open("w"),
        stderr=subprocess.STDOUT,
    )
    time.sleep(1.5)  # Flask bind
    return run_dir, dashboard_proc


def replay_and_serve(
    source: Path,
    speed: float = 4.0,
    run_id: str | None = None,
    dashboard_port: int = 8000,
    output_dir: str | None = None,
) -> tuple[Path, subprocess.Popen]:
    """Create the run dir, launch the dashboard, and start the replay
    in the foreground (this function blocks until replay completes).

    Returns `(run_dir, dashboard_proc)`. Caller is responsible for
    terminating the dashboard subprocess.
    """
    run_dir = _prepare_run_dir(run_id, output_dir)
    _write_backend_summary(run_dir, speed, source)

    # Pre-create empty stream files so the dashboard reads don't crash
    for stream in _REPLAY_STREAMS:
        (run_dir / f"{stream}.jsonl").touch()

    env = os.environ.copy()
    env["MDK_STREAM_DIR"] = str(run_dir)

    log_path = run_dir / "logs" / "dashboard.log"
    dashboard_proc = subprocess.Popen(
        [_PYTHON, "-m", "dashboard.main", f"--port={dashboard_port}"],
        env=env,
        cwd=str(_REPO_ROOT),
        stdout=log_path.open("w"),
        stderr=subprocess.STDOUT,
    )

    # Give Flask a moment to bind
    time.sleep(1.5)

    sys.stderr.write(
        f"\n[replay] source: {source}\n"
        f"[replay] run_dir: {run_dir}\n"
        f"[replay] dashboard: http://127.0.0.1:{dashboard_port}/\n"
        f"[replay] speed: {speed}×\n\n"
    )

    # Open files in append mode so the dashboard tailing sees growth
    writers = {
        stream: (run_dir / f"{stream}.jsonl").open("a", encoding="utf-8")
        for stream in _REPLAY_STREAMS
    }
    try:
        meta = _run_replay(source, run_dir, speed, open_files=writers)
    finally:
        for w in writers.values():
            try:
                w.close()
            except Exception:
                pass
    (run_dir / "replay_meta.json").write_text(json.dumps(meta, indent=2))

    return run_dir, dashboard_proc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="mdk-orchestra replay",
        description=(
            "Replay a recorded event-stream run at a configurable speed. "
            "No API key, no LLM required — the dashboard shows the replay "
            "as if it were live."
        ),
    )
    p.add_argument(
        "--source",
        default=str(_REPO_ROOT / "examples" / "demo_replay"),
        help="Source directory with flags/decisions/actions/snapshots JSONL "
             "(default: examples/demo_replay/)",
    )
    p.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (default: 1.0, i.e. real-time). "
             "Pass e.g. --speed 4 for a 4× accelerated replay.",
    )
    p.add_argument(
        "--run-id",
        default=None,
        help="Run identifier (default: demo_<timestamp>)",
    )
    p.add_argument(
        "--dashboard-port",
        type=int,
        default=8000,
        help="Dashboard HTTP port (default: 8000)",
    )
    p.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Just write the replayed stream to disk; don't launch the dashboard.",
    )
    args = p.parse_args(argv)

    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        sys.stderr.write(f"Error: source directory not found: {source}\n")
        return 1

    if args.no_dashboard:
        run_dir = replay_to_run(source, speed=args.speed, run_id=args.run_id)
        print(f"Replay written to: {run_dir}")
        return 0

    run_dir, dashboard_proc = replay_and_serve(
        source, speed=args.speed, run_id=args.run_id, dashboard_port=args.dashboard_port,
    )
    sys.stderr.write(f"\n[replay] done. Run artifacts in {run_dir}\n")
    dashboard_proc.terminate()
    try:
        dashboard_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        dashboard_proc.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
