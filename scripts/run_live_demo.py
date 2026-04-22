"""Live demo orchestrator for MDK Fleet.

Launches all six services (simulator, ingest, detector, maestro, action,
dashboard), watches the stream files, and captures five screenshots at
narrative beats:

  01_fleet_healthy       baseline, all mint
  02_kpi_live            after KPIs stabilize
  03_flag_raised         first flag hits the feed
  04_maestro_reasoning   first orchestrator_decision appears
  05_post_action         miner recovers after throttle

Expects the dashboard to already be open in the foreground at
http://127.0.0.1:8000 so `screencapture -x` captures the browser window.

Environment:
  ANTHROPIC_API_KEY    must be set (real API mode; NOT mock).
  MDK_STREAM_DIR       must be set (isolated stream dir for this run).
  MDK_MEMORY_DIR       must be set (isolated memory dir for this run).

Usage:
  python scripts/run_live_demo.py --duration-min 12 --sensitivity high --speed 8
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SHOTS = REPO / "report" / "screenshots"
SHOTS.mkdir(parents=True, exist_ok=True)


def _shoot(name: str) -> None:
    """Take a macOS screenshot at full desktop resolution (retina-native)."""
    path = SHOTS / name
    subprocess.run(
        ["screencapture", "-x", "-t", "png", str(path)],
        check=True,
    )
    size = path.stat().st_size if path.exists() else 0
    print(f"[shot] {name}  ({size // 1024} KB)", flush=True)


def _file_size(p: Path) -> int:
    try:
        return p.stat().st_size
    except FileNotFoundError:
        return 0


def _spawn(cmd: list[str], log_path: Path, env: dict[str, str]) -> subprocess.Popen[bytes]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = log_path.open("wb")
    return subprocess.Popen(
        cmd,
        stdout=fh,
        stderr=subprocess.STDOUT,
        cwd=REPO,
        env=env,
    )


def _inject_synthetic_flag(stream_dir: Path, miner_id: str = "m017") -> None:
    """Emergency path: if no organic flag arrives, inject one so the demo
    can proceed. The reasoning from Maestro + specialists is still
    Claude-real — only the flag itself is synthetic."""
    import json
    from datetime import datetime, timezone

    envelope = {
        "event": "flag_raised",
        "ts": datetime.now(tz=timezone.utc).isoformat(),
        "source": "detector",
        "data": {
            "flag_id": f"flg_demo_{miner_id}",
            "miner_id": miner_id,
            "flag_type": "hashrate_degradation",
            "severity": "warn",
            "confidence": 0.74,
            "source_tool": "rule_engine",
            "evidence": {
                "metric": "hashrate_th",
                "window_min": 30.0,
                "recent_mean": 97.8,
                "baseline_mean": 104.0,
                "z_score": -3.4,
            },
            "raw_score": 0.81,
        },
    }
    flags_path = stream_dir / "flags.jsonl"
    flags_path.parent.mkdir(parents=True, exist_ok=True)
    # Also write to live.jsonl so dashboard sees it
    live_path = stream_dir / "live.jsonl"
    line = json.dumps(envelope) + "\n"
    with flags_path.open("a") as f:
        f.write(line)
    with live_path.open("a") as f:
        f.write(line)
    print(f"[inject] synthetic flag on {miner_id}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration-min", type=int, default=12)
    ap.add_argument("--sensitivity", default="high", choices=["low", "medium", "high"])
    ap.add_argument("--speed", type=int, default=8, help="simulator wall-clock speedup")
    ap.add_argument("--n-miners", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--shot-01-at", type=float, default=90, help="seconds into run for shot 01"
    )
    ap.add_argument(
        "--shot-02-at", type=float, default=240, help="seconds into run for shot 02"
    )
    ap.add_argument(
        "--organic-flag-timeout",
        type=float,
        default=420,
        help="seconds to wait for an organic flag before injecting one",
    )
    args = ap.parse_args()

    if "ANTHROPIC_API_KEY" not in os.environ:
        print("ANTHROPIC_API_KEY must be set", file=sys.stderr)
        return 2
    if "MDK_STREAM_DIR" not in os.environ or "MDK_MEMORY_DIR" not in os.environ:
        print("MDK_STREAM_DIR and MDK_MEMORY_DIR must be set", file=sys.stderr)
        return 2
    if os.environ.get("MDK_AGENT_MOCK"):
        print("MDK_AGENT_MOCK is set — unset it for a real API-mode demo", file=sys.stderr)
        return 2

    stream_dir = Path(os.environ["MDK_STREAM_DIR"])
    stream_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = stream_dir.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    env = {
        **os.environ,
        "PYTHONPATH": str(REPO),
    }
    py = sys.executable  # inherited venv python

    procs: list[subprocess.Popen[bytes]] = []

    def cleanup(*_a) -> None:
        print("[cleanup] terminating services", flush=True)
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()

    signal.signal(signal.SIGINT, lambda *_: (cleanup(), sys.exit(130)))

    try:
        # 1 simulator
        procs.append(
            _spawn(
                [
                    py, "-m", "simulator.main",
                    "--n-miners", str(args.n_miners),
                    "--tick", "5",
                    "--duration", str(args.duration_min * 60 * args.speed),
                    "--speed", str(args.speed),
                    "--seed", str(args.seed),
                ],
                logs_dir / "simulator.log",
                env,
            )
        )
        time.sleep(3)

        # 2 ingest
        procs.append(
            _spawn([py, "-m", "ingest.main"], logs_dir / "ingest.log", env)
        )
        time.sleep(2)

        # 3 detector
        procs.append(
            _spawn(
                [py, "-m", "deterministic_tools.main", "--sensitivity", args.sensitivity],
                logs_dir / "detector.log",
                env,
            )
        )
        time.sleep(2)

        # 4 maestro (API MODE — no mock)
        procs.append(
            _spawn([py, "-m", "agents.main"], logs_dir / "maestro.log", env)
        )
        time.sleep(2)

        # 5 action executor
        procs.append(
            _spawn([py, "-m", "action.main"], logs_dir / "action.log", env)
        )
        time.sleep(2)

        # 6 dashboard
        procs.append(
            _spawn([py, "-m", "dashboard.main"], logs_dir / "dashboard.log", env)
        )

        print(f"[start] all services up. demo duration: {args.duration_min} min", flush=True)
        print(f"[start] open http://127.0.0.1:8000 in your browser NOW", flush=True)
        time.sleep(5)  # let dashboard start serving

        t0 = time.monotonic()

        # --- SHOT 01 — fleet_healthy
        wait = max(0, args.shot_01_at - (time.monotonic() - t0))
        if wait > 0:
            time.sleep(wait)
        _shoot("01_fleet_healthy.png")

        # --- SHOT 02 — kpi_live
        wait = max(0, args.shot_02_at - (time.monotonic() - t0))
        if wait > 0:
            time.sleep(wait)
        _shoot("02_kpi_live.png")

        # --- SHOT 03 — flag_raised (wait for organic, else inject)
        flags_path = stream_dir / "flags.jsonl"
        print("[wait] watching flags.jsonl for first flag...", flush=True)
        flag_deadline = time.monotonic() + args.organic_flag_timeout
        organic = False
        while time.monotonic() < flag_deadline:
            if _file_size(flags_path) > 0:
                organic = True
                break
            time.sleep(2)

        if not organic:
            print("[wait] no organic flag in time — injecting synthetic", flush=True)
            _inject_synthetic_flag(stream_dir)

        time.sleep(3)  # let dashboard render the flag
        _shoot("03_flag_raised.png")

        # --- SHOT 04 — maestro_reasoning (wait for first decision)
        decisions_path = stream_dir / "decisions.jsonl"
        print("[wait] watching decisions.jsonl for first decision...", flush=True)
        dec_deadline = time.monotonic() + 180
        while time.monotonic() < dec_deadline:
            if _file_size(decisions_path) > 0:
                break
            time.sleep(2)
        time.sleep(5)  # let the reasoning trace render fully
        _shoot("04_maestro_reasoning.png")

        # --- SHOT 05 — post_action
        actions_path = stream_dir / "actions.jsonl"
        print("[wait] watching actions.jsonl for first action...", flush=True)
        act_deadline = time.monotonic() + 180
        while time.monotonic() < act_deadline:
            if _file_size(actions_path) > 0:
                break
            time.sleep(2)
        # Give the simulator time to respond to the throttle (recovery)
        print("[wait] +90s for recovery...", flush=True)
        time.sleep(90)
        _shoot("05_post_action.png")

        print("[done] all five screenshots saved in report/screenshots/", flush=True)

        # Keep services running a little longer so Daniele can poke around
        print("[hold] services stay up for 60s more — Ctrl+C to stop early", flush=True)
        time.sleep(60)

        return 0

    finally:
        cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
