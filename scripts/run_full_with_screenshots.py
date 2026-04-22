"""Coordinator: full A/B run + dashboard pointed at Track A + Playwright shots.

Problem it solves: the A/B runner creates two independent stream dirs
(track_a/, track_b/) under a per-run UUID path. The dashboard and the
Playwright capture script both need to read from *the same* stream dir
as Track A (the Orchestra-ON track — the one whose reasoning traces are
interesting). This coordinator picks an output root ahead of time,
launches the A/B runner, waits for the track_a/stream directory to
exist, then launches dashboard + capture with MDK_STREAM_DIR pinned to
that path.

Invocation:
    ANTHROPIC_API_KEY=... \\
        python scripts/run_full_with_screenshots.py \\
        --duration-min 12 --miners 50 --seed 43

`--mock` runs the A/B in MDK_AGENT_MOCK=1 for pre-flight verification;
the capture script stays in mock mode so missing beats warn instead of
fail.
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _spawn(cmd: list[str], env: dict[str, str], log_path: Path) -> subprocess.Popen[bytes]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = log_path.open("wb")
    return subprocess.Popen(
        cmd,
        stdout=fh,
        stderr=subprocess.STDOUT,
        cwd=REPO,
        env=env,
    )


def _wait_for_dir(path: Path, timeout_s: float) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if path.exists() and path.is_dir():
            return True
        time.sleep(0.5)
    return False


def _find_track_a_stream(output_root: Path, timeout_s: float) -> Path | None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        candidates = sorted(output_root.glob("*/track_a/stream"), key=lambda p: p.stat().st_mtime if p.exists() else 0)
        if candidates:
            return candidates[-1]
        time.sleep(0.5)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration-min", type=float, default=12.0)
    ap.add_argument("--miners", type=int, default=50)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--scenario", default="realistic")
    ap.add_argument("--output-root", default="/tmp/mdk_ab_live")
    ap.add_argument("--mock", action="store_true",
                    help="Run A/B in mock mode (no Claude API calls).")
    ap.add_argument("--dashboard-port", type=int, default=8000)
    args = ap.parse_args()

    if not args.mock and "ANTHROPIC_API_KEY" not in os.environ:
        print("ANTHROPIC_API_KEY required unless --mock is set", file=sys.stderr)
        return 2

    output_root = Path(args.output_root)
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    base_env = {
        **os.environ,
        "PYTHONPATH": str(REPO),
    }
    if args.mock:
        base_env["MDK_AGENT_MOCK"] = "1"
    else:
        base_env.pop("MDK_AGENT_MOCK", None)

    py = sys.executable
    logs_dir = output_root / "_coordinator_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    procs: list[subprocess.Popen[bytes]] = []

    def cleanup(*_a) -> None:
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
        # ---- 1. Launch A/B runner ----
        ab_cmd = [
            py, "-m", "ab_experiment.main",
            "--scenario", args.scenario,
            "--duration-min", str(int(args.duration_min * 10)),  # wall × 10 = sim-min
            "--n-miners", str(args.miners),
            "--seed", str(args.seed),
            "--output", str(output_root),
            "--log-level", "INFO",
        ]
        if not args.mock:
            ab_cmd.append("--api-mode")

        print(f"[coord] launching A/B: {' '.join(ab_cmd)}", flush=True)
        ab_proc = _spawn(ab_cmd, base_env, logs_dir / "ab_experiment.out")
        procs.append(ab_proc)

        # ---- 2. Wait for track_a/stream to appear ----
        track_a_stream = _find_track_a_stream(output_root, timeout_s=30.0)
        if track_a_stream is None:
            print("[fatal] track_a/stream never appeared", file=sys.stderr)
            cleanup()
            return 4
        print(f"[coord] track_a stream = {track_a_stream}", flush=True)

        # ---- 3. Launch dashboard pointed at track_a ----
        dash_env = {**base_env, "MDK_STREAM_DIR": str(track_a_stream),
                    "MDK_MEMORY_DIR": str(track_a_stream.parent.parent / "memory")}
        dash_proc = _spawn(
            [py, "-m", "dashboard.main", "--port", str(args.dashboard_port)],
            dash_env,
            logs_dir / "dashboard.out",
        )
        procs.append(dash_proc)
        time.sleep(3.0)  # let Flask bind

        # ---- 4. Launch capture script ----
        cap_cmd = [
            py, "-m", "report.capture_screenshots",
            "--total-duration-min", str(args.duration_min),
            "--stream-dir", str(track_a_stream),
            "--url", f"http://127.0.0.1:{args.dashboard_port}",
        ]
        if args.mock:
            cap_cmd.append("--mock")

        print(f"[coord] launching capture: {' '.join(cap_cmd)}", flush=True)
        cap_env = {**base_env, "MDK_STREAM_DIR": str(track_a_stream)}
        cap_proc = _spawn(cap_cmd, cap_env, logs_dir / "capture.out")
        procs.append(cap_proc)

        # ---- 5. Wait: capture finishes first (internal --duration-min) ----
        cap_rc = cap_proc.wait()
        print(f"[coord] capture exited rc={cap_rc}", flush=True)

        # ---- 6. Wait for A/B to complete metrics + figures ----
        try:
            ab_rc = ab_proc.wait(timeout=180)
        except subprocess.TimeoutExpired:
            print("[warn] A/B still running 180s after capture done — terminating", flush=True)
            ab_proc.terminate()
            ab_rc = ab_proc.wait(timeout=30)
        print(f"[coord] ab_experiment exited rc={ab_rc}", flush=True)

        return cap_rc or ab_rc or 0

    finally:
        cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
