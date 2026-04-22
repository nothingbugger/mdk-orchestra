"""Event-driven dashboard screenshot automation.

Each new flag, each new orchestrator_decision, and each new action_taken
gets its own screenshot a few seconds later (time for the dashboard SSE
to push and render). In addition, an "ambient" shot is taken every
`--ambient-interval-min` minutes so the resulting gallery shows both
event-specific states and calm-state fleet maps.

File naming convention (filesafe, sortable):
  evt_{Tmmmss}_flag_{miner_id}_{flag_type}.png
  evt_{Tmmmss}_decision_{miner_id}_{action}.png
  evt_{Tmmmss}_action_{miner_id}_{action}.png
  ambient_{Tmmmss}_fleet.png

Where {Tmmmss} is seconds-since-start zero-padded to 5 digits (e.g.
T00180 = 3 minutes into the run). This keeps `ls` output chronological.

Invocation:
  python -m report.capture_screenshots --total-duration-min 30
  python -m report.capture_screenshots --total-duration-min 2 --mock

`--mock` weakens failure modes (doesn't exit non-zero if no events
arrive) and is intended for pre-flight dashboard-only sanity checks.

Stream dir resolution: same cascade as before (CLI arg → env →
spec path → repo-local events/).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from queue import Queue, Empty
from typing import Any

REPO = Path(__file__).resolve().parent.parent
_SAFE_CHARS = re.compile(r"[^a-zA-Z0-9._-]+")


def _resolve_stream_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg).expanduser().resolve()
    env = os.environ.get("MDK_STREAM_DIR")
    if env:
        return Path(env).expanduser().resolve()
    spec = Path("/run/mdk_fleet/stream")
    if spec.exists() and os.access(spec, os.W_OK):
        return spec
    return REPO / "events"


def _sanitize(s: str) -> str:
    return _SAFE_CHARS.sub("_", s).strip("_")[:40] or "x"


def _tsec(start: float) -> str:
    return f"T{int(time.monotonic() - start):05d}"


def _tail_events(
    path: Path,
    out_q: Queue,
    event_filter: str,
    stop_event: threading.Event,
    from_end: bool = True,
) -> None:
    """Tail a JSONL file, parse each line, push parsed envelope to queue."""
    # Wait for file to exist
    while not path.exists() and not stop_event.is_set():
        time.sleep(0.2)
    try:
        with path.open("r", encoding="utf-8") as f:
            if from_end:
                f.seek(0, 2)  # SEEK_END
            while not stop_event.is_set():
                line = f.readline()
                if not line:
                    time.sleep(0.2)
                    continue
                try:
                    env = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if env.get("event") == event_filter:
                    out_q.put(env)
    except Exception as exc:  # noqa: BLE001
        print(f"[tail-error] {path.name}: {exc}", file=sys.stderr, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-duration-min", type=float, default=30.0)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--output-dir", default=str(REPO / "report" / "screenshots"))
    ap.add_argument("--stream-dir", default=None)
    ap.add_argument("--mock", action="store_true",
                    help="Relax failure modes — don't exit non-zero on 0 events.")
    ap.add_argument("--ready-selector", default=".fleet-grid")
    ap.add_argument("--viewport-width", type=int, default=1600)
    ap.add_argument("--viewport-height", type=int, default=1000)
    ap.add_argument("--device-scale-factor", type=float, default=2.0)
    ap.add_argument("--ambient-interval-min", type=float, default=3.0,
                    help="Minutes between ambient fleet screenshots.")
    ap.add_argument("--event-ui-settle-s", type=float, default=3.0,
                    help="Seconds between event arrival and screenshot (for SSE to paint).")
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright  # noqa: import-within-function

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stream_dir = _resolve_stream_dir(args.stream_dir)
    flags_path = stream_dir / "flags.jsonl"
    decisions_path = stream_dir / "decisions.jsonl"
    actions_path = stream_dir / "actions.jsonl"

    print(f"[capture] stream_dir={stream_dir}", flush=True)
    print(f"[capture] output_dir={out_dir}", flush=True)
    print(f"[capture] total_duration_min={args.total_duration_min}", flush=True)
    print(f"[capture] ambient_interval_min={args.ambient_interval_min}", flush=True)

    # --- tailer threads ---
    stop_event = threading.Event()
    flag_q: Queue = Queue()
    decision_q: Queue = Queue()
    action_q: Queue = Queue()

    t_flags = threading.Thread(
        target=_tail_events,
        args=(flags_path, flag_q, "flag_raised", stop_event),
        name="tail-flags",
        daemon=True,
    )
    t_decs = threading.Thread(
        target=_tail_events,
        args=(decisions_path, decision_q, "orchestrator_decision", stop_event),
        name="tail-decisions",
        daemon=True,
    )
    t_acts = threading.Thread(
        target=_tail_events,
        args=(actions_path, action_q, "action_taken", stop_event),
        name="tail-actions",
        daemon=True,
    )
    for t in (t_flags, t_decs, t_acts):
        t.start()

    counts = {"flag": 0, "decision": 0, "action": 0, "ambient": 0}
    start = time.monotonic()
    deadline = start + args.total_duration_min * 60.0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": args.viewport_width, "height": args.viewport_height},
            device_scale_factor=args.device_scale_factor,
        )
        page = context.new_page()

        try:
            page.goto(args.url, wait_until="domcontentloaded", timeout=30_000)
            page.wait_for_selector(args.ready_selector, timeout=30_000)
        except Exception as exc:  # noqa: BLE001
            print(f"[fatal] dashboard not reachable at {args.url}: {exc}", file=sys.stderr)
            stop_event.set()
            browser.close()
            return 3

        time.sleep(2.0)

        # Initial baseline
        bpath = out_dir / f"ambient_{_tsec(start)}_fleet.png"
        page.screenshot(path=str(bpath), full_page=False)
        counts["ambient"] += 1
        print(f"[shot] {bpath.name}", flush=True)

        next_ambient_at = args.ambient_interval_min * 60.0

        while time.monotonic() < deadline:
            sleep_quantum = 0.5

            # Ambient beat
            elapsed = time.monotonic() - start
            if elapsed >= next_ambient_at:
                bpath = out_dir / f"ambient_{_tsec(start)}_fleet.png"
                page.screenshot(path=str(bpath), full_page=False)
                counts["ambient"] += 1
                print(f"[shot] {bpath.name}", flush=True)
                next_ambient_at += args.ambient_interval_min * 60.0

            # Flag events
            try:
                env = flag_q.get_nowait()
            except Empty:
                pass
            else:
                d = env.get("data", {})
                miner = _sanitize(str(d.get("miner_id", "x")))
                ftype = _sanitize(str(d.get("flag_type", "x")))
                time.sleep(args.event_ui_settle_s)
                sp = out_dir / f"evt_{_tsec(start)}_flag_{miner}_{ftype}.png"
                page.screenshot(path=str(sp), full_page=False)
                counts["flag"] += 1
                print(f"[shot] {sp.name}", flush=True)
                continue  # fast-loop to drain any queued flags

            # Decision events
            try:
                env = decision_q.get_nowait()
            except Empty:
                pass
            else:
                d = env.get("data", {})
                miner = _sanitize(str(d.get("miner_id", "x")))
                act = _sanitize(str(d.get("action", "x")))
                time.sleep(args.event_ui_settle_s)
                sp = out_dir / f"evt_{_tsec(start)}_decision_{miner}_{act}.png"
                page.screenshot(path=str(sp), full_page=False)
                counts["decision"] += 1
                print(f"[shot] {sp.name}", flush=True)
                continue

            # Action events
            try:
                env = action_q.get_nowait()
            except Empty:
                pass
            else:
                d = env.get("data", {})
                miner = _sanitize(str(d.get("miner_id", "x")))
                act = _sanitize(str(d.get("action", "x")))
                time.sleep(args.event_ui_settle_s)
                sp = out_dir / f"evt_{_tsec(start)}_action_{miner}_{act}.png"
                page.screenshot(path=str(sp), full_page=False)
                counts["action"] += 1
                print(f"[shot] {sp.name}", flush=True)
                continue

            time.sleep(sleep_quantum)

        # Final ambient
        bpath = out_dir / f"ambient_{_tsec(start)}_fleet.png"
        page.screenshot(path=str(bpath), full_page=False)
        counts["ambient"] += 1
        print(f"[shot] {bpath.name}", flush=True)

        browser.close()

    stop_event.set()
    total = sum(counts.values())
    print(
        f"[done] shots: flag={counts['flag']} decision={counts['decision']} "
        f"action={counts['action']} ambient={counts['ambient']} total={total}",
        flush=True,
    )
    if total == 0 and not args.mock:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
