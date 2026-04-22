"""Live terminal progress panels for `demo` / `run` / `explore`.

The panel polls the run's JSONL stream files every ~500 ms and renders
a mactop-styled panel with counts, autonomy mix, and the most recent
event. It coexists with the browser dashboard — the user can watch
either (or both) since they read the same files.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text


_POLL_INTERVAL_S = 0.5


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def _last_line(path: Path) -> dict | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        with path.open("rb") as f:
            # Seek to end, walk back to find the last complete line
            f.seek(0, 2)
            size = f.tell()
            if size == 0:
                return None
            block = min(4096, size)
            f.seek(size - block)
            tail = f.read(block)
        line = tail.strip().splitlines()[-1]
        return json.loads(line)
    except (OSError, json.JSONDecodeError, IndexError):
        return None


@dataclass
class RunState:
    flags: int = 0
    decisions: int = 0
    actions: int = 0
    autonomy: dict[str, int] = field(default_factory=dict)
    latest_event: str = ""
    total_cost_usd: float = 0.0

    @classmethod
    def from_dir(cls, run_dir: Path) -> "RunState":
        s = cls()
        s.flags = _count_lines(run_dir / "flags.jsonl")
        s.decisions = _count_lines(run_dir / "decisions.jsonl")
        s.actions = _count_lines(run_dir / "actions.jsonl")

        # Walk decisions.jsonl once to count autonomy levels + cost.
        decisions_path = run_dir / "decisions.jsonl"
        if decisions_path.exists():
            try:
                with decisions_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            ev = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        data = ev.get("data") or {}
                        level = data.get("autonomy_level")
                        if level:
                            s.autonomy[level] = s.autonomy.get(level, 0) + 1
                        cost = data.get("total_cost_usd")
                        if isinstance(cost, (int, float)):
                            s.total_cost_usd += float(cost)
            except OSError:
                pass

        # Latest event: decision > flag > action, preferring the most informative.
        last_dec = _last_line(decisions_path)
        if last_dec:
            d = last_dec.get("data", {})
            s.latest_event = (
                f"{d.get('miner_id', '?')} → {d.get('action', '?')} "
                f"({d.get('autonomy_level', '?')})"
            )
        else:
            last_flag = _last_line(run_dir / "flags.jsonl")
            if last_flag:
                d = last_flag.get("data", {})
                s.latest_event = (
                    f"{d.get('miner_id', '?')} flag: {d.get('flag_type', '?')} "
                    f"({d.get('severity', '?')})"
                )
        return s


# ---------------------------------------------------------------------------
# Panel builders
# ---------------------------------------------------------------------------


def _autonomy_line(autonomy: dict[str, int]) -> Text:
    """Render autonomy counts in L1/L2/L3/L4 order with semantic colors."""
    pieces: list[tuple[str, str]] = []
    for level, style in (("L1_observe", "l1"), ("L2_suggest", "l2"),
                         ("L3_bounded_auto", "l3"), ("L4_human_only", "l4")):
        count = autonomy.get(level, 0)
        short = level.split("_", 1)[0]  # "L1", "L2", "L3", "L4"
        pieces.append((f"{short}:{count}", style))
    txt = Text()
    for i, (piece, style) in enumerate(pieces):
        if i:
            txt.append("  ·  ", style="dim")
        txt.append(piece, style=style)
    return txt


def _build_demo_panel(
    state: RunState,
    elapsed_s: float,
    total_events: int,
    title: str,
    dashboard_url: str,
) -> Panel:
    """Demo panel: bounded progress (we know how many events total)."""
    done = state.flags + state.decisions + state.actions
    pct = (done / total_events * 100) if total_events else 0.0
    elapsed_str = f"{int(elapsed_s // 60):02d}:{int(elapsed_s % 60):02d}"

    body = Table.grid(padding=(0, 2))
    body.add_column(style="muted", no_wrap=True)
    body.add_column(style="title")
    body.add_row("Status", Text("Replaying canonical run", style="accent"))
    body.add_row("Elapsed", Text(elapsed_str, style="title"))
    body.add_row("Flags", Text(f"{state.flags}/38", style="title"))
    body.add_row("Decisions", Text(f"{state.decisions}/28", style="title"))
    body.add_row("Autonomy", _autonomy_line(state.autonomy))
    if state.latest_event:
        body.add_row("Latest", Text(state.latest_event, style="muted"))

    # Progress bar rendered inline via Rich
    bar = Progress(
        TextColumn("  [muted]Progress[/]"),
        BarColumn(bar_width=38, complete_style="bar.done", finished_style="accent",
                  pulse_style="bar.remain"),
        TextColumn("[title]{task.percentage:>3.0f}%[/]"),
        expand=False,
    )
    bar.add_task("replay", total=100, completed=pct)

    footer = Text.from_markup(
        f"  [muted]Dashboard:[/] [accent]{dashboard_url}[/]   [muted](Ctrl+C to stop)[/]"
    )
    return Panel(
        Group(body, Text(""), bar, Text(""), footer),
        title=f"[brand]{title}[/]",
        border_style="border.brand",
        padding=(1, 2),
    )


def _build_live_panel(
    state: RunState,
    elapsed_s: float,
    title: str,
    dashboard_url: str,
    show_cost: bool = False,
) -> Panel:
    """Live run panel (API / Local / Explore): unbounded counts."""
    elapsed_str = f"{int(elapsed_s // 60):02d}:{int(elapsed_s % 60):02d}"
    body = Table.grid(padding=(0, 2))
    body.add_column(style="muted", no_wrap=True)
    body.add_column(style="title")
    body.add_row("Status", Text("Orchestra active", style="accent"))
    body.add_row("Elapsed", Text(elapsed_str, style="title"))
    body.add_row("Flags", Text(str(state.flags), style="title"))
    body.add_row("Decisions", Text(str(state.decisions), style="title"))
    if state.autonomy:
        body.add_row("Autonomy mix", _autonomy_line(state.autonomy))
    if state.latest_event:
        body.add_row("Latest", Text(state.latest_event, style="muted"))
    if show_cost:
        body.add_row("Est. cost", Text(f"${state.total_cost_usd:.4f}", style="title"))

    footer = Text.from_markup(
        f"  [muted]Dashboard:[/] [accent]{dashboard_url}[/]   [muted](Ctrl+C to stop)[/]"
    )
    return Panel(
        Group(body, Text(""), footer),
        title=f"[brand]{title}[/]",
        border_style="border.brand",
        padding=(1, 2),
    )


# ---------------------------------------------------------------------------
# Background watcher
# ---------------------------------------------------------------------------


class ProgressWatcher:
    """Run a Rich Live panel bound to a directory's JSONL streams.

    Designed to be used as a context manager around a blocking operation
    (e.g. waiting for a subprocess). The watcher thread polls the files
    every _POLL_INTERVAL_S; Live refreshes the panel accordingly.
    """

    def __init__(
        self,
        console,
        run_dir: Path,
        *,
        title: str,
        dashboard_url: str = "http://127.0.0.1:8000/",
        mode: str = "live",                   # "demo" | "live" | "explore"
        total_events: int = 0,                 # for demo-mode bounded progress
        show_cost: bool = False,
    ) -> None:
        self.console = console
        self.run_dir = run_dir
        self.title = title
        self.dashboard_url = dashboard_url
        self.mode = mode
        self.total_events = total_events
        self.show_cost = show_cost
        self._stop = threading.Event()
        self._state = RunState()
        self._start_t = 0.0
        self._live: Live | None = None

    def _render(self) -> Panel:
        elapsed = time.monotonic() - self._start_t
        if self.mode == "demo":
            return _build_demo_panel(
                self._state, elapsed, self.total_events,
                self.title, self.dashboard_url,
            )
        return _build_live_panel(
            self._state, elapsed, self.title, self.dashboard_url,
            show_cost=self.show_cost,
        )

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            self._state = RunState.from_dir(self.run_dir)
            if self._live is not None:
                self._live.update(self._render())
            self._stop.wait(_POLL_INTERVAL_S)

    def __enter__(self) -> "ProgressWatcher":
        self._start_t = time.monotonic()
        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=4,
            transient=False,
        )
        self._live.__enter__()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass
        if self._live is not None:
            try:
                self._live.__exit__(exc_type, exc, tb)
            except Exception:
                pass

    @property
    def state(self) -> RunState:
        return self._state
