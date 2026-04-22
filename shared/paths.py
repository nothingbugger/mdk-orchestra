"""Canonical filesystem paths for MDK Fleet runtime artefacts.

Every module reads paths from here — never hard-codes. Defaults target the
spec-defined `/run/mdk_fleet/...` layout, but any path can be overridden via
environment variables (see `.env.example`) so the system runs on laptops
without root access to `/run`.

The module is pure: it does not create directories. Callers that intend to
write should first call `ensure_stream_dirs()` or pass the dirs to
`shared.event_bus.write_event` which will create them on first write.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Local fallback root — used when /run/mdk_fleet is not writable (most dev
# machines). Points at the repo's top-level `events/` and `memory/` dirs.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_FALLBACK_STREAM = _REPO_ROOT / "events"
_LOCAL_FALLBACK_MEMORY = _REPO_ROOT / "memory" / "agent_episodes"
_LOCAL_FALLBACK_LOG = _REPO_ROOT / "events" / "log"


def _env_path(var: str, default: Path) -> Path:
    raw = os.environ.get(var)
    if raw:
        return Path(raw).expanduser().resolve()
    return default


def _prefer_writable(primary: Path, fallback: Path) -> Path:
    """Return `primary` if its parent is writable, else `fallback`.

    This makes `/run/mdk_fleet/...` the default on the Mac mini (where Daniele
    may have set up a tmpfs) while falling back to the repo-local `events/`
    dir on dev machines without root."""
    try:
        primary.parent.mkdir(parents=True, exist_ok=True)
        # Touch-probe: if the parent is writable we're done.
        return primary
    except (PermissionError, OSError):
        return fallback


@dataclass(frozen=True)
class StreamPaths:
    """Paths to the live event streams. One JSONL file per channel."""

    root: Path

    @property
    def live(self) -> Path:
        return self.root / "live.jsonl"

    @property
    def telemetry(self) -> Path:
        return self.root / "telemetry.jsonl"

    @property
    def kpis(self) -> Path:
        return self.root / "kpis.jsonl"

    @property
    def snapshots(self) -> Path:
        return self.root / "snapshots.jsonl"

    @property
    def flags(self) -> Path:
        return self.root / "flags.jsonl"

    @property
    def decisions(self) -> Path:
        return self.root / "decisions.jsonl"

    @property
    def actions(self) -> Path:
        return self.root / "actions.jsonl"

    def for_event(self, event: str) -> Path:
        """Route an event name to its canonical stream file."""
        return {
            "telemetry_tick": self.telemetry,
            "kpi_update": self.kpis,
            "fleet_snapshot": self.snapshots,
            "flag_raised": self.flags,
            "reasoning_request": self.live,
            "reasoning_response": self.live,
            "orchestrator_decision": self.decisions,
            "action_taken": self.actions,
            "episodic_memory_write": self.live,
        }.get(event, self.live)


@dataclass(frozen=True)
class MemoryPaths:
    """Per-agent episodic memory directories."""

    root: Path

    def agent_dir(self, agent_name: str) -> Path:
        return self.root / agent_name

    def agent_events(self, agent_name: str) -> Path:
        return self.agent_dir(agent_name) / "events.jsonl"


# ---------------------------------------------------------------------------
# Runs directory — where per-session run artifacts live
# ---------------------------------------------------------------------------


def get_runs_dir(override: str | Path | None = None) -> Path:
    """Return the directory under which per-run subdirs are created.

    Resolution priority (highest first):
      1. Explicit `override` argument (from a CLI flag).
      2. `MDK_RUNS_DIR` environment variable.
      3. Default: `~/.mdk-orchestra/runs/`.

    The returned directory exists on return (created with parents).

    Rationale: previous versions defaulted to a repo-relative
    `events/ab_runs/` path, which worked for developers who installed
    the repo in editable mode but broke for users who `pipx install`-ed
    the package — that path resolved inside `site-packages/`, meaning
    runs got wiped on `pipx uninstall`. The user-home default survives
    any package lifecycle operation and respects POSIX conventions for
    application state.
    """
    if override is not None:
        d = Path(override).expanduser().resolve()
    else:
        raw = os.environ.get("MDK_RUNS_DIR")
        if raw:
            d = Path(raw).expanduser().resolve()
        else:
            d = Path.home() / ".mdk-orchestra" / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Public accessors
# ---------------------------------------------------------------------------


def stream_paths() -> StreamPaths:
    """Default stream paths (env-overridable)."""
    default = _prefer_writable(
        Path("/run/mdk_fleet/stream"),
        _LOCAL_FALLBACK_STREAM,
    )
    root = _env_path("MDK_STREAM_DIR", default)
    return StreamPaths(root=root)


def memory_paths() -> MemoryPaths:
    """Default memory paths (env-overridable)."""
    default = _prefer_writable(
        Path("/run/mdk_fleet/memory"),
        _LOCAL_FALLBACK_MEMORY,
    )
    root = _env_path("MDK_MEMORY_DIR", default)
    return MemoryPaths(root=root)


def log_dir() -> Path:
    """Persistent (not cleared between runs) log dir."""
    default = _prefer_writable(Path("/run/mdk_fleet/log"), _LOCAL_FALLBACK_LOG)
    return _env_path("MDK_LOG_DIR", default)


def ab_run_dir(run_id: str) -> Path:
    """Per-A/B-run isolated directory under the same root as streams."""
    root = stream_paths().root.parent
    return root / f"ab_run_{run_id}"


def ensure_stream_dirs() -> StreamPaths:
    """Create stream root if missing. Returns the paths for chaining."""
    sp = stream_paths()
    sp.root.mkdir(parents=True, exist_ok=True)
    return sp


def ensure_memory_dir(agent_name: str) -> Path:
    """Create per-agent memory dir if missing. Returns the dir path."""
    mp = memory_paths()
    agent_dir = mp.agent_dir(agent_name)
    agent_dir.mkdir(parents=True, exist_ok=True)
    return agent_dir
