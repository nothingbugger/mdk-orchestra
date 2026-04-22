"""JSONL-based event bus for MDK Fleet.

No Redis, no Kafka — the prototype runs at 50 miners × 0.2 Hz = 10 events/s,
which a plain JSONL append handles trivially and keeps the whole system
debuggable with `tail -f`.

Design notes
------------
- **One line = one envelope**. Writers never split an event across lines.
- **Append-only**. Consumers tail-follow; we never truncate in-flight streams.
- **Locking**: on POSIX, single-line writes under a few KB are atomic w.r.t.
  O_APPEND, so multiple producers on the same file are safe. We do not add
  explicit locks.
- **Rotation**: not handled here. The A/B experiment runs are scoped by their
  own `ab_run_<id>/` directory; for the live system we rely on manual
  housekeeping for now.
- **Tail semantics**: `tail_events` yields envelopes forever unless
  `stop_after` is set. On EOF it sleeps `poll_interval_s` and tries again.
  This makes it a drop-in `for env in tail_events(...)` loop for consumers.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import IO, Any, Callable

from pydantic import BaseModel, ValidationError

from shared.paths import stream_paths
from shared.schemas.events import Envelope, EventName, Source, parse_event

_DEFAULT_POLL_S = 0.1


def write_event(
    event: EventName,
    source: Source,
    payload: BaseModel | dict[str, Any],
    stream_path: Path | None = None,
    also_live: bool = True,
) -> Envelope:
    """Wrap `payload` in an envelope and append it to the stream.

    Args:
        event: the canonical event name.
        source: the producing module.
        payload: either a Pydantic model from `shared.schemas.events` or a dict
            conforming to its shape. Models are preferred; dicts are a shortcut.
        stream_path: override target file. When None, routes to the canonical
            per-channel file via `StreamPaths.for_event`.
        also_live: if True, also append a copy to the combined `live.jsonl`
            stream used by the dashboard. Routed events always flow through
            both the channel file and `live.jsonl` so the dashboard only needs
            to tail one file.

    Returns the envelope that was written (useful for logging/tests).
    """
    env = Envelope.wrap(event=event, source=source, payload=payload)
    line = env.model_dump_json() + "\n"

    sp = stream_paths()
    sp.root.mkdir(parents=True, exist_ok=True)
    target = stream_path or sp.for_event(event)
    _append(target, line)

    if also_live and target != sp.live:
        _append(sp.live, line)

    return env


def write_raw(envelope: Envelope, stream_path: Path | None = None, also_live: bool = True) -> None:
    """Append an already-built envelope. Use when replaying or forwarding."""
    line = envelope.model_dump_json() + "\n"
    sp = stream_paths()
    sp.root.mkdir(parents=True, exist_ok=True)
    target = stream_path or sp.for_event(envelope.event)
    _append(target, line)
    if also_live and target != sp.live:
        _append(sp.live, line)


def read_events(path: Path) -> Iterator[Envelope]:
    """One-shot read of a JSONL file, yielding parsed envelopes.

    Skips malformed lines with no error — downstream consumers can add
    stricter handling if needed. Use `tail_events` for streaming reads."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for raw in _iter_lines(f):
            yield raw


def tail_events(
    path: Path,
    from_start: bool = False,
    poll_interval_s: float = _DEFAULT_POLL_S,
    stop_after: float | None = None,
    stop_when: Callable[[], bool] | None = None,
) -> Iterator[Envelope]:
    """Follow a JSONL file forever, yielding envelopes as they arrive.

    Args:
        path: the file to tail. Created if missing (empty file), then watched.
        from_start: if True, start from byte 0 (replay + follow). Default
            False, which seeks to EOF before following.
        poll_interval_s: sleep between EOF-polls. Keep ~= the event cadence.
        stop_after: seconds after which to stop and return. None = forever.
        stop_when: optional predicate; return when it becomes True. Checked on
            every poll cycle.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)

    start = time.monotonic()
    with path.open("r", encoding="utf-8") as f:
        if not from_start:
            f.seek(0, 2)  # SEEK_END

        while True:
            for env in _iter_lines(f):
                yield env

            if stop_when and stop_when():
                return
            if stop_after is not None and (time.monotonic() - start) >= stop_after:
                return

            time.sleep(poll_interval_s)


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _append(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line)


def _iter_lines(f: IO[str]) -> Iterator[Envelope]:
    """Yield parsed envelopes from current file position to EOF. Skips lines
    that fail JSON or schema validation — logs nothing by design to keep the
    bus hot-path cheap. Consumers doing strict validation should re-parse."""
    for raw_line in f:
        line = raw_line.rstrip("\n")
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            env, _ = parse_event(raw)
        except ValidationError:
            continue
        yield env
