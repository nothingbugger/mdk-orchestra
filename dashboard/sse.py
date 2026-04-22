"""SSE (Server-Sent Events) generator helpers for the MDK Fleet dashboard.

Each generator tails a JSONL stream file and yields SSE-formatted text.
Flask routes use these as streaming response bodies.

SSE wire format:
    data: <json-line>\\n
    \\n

A `keepalive` comment is emitted every `keepalive_s` seconds so proxies
and browsers don't time out the connection.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)


def _sse_line(data: dict[str, Any]) -> str:
    """Format one dict as a single SSE data line."""
    return f"data: {json.dumps(data, default=str)}\n\n"


def _keepalive() -> str:
    """SSE comment used as keepalive — browsers ignore these."""
    return f": keepalive {int(time.time())}\n\n"


def tail_jsonl_sse(
    path: Path,
    poll_interval_s: float = 0.2,
    keepalive_s: float = 15.0,
    from_start: bool = False,
) -> Iterator[str]:
    """Tail a JSONL file and yield SSE-formatted strings forever.

    Args:
        path: path to the JSONL stream file.
        poll_interval_s: seconds to sleep between EOF polls.
        keepalive_s: emit a keepalive comment this often (seconds).
        from_start: if True, replay file from line 0 then follow.
            Defaults False (seek to EOF, then follow new lines only).

    Yields SSE-formatted strings:  ``"data: {...}\\n\\n"``
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)

    last_keepalive = time.monotonic()

    try:
        with path.open("r", encoding="utf-8") as f:
            if not from_start:
                f.seek(0, 2)  # SEEK_END

            while True:
                raw_line = f.readline()
                if raw_line:
                    raw_line = raw_line.rstrip("\n")
                    if not raw_line:
                        continue
                    try:
                        obj = json.loads(raw_line)
                    except json.JSONDecodeError:
                        log.debug("sse.skip_malformed_json", line=raw_line[:80])
                        continue
                    yield _sse_line(obj)
                else:
                    # EOF — poll
                    now = time.monotonic()
                    if now - last_keepalive >= keepalive_s:
                        yield _keepalive()
                        last_keepalive = now
                    time.sleep(poll_interval_s)
    except GeneratorExit:
        pass
    except Exception as exc:
        log.error("sse.generator_error", error=str(exc))
        raise


def replay_jsonl_sse(path: Path, limit: int = 100) -> Iterator[str]:
    """One-shot replay of the last `limit` lines from a JSONL file as SSE.

    Used to seed the client with recent history on first connect, before
    switching to the live tail.

    Args:
        path: JSONL file to replay.
        limit: maximum number of lines to send.

    Yields SSE-formatted strings.
    """
    if not path.exists():
        return

    lines: list[str] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.rstrip("\n")
                if raw:
                    lines.append(raw)
    except OSError as exc:
        log.error("sse.replay_read_error", path=str(path), error=str(exc))
        return

    # Take the last `limit` lines
    for raw_line in lines[-limit:]:
        try:
            obj = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        yield _sse_line(obj)
