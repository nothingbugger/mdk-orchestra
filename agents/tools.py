"""Agent tools — primitives that Maestro (and only Maestro) can call.

The only tool defined here today is `write_memory_pattern`. Specialists
do not call tools from this module — they are passive readers of the
`<domain>_memory.md` files Maestro curates.

Design notes
------------

- **Atomic write**: every update goes through tmp-file + `os.replace`
  so a crash mid-write never leaves a memory file half-flushed.
- **Pattern match by name**: a pattern entry is uniquely identified by
  its `pattern_name` within a file. Re-writing the same name with
  `increment_if_exists=True` updates the existing block (occurrences
  += 1, Last seen bumped, confidence rolling-averaged); with
  `increment_if_exists=False` it is a no-op if the name already exists
  (prevents the curator from overwriting a careful prior entry).
- **LRU cap**: each memory file caps at `MEMORY_CAP_PATTERNS` entries.
  On overflow we evict the pattern with the fewest occurrences; ties
  break on oldest `Last seen`. The eviction is logged on stdout via
  structlog so curator runs are auditable.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

_LOG = structlog.get_logger(__name__)

MEMORY_CAP_PATTERNS: int = 50

_TARGET_FILE_MAP: dict[str, str] = {
    "maestro": "agents/maestro_memory.md",
    "voltage": "agents/voltage_memory.md",
    "hashrate": "agents/hashrate_memory.md",
    "environment": "agents/environment_memory.md",
    "power": "agents/power_memory.md",
}

_PATTERN_HEADING_RE = re.compile(r"^## Pattern: (.+)$", re.MULTILINE)
_PATTERN_BLOCK_SEP = "\n---\n"


# ---------------------------------------------------------------------------
# Tool schema — matches the Anthropic tool-use input_schema conventions
# ---------------------------------------------------------------------------

WRITE_MEMORY_PATTERN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "target_file": {
            "type": "string",
            "enum": list(_TARGET_FILE_MAP.keys()),
            "description": (
                "Which memory file to write into. 'maestro' is for "
                "cross-domain lessons; the others are domain-specific "
                "and are read by the corresponding specialist agent."
            ),
        },
        "pattern_name": {
            "type": "string",
            "description": "snake_case unique name within the target file.",
        },
        "signature": {
            "type": "string",
            "description": (
                "Natural-language description of the pattern — ~40 words. "
                "Describe what the telemetry / consultation looks like "
                "when this pattern shows up."
            ),
        },
        "verdict_or_action": {
            "type": "string",
            "description": (
                "The conclusion attached to this pattern. For maestro: "
                "L1_observe / L2_suggest / L3_bounded_auto / L4_human_only. "
                "For specialists: real_signal / noise / inconclusive."
            ),
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "reasoning": {
            "type": "string",
            "description": (
                "2-4 sentence lesson — why this pattern justifies the "
                "verdict. Phrase as reusable knowledge, not an event log."
            ),
        },
        "example_dec_id": {
            "type": "string",
            "description": "Reference to a decision log id (e.g. 'dec_abc123').",
        },
        "increment_if_exists": {
            "type": "boolean",
            "description": (
                "If True and the pattern_name already exists in the file, "
                "increment its occurrences counter, bump Last seen to now, "
                "and rolling-average the confidence with the new value. "
                "If False, skip silently when the name already exists."
            ),
        },
    },
    "required": [
        "target_file",
        "pattern_name",
        "signature",
        "verdict_or_action",
        "confidence",
        "reasoning",
        "example_dec_id",
        "increment_if_exists",
    ],
}


# ---------------------------------------------------------------------------
# Data model for an in-memory pattern
# ---------------------------------------------------------------------------


@dataclass
class MemoryPattern:
    """In-memory representation of one `## Pattern: …` block."""

    name: str
    first_seen: datetime
    last_seen: datetime
    occurrences: int
    signature: str
    verdict_or_action: str
    confidence: float
    reasoning: str
    example_dec_id: str
    extra_examples: list[str] = field(default_factory=list)

    def render(self) -> str:
        extras = ""
        if self.extra_examples:
            extras = (
                "- Additional examples: "
                + ", ".join(self.extra_examples[-5:])  # keep last 5
                + "\n"
            )
        return (
            f"## Pattern: {self.name}\n"
            f"- First seen: {self.first_seen.isoformat()}\n"
            f"- Last seen: {self.last_seen.isoformat()}\n"
            f"- Occurrences: {self.occurrences}\n"
            f"- Signature: {self.signature}\n"
            f"- Learned verdict/action: {self.verdict_or_action}\n"
            f"- Confidence: {self.confidence:.2f}\n"
            f"- Reasoning: {self.reasoning}\n"
            f"- Example reference: {self.example_dec_id}\n"
            + extras
        )


# ---------------------------------------------------------------------------
# Parsing + rendering
# ---------------------------------------------------------------------------


def _parse_memory_file(path: Path) -> tuple[str, list[MemoryPattern]]:
    """Split a memory markdown file into (header, patterns).

    Header is everything up to (but not including) the first
    `## Pattern:` heading. Patterns are parsed one by one.
    """
    text = path.read_text() if path.exists() else ""
    header = text
    patterns: list[MemoryPattern] = []

    match = _PATTERN_HEADING_RE.search(text)
    if not match:
        return header, patterns

    header = text[: match.start()]
    # Split body on `## Pattern: ` (keep the names in a parallel list)
    body = text[match.start() :]
    parts = re.split(r"^## Pattern: ", body, flags=re.MULTILINE)
    # parts[0] is empty (text before the first split marker)
    for raw in parts[1:]:
        if not raw.strip():
            continue
        # First line is the name
        first_nl = raw.find("\n")
        if first_nl < 0:
            continue
        name = raw[:first_nl].strip()
        body_block = raw[first_nl + 1 :]
        patterns.append(_parse_pattern_block(name, body_block))
    return header, patterns


def _parse_pattern_block(name: str, body: str) -> MemoryPattern:
    """Parse a single `## Pattern: <name>` body (the text after the heading)."""

    def _field(key: str, default: str = "") -> str:
        m = re.search(rf"^- {re.escape(key)}:\s*(.*)$", body, re.MULTILINE)
        return m.group(1).strip() if m else default

    first = _field("First seen")
    last = _field("Last seen")
    occ_raw = _field("Occurrences", "1")
    try:
        occ = int(occ_raw)
    except ValueError:
        occ = 1
    try:
        conf = float(_field("Confidence", "0.5"))
    except ValueError:
        conf = 0.5

    first_dt = _safe_iso(first)
    last_dt = _safe_iso(last) or first_dt
    first_dt = first_dt or datetime.now(tz=timezone.utc)
    last_dt = last_dt or first_dt

    # "Learned verdict/action" or legacy "Learned verdict" / "Learned action"
    verdict = (
        _field("Learned verdict/action")
        or _field("Learned verdict")
        or _field("Learned action")
        or ""
    )

    return MemoryPattern(
        name=name,
        first_seen=first_dt,
        last_seen=last_dt,
        occurrences=occ,
        signature=_field("Signature"),
        verdict_or_action=verdict,
        confidence=conf,
        reasoning=_field("Reasoning"),
        example_dec_id=_field("Example reference"),
    )


def _safe_iso(s: str) -> datetime | None:
    if not s:
        return None
    try:
        s = s.replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _render_file(header: str, patterns: list[MemoryPattern]) -> str:
    """Rebuild the full file content from header + ordered patterns."""
    header = header.rstrip() + "\n\n" if header.strip() else ""
    if not patterns:
        return header.rstrip() + "\n"
    blocks = "\n".join(p.render() for p in patterns)
    return header + blocks


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tf:
        tf.write(content)
        tmp_path = tf.name
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# Eviction
# ---------------------------------------------------------------------------


def _evict_lru(patterns: list[MemoryPattern]) -> MemoryPattern:
    """Remove one pattern in-place and return it. Evicts the pattern with
    the fewest occurrences; ties break on oldest `last_seen`."""
    assert patterns, "cannot evict from empty list"
    victim_idx = 0
    for i, p in enumerate(patterns):
        cur = patterns[victim_idx]
        if (p.occurrences, p.last_seen) < (cur.occurrences, cur.last_seen):
            victim_idx = i
    victim = patterns.pop(victim_idx)
    return victim


# ---------------------------------------------------------------------------
# Public API — the tool implementation
# ---------------------------------------------------------------------------


def write_memory_pattern(
    target_file: str,
    pattern_name: str,
    signature: str,
    verdict_or_action: str,
    confidence: float,
    reasoning: str,
    example_dec_id: str,
    increment_if_exists: bool,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Persist (or increment) a pattern entry in a memory file.

    Returns a structured result dict with: action, target_file, pattern_name,
    occurrences_after, evicted_name (if LRU fired), file_bytes_after.
    """
    if target_file not in _TARGET_FILE_MAP:
        raise ValueError(
            f"target_file must be one of {list(_TARGET_FILE_MAP)}, got {target_file!r}"
        )

    root = repo_root or Path(__file__).resolve().parent.parent
    path = root / _TARGET_FILE_MAP[target_file]

    header, patterns = _parse_memory_file(path)
    now = datetime.now(tz=timezone.utc)

    # --- find existing pattern by name ---
    existing = next((p for p in patterns if p.name == pattern_name), None)
    evicted_name: str | None = None

    if existing is not None:
        if not increment_if_exists:
            _LOG.info(
                "tools.write_memory_pattern.skipped_existing",
                target=target_file,
                pattern=pattern_name,
            )
            return {
                "action": "skipped_existing",
                "target_file": target_file,
                "pattern_name": pattern_name,
                "occurrences_after": existing.occurrences,
                "evicted_name": None,
            }
        # Increment
        existing.occurrences += 1
        existing.last_seen = now
        # Rolling average confidence
        n = existing.occurrences
        existing.confidence = round(
            ((existing.confidence * (n - 1)) + confidence) / n, 4
        )
        # Accumulate examples (cap at 5 extras)
        if example_dec_id and example_dec_id != existing.example_dec_id:
            if example_dec_id not in existing.extra_examples:
                existing.extra_examples.append(example_dec_id)
        # Update signature/reasoning only if new content is substantially longer
        if len(signature) > len(existing.signature) * 1.2:
            existing.signature = signature
        if len(reasoning) > len(existing.reasoning) * 1.2:
            existing.reasoning = reasoning
        existing.verdict_or_action = verdict_or_action
        action = "incremented"
    else:
        new_pattern = MemoryPattern(
            name=pattern_name,
            first_seen=now,
            last_seen=now,
            occurrences=1,
            signature=signature,
            verdict_or_action=verdict_or_action,
            confidence=round(confidence, 4),
            reasoning=reasoning,
            example_dec_id=example_dec_id,
        )
        patterns.append(new_pattern)
        # LRU cap
        while len(patterns) > MEMORY_CAP_PATTERNS:
            victim = _evict_lru(patterns)
            evicted_name = victim.name
            _LOG.info(
                "tools.write_memory_pattern.evicted",
                target=target_file,
                victim=victim.name,
                victim_occurrences=victim.occurrences,
                victim_last_seen=victim.last_seen.isoformat(),
            )
        action = "written"

    # --- atomic write ---
    content = _render_file(header, patterns)
    _atomic_write(path, content)
    file_bytes = len(content.encode("utf-8"))

    occ_after = (existing.occurrences if existing else 1)
    _LOG.info(
        "tools.write_memory_pattern.ok",
        action=action,
        target=target_file,
        pattern=pattern_name,
        occurrences_after=occ_after,
        evicted=evicted_name,
    )
    return {
        "action": action,
        "target_file": target_file,
        "pattern_name": pattern_name,
        "occurrences_after": occ_after,
        "evicted_name": evicted_name,
        "file_bytes_after": file_bytes,
    }


# ---------------------------------------------------------------------------
# Read helpers (for specialists)
# ---------------------------------------------------------------------------


def load_memory_file(target: str, repo_root: Path | None = None) -> str:
    """Return the full content of a memory file as a string (for system prompt)."""
    if target not in _TARGET_FILE_MAP:
        raise ValueError(f"target must be one of {list(_TARGET_FILE_MAP)}")
    root = repo_root or Path(__file__).resolve().parent.parent
    path = root / _TARGET_FILE_MAP[target]
    if not path.exists():
        return ""
    return path.read_text()


def list_patterns(target: str, repo_root: Path | None = None) -> list[MemoryPattern]:
    """Return parsed pattern list for a memory file."""
    if target not in _TARGET_FILE_MAP:
        raise ValueError(f"target must be one of {list(_TARGET_FILE_MAP)}")
    root = repo_root or Path(__file__).resolve().parent.parent
    path = root / _TARGET_FILE_MAP[target]
    _, patterns = _parse_memory_file(path)
    return patterns


__all__ = [
    "MEMORY_CAP_PATTERNS",
    "MemoryPattern",
    "WRITE_MEMORY_PATTERN_SCHEMA",
    "list_patterns",
    "load_memory_file",
    "write_memory_pattern",
]
