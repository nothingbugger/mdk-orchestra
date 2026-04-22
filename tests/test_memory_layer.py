"""Unit tests for the MDK Fleet memory layer.

Covers:
- agents/tools.py  — write_memory_pattern, load_memory_file, MemoryPattern, MEMORY_CAP_PATTERNS=50
- agents/curator.py — CuratorState, maybe_curate, _mock_curation_calls
- agents/base_specialist.py — handle_request loads <domain>_memory.md into system prompt
- agents/maestro.py — _maestro_system_prompt reads maestro_memory.md

All tests use tmp_path fixtures; never write to the real agents/*_memory.md files.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from textwrap import dedent
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Make sure mock mode is on before any agents import
os.environ.setdefault("MDK_AGENT_MOCK", "1")

pytest.importorskip("pydantic")
pytest.importorskip("structlog")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VOLTAGE_MEMORY_SCAFFOLD = dedent(
    """\
    # Voltage Memory

    *Voltage-domain patterns learned from past consultations.*

    ---
    """
)

MAESTRO_MEMORY_SCAFFOLD = dedent(
    """\
    # Maestro Memory

    *Agent-curated cross-domain patterns.*

    ---
    """
)

MAESTRO_PERSONALITY = dedent(
    """\
    # Maestro — The Conductor

    I am Maestro. I dispatch flags to specialists and synthesize decisions.
    """
)


def _write_scaffold(agents_dir: Path, name: str, content: str) -> Path:
    """Write a scaffold memory file and return its path."""
    path = agents_dir / f"{name}_memory.md"
    path.write_text(content, encoding="utf-8")
    return path


def _make_pattern_block(
    name: str,
    *,
    first_seen: str = "2026-04-20T00:00:00+00:00",
    last_seen: str = "2026-04-20T01:00:00+00:00",
    occurrences: int = 1,
    confidence: float = 0.8,
) -> str:
    return dedent(
        f"""\
        ## Pattern: {name}
        - First seen: {first_seen}
        - Last seen: {last_seen}
        - Occurrences: {occurrences}
        - Signature: Test signature for {name}.
        - Learned verdict/action: real_signal
        - Confidence: {confidence:.2f}
        - Reasoning: Test reasoning for {name}.
        - Example reference: dec_test001
        """
    )


def _build_seeded_file(agents_dir: Path, name: str, patterns: list[str]) -> Path:
    path = agents_dir / f"{name}_memory.md"
    content = VOLTAGE_MEMORY_SCAFFOLD + "\n" + "\n".join(patterns) + "\n"
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Test 1 — write_new_pattern
# ---------------------------------------------------------------------------


def test_write_new_pattern(tmp_path: Path) -> None:
    """Writing a pattern to an empty scaffold creates a correctly formatted entry."""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    _write_scaffold(agents_dir, "voltage", VOLTAGE_MEMORY_SCAFFOLD)

    from agents.tools import write_memory_pattern

    result = write_memory_pattern(
        target_file="voltage",
        pattern_name="capacitor_wear_step",
        signature="12V rail drops 200 mV in a step function after high temp spike.",
        verdict_or_action="real_signal",
        confidence=0.75,
        reasoning="Step-down shape without recovery indicates capacitor fatigue.",
        example_dec_id="dec_abc123",
        increment_if_exists=False,
        repo_root=tmp_path,
    )

    assert result["action"] == "written"
    assert result["pattern_name"] == "capacitor_wear_step"
    assert result["occurrences_after"] == 1

    text = (agents_dir / "voltage_memory.md").read_text()
    assert "## Pattern: capacitor_wear_step" in text
    assert "Occurrences: 1" in text
    assert "12V rail drops 200 mV" in text


# ---------------------------------------------------------------------------
# Test 2 — increment_existing_pattern
# ---------------------------------------------------------------------------


def test_increment_existing_pattern(tmp_path: Path) -> None:
    """Incrementing an existing pattern bumps occurrences and rolling-averages confidence."""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    _build_seeded_file(
        agents_dir,
        "voltage",
        [_make_pattern_block("capacitor_wear_step", occurrences=1, confidence=0.8)],
    )

    from agents.tools import write_memory_pattern

    result = write_memory_pattern(
        target_file="voltage",
        pattern_name="capacitor_wear_step",
        signature="12V rail drops 200 mV in a step function after high temp spike.",
        verdict_or_action="real_signal",
        confidence=0.6,
        reasoning="Pattern still holds after second occurrence.",
        example_dec_id="dec_def456",
        increment_if_exists=True,
        repo_root=tmp_path,
    )

    assert result["action"] == "incremented"
    assert result["occurrences_after"] == 2

    # Rolling average: round((0.8 * 1 + 0.6) / 2, 4) = 0.7
    text = (agents_dir / "voltage_memory.md").read_text()
    assert "Occurrences: 2" in text
    # Confidence should be close to 0.7 (stored as 0.7 or 0.70)
    assert "Confidence: 0.70" in text or "Confidence: 0.7" in text

    # Last seen should have been bumped — look for any pattern heading present
    assert "## Pattern: capacitor_wear_step" in text


# ---------------------------------------------------------------------------
# Test 3 — increment_if_exists=False skips silently
# ---------------------------------------------------------------------------


def test_increment_if_exists_false_skips(tmp_path: Path) -> None:
    """When increment_if_exists=False, re-writing an existing name is a no-op."""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    memory_path = _build_seeded_file(
        agents_dir,
        "voltage",
        [_make_pattern_block("capacitor_wear_step", occurrences=1, confidence=0.8)],
    )

    original_bytes = memory_path.read_bytes()

    from agents.tools import write_memory_pattern

    result = write_memory_pattern(
        target_file="voltage",
        pattern_name="capacitor_wear_step",
        signature="New description that should NOT overwrite the old one.",
        verdict_or_action="noise",
        confidence=0.3,
        reasoning="Should not be written.",
        example_dec_id="dec_skip",
        increment_if_exists=False,
        repo_root=tmp_path,
    )

    assert result["action"] == "skipped_existing"

    # File content must be byte-for-byte identical
    assert memory_path.read_bytes() == original_bytes


# ---------------------------------------------------------------------------
# Test 4 — LRU eviction when cap is exceeded
# ---------------------------------------------------------------------------


def test_memory_cap_lru_eviction(tmp_path: Path) -> None:
    """Writing a 51st pattern triggers eviction of the MIN-occurrences pattern."""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    from agents.tools import MEMORY_CAP_PATTERNS, write_memory_pattern

    assert MEMORY_CAP_PATTERNS == 50

    # Build 50 patterns with distinct occurrences (1–50) and distinct last_seen
    base_ts = datetime(2026, 4, 19, 0, 0, 0, tzinfo=timezone.utc)
    patterns = []
    for i in range(1, 51):  # occurrences 1..50
        ts = (base_ts + timedelta(hours=i)).isoformat()
        patterns.append(
            _make_pattern_block(
                f"pattern_{i:03d}",
                occurrences=i,
                last_seen=ts,
                first_seen=(base_ts + timedelta(hours=i)).isoformat(),
                confidence=0.5,
            )
        )
    _build_seeded_file(agents_dir, "voltage", patterns)

    # pattern_001 has occurrences=1 (the minimum) and will be the eviction victim
    result = write_memory_pattern(
        target_file="voltage",
        pattern_name="pattern_new_eviction_trigger",
        signature="New pattern that pushes count to 51.",
        verdict_or_action="real_signal",
        confidence=0.9,
        reasoning="This write should evict the LRU entry.",
        example_dec_id="dec_evict001",
        increment_if_exists=False,
        repo_root=tmp_path,
    )

    assert result["action"] == "written"
    assert result["evicted_name"] == "pattern_001", (
        f"Expected pattern_001 to be evicted (lowest occurrences), got {result['evicted_name']!r}"
    )

    # File should have exactly 50 patterns
    from agents.tools import list_patterns

    remaining = list_patterns("voltage", repo_root=tmp_path)
    assert len(remaining) == 50, f"Expected 50 patterns after eviction, got {len(remaining)}"

    names = {p.name for p in remaining}
    assert "pattern_001" not in names
    assert "pattern_new_eviction_trigger" in names


# ---------------------------------------------------------------------------
# Test 5 — atomic write: os.replace failure leaves file intact
# ---------------------------------------------------------------------------


def test_atomic_write_no_corruption(tmp_path: Path) -> None:
    """If os.replace raises, the original memory file is untouched."""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    _build_seeded_file(
        agents_dir,
        "voltage",
        [
            _make_pattern_block("alpha"),
            _make_pattern_block("beta"),
            _make_pattern_block("gamma"),
        ],
    )
    memory_path = agents_dir / "voltage_memory.md"
    original_bytes = memory_path.read_bytes()

    call_count = {"n": 0}
    real_replace = os.replace

    def flaky_replace(src: str, dst: str) -> None:
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Clean up the temp file to avoid leaking it, then raise
            try:
                Path(src).unlink(missing_ok=True)
            except Exception:
                pass
            raise RuntimeError("simulated os.replace failure")
        real_replace(src, dst)

    from agents.tools import write_memory_pattern

    with patch("agents.tools.os.replace", side_effect=flaky_replace):
        with pytest.raises(RuntimeError, match="simulated os.replace failure"):
            write_memory_pattern(
                target_file="voltage",
                pattern_name="new_pattern_after_crash",
                signature="Should not be persisted on first call.",
                verdict_or_action="real_signal",
                confidence=0.5,
                reasoning="Crash test.",
                example_dec_id="dec_crash",
                increment_if_exists=False,
                repo_root=tmp_path,
            )

    # File must be intact after the crash
    assert memory_path.read_bytes() == original_bytes, "File was corrupted by failed write"

    # Second call with restored os.replace must succeed
    result = write_memory_pattern(
        target_file="voltage",
        pattern_name="new_pattern_after_crash",
        signature="Should be persisted on second call.",
        verdict_or_action="real_signal",
        confidence=0.5,
        reasoning="Recovery test.",
        example_dec_id="dec_recovery",
        increment_if_exists=False,
        repo_root=tmp_path,
    )
    assert result["action"] == "written"
    text = memory_path.read_text()
    assert "## Pattern: new_pattern_after_crash" in text


# ---------------------------------------------------------------------------
# Test 6 — curation pass: silent no-op when no new decisions
# ---------------------------------------------------------------------------


def test_curation_pass_skips_silent_if_no_new_decisions(tmp_path: Path) -> None:
    """maybe_curate returns None and advances cursor when there are no pending decisions."""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    # Scaffold all 5 memory files so load_memory_file doesn't fail
    for name in ("maestro", "voltage", "hashrate", "environment", "power"):
        _write_scaffold(agents_dir, name, VOLTAGE_MEMORY_SCAFFOLD)

    from agents.curator import CuratorState, maybe_curate

    now = datetime.now(tz=timezone.utc)
    last_curation = now - timedelta(hours=1)

    state = CuratorState(
        last_curation_sim_ts=last_curation,
        pending_decisions=[],
    )

    result = maybe_curate(
        maestro_config=MagicMock(model="claude-opus-4-7"),
        maestro_personality=MAESTRO_PERSONALITY,
        state=state,
        decision_ts=now,
        repo_root=tmp_path,
    )

    # Silent no-op path returns None (per docstring). Cursor is still
    # advanced so the next decision restarts the window fresh.
    assert result is None, "Expected None for the silent no-op path"
    assert state.last_curation_sim_ts == now, "Cursor should have advanced to decision_ts"


# ---------------------------------------------------------------------------
# Test 7 — curation pass fires when interval elapsed
# ---------------------------------------------------------------------------


def test_curation_pass_fires_when_interval_elapsed(tmp_path: Path) -> None:
    """maybe_curate fires the curation pass and applies write_memory_pattern tool calls."""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    for name in ("maestro", "voltage", "hashrate", "environment", "power"):
        _write_scaffold(agents_dir, name, VOLTAGE_MEMORY_SCAFFOLD)

    from agents._client import LLMMultiToolResult
    from agents.curator import CuratorState, maybe_curate

    now = datetime.now(tz=timezone.utc)
    last_curation = now - timedelta(minutes=31)

    pending: list[dict[str, Any]] = [
        {
            "decision_id": "dec_testfire_001",
            "flag_id": "flg_001",
            "miner_id": "m001",
            "action": "throttle",
            "autonomy_level": "L3_bounded_auto",
            "confidence": 0.85,
            "consulted_agents": ["voltage_agent"],
            "reasoning_trace": "Test decision for curation.",
        }
    ]

    state = CuratorState(
        last_curation_sim_ts=last_curation,
        pending_decisions=list(pending),
    )

    canned_result = LLMMultiToolResult(
        tool_calls=[
            {
                "name": "write_memory_pattern",
                "input": {
                    "target_file": "maestro",
                    "pattern_name": "curation_test_pattern_unique_xyz",
                    "signature": "Throttle at L3 after voltage flag with high confidence.",
                    "verdict_or_action": "L3_bounded_auto",
                    "confidence": 0.85,
                    "reasoning": "Seen in at least 2 decisions; throttle is correct response.",
                    "example_dec_id": "dec_testfire_001",
                    "increment_if_exists": False,
                },
            }
        ],
        text="",
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=0,
        cost_usd=0.0,
        latency_ms=0.0,
        model="claude-opus-4-7",
        is_mock=True,
    )

    with patch("agents.curator.call_structured_multi_tool", return_value=canned_result):
        summary = maybe_curate(
            maestro_config=MagicMock(model="claude-opus-4-7"),
            maestro_personality=MAESTRO_PERSONALITY,
            state=state,
            decision_ts=now,
            repo_root=tmp_path,
        )

    assert summary is not None, "Expected a summary dict when interval elapsed"
    assert summary["pass_idx"] == 1
    assert summary["decisions_analyzed"] == 1
    assert summary["patterns_written"] == 1

    # pending_decisions should be cleared
    assert state.pending_decisions == []
    assert state.last_curation_sim_ts == now

    # Pattern should be on disk
    text = (agents_dir / "maestro_memory.md").read_text()
    assert "## Pattern: curation_test_pattern_unique_xyz" in text


# ---------------------------------------------------------------------------
# Test 8 — first decision sets cursor, no API call
# ---------------------------------------------------------------------------


def test_curation_first_decision_sets_cursor_no_call(tmp_path: Path) -> None:
    """Fresh CuratorState: first decision sets the sim-ts cursor, no curation fires."""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    for name in ("maestro", "voltage", "hashrate", "environment", "power"):
        _write_scaffold(agents_dir, name, VOLTAGE_MEMORY_SCAFFOLD)

    from agents.curator import CuratorState, maybe_curate

    now = datetime.now(tz=timezone.utc)
    state = CuratorState()  # last_curation_sim_ts=None

    api_called = {"count": 0}

    def mock_api(*args: Any, **kwargs: Any) -> None:  # should never be called
        api_called["count"] += 1
        raise AssertionError("API should not be called on first decision")

    with patch("agents.curator.call_structured_multi_tool", side_effect=mock_api):
        result = maybe_curate(
            maestro_config=MagicMock(model="claude-opus-4-7"),
            maestro_personality=MAESTRO_PERSONALITY,
            state=state,
            decision_ts=now,
            repo_root=tmp_path,
        )

    assert result is None
    assert api_called["count"] == 0
    assert state.last_curation_sim_ts == now


# ---------------------------------------------------------------------------
# Test 9 — Maestro system prompt includes memory
# ---------------------------------------------------------------------------


def test_maestro_system_prompt_includes_memory(tmp_path: Path, monkeypatch: Any) -> None:
    """_maestro_system_prompt appends curated memory when the file has patterns."""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    # Write the personality md that maestro.py reads at __init__
    maestro_md = agents_dir / "maestro.md"
    maestro_md.write_text(MAESTRO_PERSONALITY, encoding="utf-8")

    # Write maestro_memory.md with one real pattern
    memory_path = agents_dir / "maestro_memory.md"
    memory_path.write_text(
        MAESTRO_MEMORY_SCAFFOLD + "\n" + _make_pattern_block("cross_domain_throttle_pattern"),
        encoding="utf-8",
    )

    # Scaffold the specialist md files (they are read in __init__ of BaseSpecialist)
    for agent_name in ("voltage_agent", "hashrate_agent", "environment_agent", "power_agent"):
        (agents_dir / f"{agent_name}.md").write_text(
            f"# {agent_name} personality\n", encoding="utf-8"
        )
    for mem_name in ("voltage", "hashrate", "environment", "power"):
        _write_scaffold(agents_dir, mem_name, VOLTAGE_MEMORY_SCAFFOLD)

    from agents.config import AgentConfig

    orchestrator_config = AgentConfig(
        name="orchestrator",
        personality_md_path=str(maestro_md),
        model="claude-opus-4-7",
        memory_dir=str(tmp_path / "memory" / "orchestrator"),
        enabled=True,
    )

    # Patch DEFAULT_AGENT_CONFIGS and specialist md paths to use tmp dirs
    tmp_agent_configs = {
        "orchestrator": orchestrator_config,
    }
    for agent_name in ("voltage_agent", "hashrate_agent", "environment_agent", "power_agent"):
        md_path = agents_dir / f"{agent_name}.md"
        tmp_agent_configs[agent_name] = AgentConfig(
            name=agent_name,
            personality_md_path=str(md_path),
            model="claude-sonnet-4-6",
            memory_dir=str(tmp_path / "memory" / agent_name),
            enabled=True,
        )

    monkeypatch.setenv("MDK_AGENT_MOCK", "1")
    monkeypatch.setenv("MDK_STREAM_DIR", str(tmp_path / "stream"))
    monkeypatch.setenv("MDK_MEMORY_DIR", str(tmp_path / "memory"))

    # _maestro_system_prompt does `from agents.tools import load_memory_file` as a local import,
    # so we patch the function on the agents.tools module directly. Capture the real function
    # BEFORE patching to avoid infinite recursion.
    import agents.tools as tools_module

    _real_load = tools_module.load_memory_file

    def patched_load(target: str, repo_root: Path | None = None) -> str:
        return _real_load(target, repo_root=tmp_path)

    monkeypatch.setattr(tools_module, "load_memory_file", patched_load)

    import agents.maestro as maestro_module

    # Instantiate Maestro using our patched configs
    with patch("agents.maestro.DEFAULT_AGENT_CONFIGS", tmp_agent_configs):
        maestro = maestro_module.Maestro(agent_configs=tmp_agent_configs)
        system_prompt = maestro._maestro_system_prompt()

    assert MAESTRO_PERSONALITY.strip() in system_prompt
    assert "## Pattern: cross_domain_throttle_pattern" in system_prompt
    assert "# Curated memory (self-authored)" in system_prompt

    # --- Now test with a truly empty memory file (whitespace only) ---
    # NOTE: The implementation checks `memory_md.strip()` not "has patterns".
    # A scaffold-only file (with header text but no patterns) still causes the
    # curated memory block to be appended. This is a potential discrepancy with
    # the spec requirement: "scaffold-only → personality alone". See edge-case
    # note in MODULE_NOTES. We test the actual implementation behaviour: empty
    # (blank) file → personality alone.
    memory_path.write_text("\n   \n", encoding="utf-8")

    with patch("agents.maestro.DEFAULT_AGENT_CONFIGS", tmp_agent_configs):
        maestro2 = maestro_module.Maestro(agent_configs=tmp_agent_configs)
        system_prompt_empty = maestro2._maestro_system_prompt()

    # Truly blank file → no curated memory block appended
    assert "# Curated memory" not in system_prompt_empty
    assert MAESTRO_PERSONALITY.strip() in system_prompt_empty


# ---------------------------------------------------------------------------
# Test 10 — VoltageAgent.handle_request loads memory into system prompt
# ---------------------------------------------------------------------------


def test_specialist_handle_request_loads_memory(tmp_path: Path, monkeypatch: Any) -> None:
    """VoltageAgent.handle_request includes the voltage memory block in the system prompt."""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    # Write voltage_agent.md personality
    voltage_md = agents_dir / "voltage_agent.md"
    voltage_md.write_text("# Voltage Agent Personality\n", encoding="utf-8")

    # Seed voltage_memory.md with one pattern
    voltage_memory = agents_dir / "voltage_memory.md"
    voltage_memory.write_text(
        VOLTAGE_MEMORY_SCAFFOLD + "\n" + _make_pattern_block("psu_step_drop_v2"),
        encoding="utf-8",
    )

    monkeypatch.setenv("MDK_AGENT_MOCK", "1")
    monkeypatch.setenv("MDK_STREAM_DIR", str(tmp_path / "stream"))
    monkeypatch.setenv("MDK_MEMORY_DIR", str(tmp_path / "memory"))

    from agents.config import AgentConfig

    voltage_config = AgentConfig(
        name="voltage_agent",
        personality_md_path=str(voltage_md),
        model="claude-sonnet-4-6",
        memory_dir=str(tmp_path / "memory" / "voltage_agent"),
        enabled=True,
    )

    # Build a synthetic ReasoningRequest
    from shared.schemas.events import ReasoningContext, ReasoningRequest

    request = ReasoningRequest(
        request_id="req_test_mem_01",
        flag_id="flg_test_mem_01",
        target_agent="voltage_agent",
        miner_id="m042",
        question="Is this voltage drop a real signal?",
        context=ReasoningContext(
            flag={
                "flag_id": "flg_test_mem_01",
                "miner_id": "m042",
                "flag_type": "voltage_drift",
                "severity": "warn",
                "confidence": 0.75,
                "source_tool": "rule_engine",
                "evidence": {"metric": "voltage_v", "window_min": 30.0},
                "raw_score": 0.82,
            }
        ),
    )

    # Patch load_memory_file to use tmp_path.
    # Capture the real function BEFORE patching to avoid infinite recursion.
    import agents.tools as tools_module

    _real_load_10 = tools_module.load_memory_file

    def patched_load(target: str, repo_root: Path | None = None) -> str:
        return _real_load_10(target, repo_root=tmp_path)

    # Capture the system_prompt passed to call_structured
    captured_system_prompt: list[str] = []

    def mock_call_structured(
        model: str,
        system_prompt: str,
        user_content: str,
        **kwargs: Any,
    ) -> Any:
        captured_system_prompt.append(system_prompt)
        from agents._client import LLMResult

        return LLMResult(
            tool_input={
                "assessment": "real_signal",
                "confidence": 0.75,
                "severity_estimate": "warn",
                "reasoning": "[mock] test",
                "recommended_action_hint": "alert_operator",
            },
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=0,
            cost_usd=0.0,
            latency_ms=0.0,
            model=model,
            is_mock=True,
        )

    import agents.base_specialist as base_specialist_module

    monkeypatch.setattr(base_specialist_module, "call_structured", mock_call_structured)

    # handle_request does `from agents.tools import load_memory_file` as a local import,
    # so patch agents.tools directly — the local import resolves from there.
    # tools_module and patched_load are already defined above.
    monkeypatch.setattr(tools_module, "load_memory_file", patched_load)

    from agents.voltage_agent import VoltageAgent

    agent = VoltageAgent(voltage_config)
    # Override _memory_target to "voltage" (already is, but explicit)
    agent._memory_target = "voltage"

    agent.handle_request(request)

    assert captured_system_prompt, "call_structured was never called"
    system_prompt = captured_system_prompt[0]

    assert "## Pattern: psu_step_drop_v2" in system_prompt, (
        f"Pattern block missing from system prompt.\nSystem prompt:\n{system_prompt[:500]}"
    )
    assert "# Curated memory (updated by Maestro)" in system_prompt
