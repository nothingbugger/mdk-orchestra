"""Anthropic API wrapper for the agent fleet.

Two concerns live here:

1. A thin convenience around `anthropic.Anthropic()` that adds prompt
   caching on the personality `.md` (the system prompt rarely changes, so
   the cache hit rate is effectively 100% after the first call) and forces
   structured output via tool-use.
2. A **mock mode** that short-circuits the network call and returns a
   deterministic canned response synthesized from the flag/context. Mock
   mode is triggered by `MDK_AGENT_MOCK=1` or by the absence of
   `ANTHROPIC_API_KEY` — both conditions mean "run end-to-end without
   burning credits." This is what lets the smoke test pass without an API
   key set up.

Nothing here parses Pydantic — that happens in `base_specialist.py` and
`maestro.py`. This module speaks dicts.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

try:
    from anthropic import Anthropic  # type: ignore[import-not-found]
except Exception:  # pragma: no cover — anthropic optional at import time
    Anthropic = None  # type: ignore[assignment,misc]

from agents.config import estimate_cost_usd, mock_mode_enabled


@dataclass(frozen=True)
class LLMResult:
    """Output of a single forced-tool-use call."""

    tool_input: dict[str, Any]
    """The structured payload the model produced via the forced tool."""

    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cost_usd: float
    latency_ms: float
    model: str
    is_mock: bool


@dataclass(frozen=True)
class LLMMultiToolResult:
    """Output of a `tool_choice=auto` call that may emit zero, one, or many
    tool uses plus free-form text."""

    tool_calls: list[dict[str, Any]]
    """List of {'name': ..., 'input': {...}} entries — one per tool_use block."""

    text: str
    """Free-form text the model produced alongside (or instead of) tool uses."""

    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cost_usd: float
    latency_ms: float
    model: str
    is_mock: bool


_SHARED_CLIENT: Any | None = None


def get_client() -> Any:
    """Lazy singleton Anthropic client. Returns None in mock mode."""
    global _SHARED_CLIENT
    if mock_mode_enabled():
        return None
    if _SHARED_CLIENT is None:
        if Anthropic is None:
            raise RuntimeError(
                "anthropic SDK is not installed. Install deps or set MDK_AGENT_MOCK=1."
            )
        _SHARED_CLIENT = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _SHARED_CLIENT


def call_structured(
    model: str,
    system_prompt: str,
    user_content: str,
    tool_name: str,
    tool_description: str,
    tool_schema: dict[str, Any],
    max_tokens: int = 1024,
    mock_fallback: dict[str, Any] | None = None,
    agent_slot: str | None = None,
) -> LLMResult:
    """Call a backend with forced tool-use and return structured output.

    With the backend abstraction (`agents/llm_backend.py`) this function
    now routes through `get_backend(agent_slot)` when `agent_slot` is
    provided. That means the same call returns a Sonnet response on
    the canonical default, an Ollama/Qwen response when the routing
    profile selects `ollama` for that slot, etc. The `model` positional
    argument is overridden by the routing unless the caller ALSO omits
    `agent_slot` — in that case we fall back to the legacy path
    (Anthropic-direct with the given `model`), preserving zero-regression
    behavior for callers that haven't opted in yet.
    """
    # Backend routing path
    if agent_slot is not None:
        from agents.llm_backend import get_backend  # local to avoid cycle

        backend, routed_model = get_backend(agent_slot)
        result = backend.call(
            model=routed_model,
            system_prompt=system_prompt,
            user_content=user_content,
            tools=[
                {
                    "name": tool_name,
                    "description": tool_description,
                    "input_schema": tool_schema,
                }
            ],
            tool_choice={"type": "tool", "name": tool_name},
            max_tokens=max_tokens,
            mock_fallback=mock_fallback,
        )
        # Convert the backend's `tool_calls` list into the legacy
        # `tool_input` single-dict shape expected by existing callers.
        tool_input: dict[str, Any] = {}
        for call in result.tool_calls:
            if call.get("name") == tool_name:
                tool_input = dict(call.get("input") or {})
                break
        return LLMResult(
            tool_input=tool_input,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cache_read_tokens=result.cache_read_tokens,
            cost_usd=result.cost_usd,
            latency_ms=result.latency_ms,
            model=result.model_used,
            is_mock=result.is_mock,
        )

    # Legacy Anthropic-direct path (unchanged for callers not yet passing agent_slot).
    if mock_mode_enabled():
        return LLMResult(
            tool_input=mock_fallback or {},
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=0,
            cost_usd=0.0,
            latency_ms=0.0,
            model=model,
            is_mock=True,
        )

    client = get_client()
    t0 = time.monotonic()
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_content}],
        tools=[
            {
                "name": tool_name,
                "description": tool_description,
                "input_schema": tool_schema,
            }
        ],
        tool_choice={"type": "tool", "name": tool_name},
    )
    latency_ms = (time.monotonic() - t0) * 1000.0

    tool_input: dict[str, Any] = {}
    for block in resp.content:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == tool_name:
            tool_input = dict(block.input)
            break

    usage = resp.usage
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cost = estimate_cost_usd(model, input_tokens, output_tokens, cache_read)

    return LLMResult(
        tool_input=tool_input,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read,
        cost_usd=cost,
        latency_ms=latency_ms,
        model=model,
        is_mock=False,
    )


def call_structured_multi_tool(
    model: str,
    system_prompt: str,
    user_content: str,
    tools: list[dict[str, Any]],
    max_tokens: int = 2048,
    mock_tool_calls: list[dict[str, Any]] | None = None,
    mock_text: str = "",
    agent_slot: str | None = None,
) -> LLMMultiToolResult:
    """Call the model with `tool_choice=auto` and return all tool_use blocks.

    Unlike `call_structured` (single forced tool), this supports the curator
    flow where Maestro may emit multiple `write_memory_pattern` tool calls
    in one response, or none at all.

    If `agent_slot` is provided, routing goes through `get_backend(slot)` —
    the `model` parameter is overridden by the YAML routing. Without it,
    the legacy Anthropic-direct path is used (zero regression).

    Args:
        tools: list of tool schemas (each with name/description/input_schema).
        mock_tool_calls: in mock mode, return these as `tool_calls`. Each entry
            is `{"name": <tool_name>, "input": <dict>}`.
        mock_text: in mock mode, return this as the `.text` field.
        agent_slot: optional routing slot (e.g. "maestro.curation").
    """
    # Backend-routed path.
    if agent_slot is not None:
        from agents.llm_backend import get_backend  # local to avoid cycle

        backend, routed_model = get_backend(agent_slot)
        result = backend.call(
            model=routed_model,
            system_prompt=system_prompt,
            user_content=user_content,
            tools=tools,
            tool_choice={"type": "auto"},
            max_tokens=max_tokens,
            mock_fallback=mock_tool_calls,
        )
        return LLMMultiToolResult(
            tool_calls=list(result.tool_calls),
            text=(result.content or mock_text) if result.is_mock else (result.content or ""),
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cache_read_tokens=result.cache_read_tokens,
            cost_usd=result.cost_usd,
            latency_ms=result.latency_ms,
            model=result.model_used,
            is_mock=result.is_mock,
        )

    # Legacy Anthropic-direct path.
    if mock_mode_enabled():
        return LLMMultiToolResult(
            tool_calls=list(mock_tool_calls or []),
            text=mock_text,
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=0,
            cost_usd=0.0,
            latency_ms=0.0,
            model=model,
            is_mock=True,
        )

    client = get_client()
    t0 = time.monotonic()
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_content}],
        tools=tools,
        tool_choice={"type": "auto"},
    )
    latency_ms = (time.monotonic() - t0) * 1000.0

    tool_calls: list[dict[str, Any]] = []
    text_parts: list[str] = []
    for block in resp.content:
        btype = getattr(block, "type", None)
        if btype == "tool_use":
            tool_calls.append(
                {
                    "name": getattr(block, "name", ""),
                    "input": dict(getattr(block, "input", {}) or {}),
                }
            )
        elif btype == "text":
            text_parts.append(getattr(block, "text", "") or "")

    usage = resp.usage
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cost = estimate_cost_usd(model, input_tokens, output_tokens, cache_read)

    return LLMMultiToolResult(
        tool_calls=tool_calls,
        text="\n".join(text_parts),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read,
        cost_usd=cost,
        latency_ms=latency_ms,
        model=model,
        is_mock=False,
    )
