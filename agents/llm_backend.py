"""LLM backend abstraction — Anthropic + Ollama behind a single Protocol.

This module is the ONE place where the decision "which LLM answers this
agent call" is resolved. Every caller (Maestro dispatch, Opus second
opinion, curator, specialists) passes an `agent_slot` string; the
factory `get_backend(agent_slot)` returns a concrete backend per the
routing rules in `config/llm_routing.yaml` (plus env overrides).

Agent slots
-----------

  maestro.dispatch        — first-pass synthesis
  maestro.escalation      — Opus second opinion (L3/L4 path)
  maestro.curation        — every-30-sim-min memory curation
  specialists.voltage
  specialists.hashrate
  specialists.environment
  specialists.power

Precedence (highest first)
--------------------------

  1. Per-slot env override:
       MDK_LLM_<SECTION>_<SLOT>_BACKEND  and  _MODEL
       e.g. MDK_LLM_SPECIALISTS_VOLTAGE_BACKEND=ollama
  2. Profile override:
       MDK_LLM_PROFILE=hybrid_economic
  3. `agents` section in config/llm_routing.yaml
  4. `default` section in config/llm_routing.yaml

Contract returned by all backends
---------------------------------

A dict with:

    {
      "content": str | None,        # free-form text the model emitted
      "tool_calls": list[dict],     # [{"name": ..., "input": {...}}, ...]
      "cost_usd": float,            # Ollama always 0.0
      "latency_ms": float,
      "model_used": str,
      "backend_used": str,          # "anthropic" or "ollama"
      "input_tokens": int,
      "output_tokens": int,
      "cache_read_tokens": int,     # Anthropic only; Ollama=0
      "is_mock": bool,
    }

This uniform shape lets Maestro / specialists stay backend-agnostic:
they consume `tool_calls` for structured decisions regardless of which
model produced them.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import structlog

try:
    import yaml  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

try:
    from anthropic import Anthropic  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    Anthropic = None  # type: ignore[assignment,misc]

from agents.config import estimate_cost_usd, mock_mode_enabled

_LOG = structlog.get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "llm_routing.yaml"


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackendResult:
    """Uniform response from any backend."""

    content: str | None
    tool_calls: list[dict[str, Any]]
    cost_usd: float
    latency_ms: float
    model_used: str
    backend_used: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    is_mock: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "tool_calls": list(self.tool_calls),
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "model_used": self.model_used,
            "backend_used": self.backend_used,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "is_mock": self.is_mock,
        }


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


class LLMBackend(Protocol):
    """The single method every concrete backend must implement."""

    name: str

    def call(
        self,
        *,
        model: str,
        system_prompt: str,
        user_content: str,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        max_tokens: int = 1024,
        mock_fallback: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> BackendResult: ...


# ---------------------------------------------------------------------------
# Anthropic backend
# ---------------------------------------------------------------------------


_SHARED_ANTHROPIC_CLIENT: Any | None = None


def _get_anthropic_client() -> Any:
    global _SHARED_ANTHROPIC_CLIENT
    if _SHARED_ANTHROPIC_CLIENT is None:
        if Anthropic is None:
            raise RuntimeError("anthropic SDK not installed")
        _SHARED_ANTHROPIC_CLIENT = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _SHARED_ANTHROPIC_CLIENT


class AnthropicBackend:
    """Thin wrapper around `Anthropic().messages.create` with ephemeral
    cache on the system prompt. Preserves the original behavior of
    `agents/_client.py` — callers that don't set a routing profile keep
    getting identical results."""

    name: str = "anthropic"

    def call(
        self,
        *,
        model: str,
        system_prompt: str,
        user_content: str,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        max_tokens: int = 1024,
        mock_fallback: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> BackendResult:
        if mock_mode_enabled():
            return self._mock_result(
                model=model, mock_fallback=mock_fallback, tools=tools, tool_choice=tool_choice
            )

        client = _get_anthropic_client()

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": user_content}],
        }
        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice

        t0 = time.monotonic()
        resp = client.messages.create(**kwargs)
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

        return BackendResult(
            content="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            cost_usd=cost,
            latency_ms=latency_ms,
            model_used=model,
            backend_used="anthropic",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            is_mock=False,
        )

    def _mock_result(
        self,
        *,
        model: str,
        mock_fallback: dict[str, Any] | list[dict[str, Any]] | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
    ) -> BackendResult:
        # Preserve the old _client.py mock semantics: dict → single forced
        # tool input; list → list of tool calls; None → empty.
        tool_calls: list[dict[str, Any]] = []
        if isinstance(mock_fallback, dict):
            tool_name = None
            if isinstance(tool_choice, dict):
                tool_name = tool_choice.get("name")
            if tool_name is None and tools:
                tool_name = tools[0].get("name", "tool")
            tool_calls = [{"name": tool_name or "tool", "input": dict(mock_fallback)}]
        elif isinstance(mock_fallback, list):
            tool_calls = [dict(c) for c in mock_fallback]

        return BackendResult(
            content=None,
            tool_calls=tool_calls,
            cost_usd=0.0,
            latency_ms=0.0,
            model_used=model,
            backend_used="anthropic",
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=0,
            is_mock=True,
        )


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------


class StandardLocalBackend:
    """Backend for local LLM inference servers using the de-facto
    industry-standard HTTP API format (chat completions endpoint with
    messages/tools schema — what OpenAI's API established and what most
    local servers now speak).

    Compatible servers include: Ollama, LM Studio, llama.cpp server,
    vLLM, text-generation-webui, and anything else that implements
    `/v1/chat/completions` with tool-calling support.

    - Translates Anthropic-style tool schemas (`input_schema`) into the
      standard `function.parameters` shape on the wire.
    - Normalizes the standard `tool_calls` response back into the
      `{"name": ..., "input": ...}` shape Maestro expects.
    - Retries up to N times on malformed tool_call JSON (smaller local
      models occasionally return invalid JSON arguments).

    Configuration via `config/llm_routing.yaml`:

        backend: standard_local
        host: http://localhost:11434    # Ollama default; any compatible
                                        # server works (see docs/extending.md)
        model: qwen2.5:7b-instruct      # any model the server serves

    Cost is always reported as 0.0 (local inference).
    """

    name: str = "standard_local"

    def __init__(
        self,
        *,
        host: str,
        host_fallback: str | None = None,
        timeout_s: float = 120.0,
        tool_call_retries: int = 2,
    ) -> None:
        self.host = host.rstrip("/")
        self.host_fallback = host_fallback.rstrip("/") if host_fallback else None
        self.timeout_s = timeout_s
        self.tool_call_retries = max(1, tool_call_retries)

    def call(
        self,
        *,
        model: str,
        system_prompt: str,
        user_content: str,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        max_tokens: int = 1024,
        mock_fallback: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> BackendResult:
        if mock_mode_enabled():
            # Same mock shape as Anthropic for testability.
            tool_calls: list[dict[str, Any]] = []
            if isinstance(mock_fallback, dict):
                tn = None
                if isinstance(tool_choice, dict):
                    tn = tool_choice.get("name")
                if tn is None and tools:
                    tn = tools[0].get("name", "tool")
                tool_calls = [{"name": tn or "tool", "input": dict(mock_fallback)}]
            elif isinstance(mock_fallback, list):
                tool_calls = [dict(c) for c in mock_fallback]
            return BackendResult(
                content=None,
                tool_calls=tool_calls,
                cost_usd=0.0,
                latency_ms=0.0,
                model_used=model,
                backend_used=self.name,
                input_tokens=0,
                output_tokens=0,
                cache_read_tokens=0,
                is_mock=True,
            )

        # Translate Anthropic tool schemas to OpenAI function-calling shape.
        openai_tools = self._translate_tools(tools) if tools else None
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "stream": False,
        }
        if openai_tools:
            body["tools"] = openai_tools
            if tool_choice == "any" or (isinstance(tool_choice, dict) and tool_choice.get("type") == "tool"):
                # OpenAI equivalent of "force a tool": "required".
                body["tool_choice"] = "required"
            elif tool_choice == "auto" or (isinstance(tool_choice, dict) and tool_choice.get("type") == "auto"):
                body["tool_choice"] = "auto"

        last_exc: Exception | None = None
        for attempt in range(self.tool_call_retries + 1):
            try:
                result = self._post_once(body)
                return result
            except _MalformedToolCallError as exc:
                last_exc = exc
                _LOG.warning(
                    f"{self.name}.malformed_tool_call",
                    attempt=attempt + 1,
                    model=model,
                    exc=str(exc),
                )
                continue
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                _LOG.error(f"{self.name}.request_failed", attempt=attempt + 1, exc=str(exc))
                # Try fallback host once if we haven't
                if attempt == 0 and self.host_fallback and self.host != self.host_fallback:
                    _LOG.info(f"{self.name}.trying_host_fallback", fallback=self.host_fallback)
                    self.host = self.host_fallback
                    continue
                break

        raise RuntimeError(f"{self.name} call failed after retries: {last_exc}")

    def _post_once(self, body: dict[str, Any]) -> BackendResult:
        import urllib.request

        url = f"{self.host}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        headers.update(self._auth_headers())
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            raw = resp.read()
        latency_ms = (time.monotonic() - t0) * 1000.0
        data = json.loads(raw)
        return self._parse_chat_completion(data, latency_ms=latency_ms, model=body["model"])

    def _auth_headers(self) -> dict[str, str]:
        """Override in subclasses that need auth (StandardAPIBackend)."""
        return {}

    def _compute_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Override in subclasses that track token cost (StandardAPIBackend)."""
        return 0.0

    def _translate_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for t in tools:
            # Expect Anthropic shape: {name, description, input_schema}
            name = t.get("name", "tool")
            desc = t.get("description", "")
            schema = t.get("input_schema", {"type": "object", "properties": {}})
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": desc,
                        "parameters": schema,
                    },
                }
            )
        return out

    def _parse_chat_completion(
        self, data: dict[str, Any], *, latency_ms: float, model: str
    ) -> BackendResult:
        choices = data.get("choices") or []
        if not choices:
            raise _MalformedToolCallError("empty choices array")
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        tool_calls_raw = msg.get("tool_calls") or []
        tool_calls: list[dict[str, Any]] = []
        for tc in tool_calls_raw:
            fn = tc.get("function") or {}
            name = fn.get("name", "")
            args_raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else dict(args_raw)
            except json.JSONDecodeError as exc:
                raise _MalformedToolCallError(f"tool_call args JSON invalid: {exc}") from exc
            tool_calls.append({"name": name, "input": args})

        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)

        return BackendResult(
            content=content if isinstance(content, str) else None,
            tool_calls=tool_calls,
            cost_usd=self._compute_cost(prompt_tokens, completion_tokens, model),
            latency_ms=latency_ms,
            model_used=model,
            backend_used=self.name,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            cache_read_tokens=0,
            is_mock=False,
        )


# Backwards-compatibility alias — the backend used to be called OllamaBackend
# when Ollama was the only target. Keep the old name importable so external
# scripts don't break on the rename.
OllamaBackend = StandardLocalBackend


class StandardAPIBackend(StandardLocalBackend):
    """Backend for remote API providers using the industry-standard HTTP
    chat-completions format (the de-facto OpenAI-compatible shape).

    Known compatible providers: OpenAI, Groq, Together, OpenRouter,
    DeepSeek, Mistral, Fireworks, Perplexity, and others. Any provider
    that exposes `/v1/chat/completions` (or an equivalent path) with
    bearer-token auth and tool-calling support should work.

    Configuration via `config/llm_routing.yaml`:

        backend: standard_api
        host: https://api.groq.com/openai  # provider base URL
        api_key_env: GROQ_API_KEY          # env var holding the key
        model: llama-3.3-70b-versatile     # model served by the provider
        # Optional pricing for cost tracking:
        price_per_m_input: 0.59
        price_per_m_output: 0.79

    Cost is computed from token usage (prompt + completion) using the
    `price_per_m_*` config when set; otherwise 0.0 (unknown). Providers
    that do not expose usage will always report 0.0.
    """

    name: str = "standard_api"

    def __init__(
        self,
        *,
        host: str,
        api_key: str | None,
        host_fallback: str | None = None,
        timeout_s: float = 120.0,
        tool_call_retries: int = 2,
        price_per_m_input: float = 0.0,
        price_per_m_output: float = 0.0,
    ) -> None:
        super().__init__(
            host=host,
            host_fallback=host_fallback,
            timeout_s=timeout_s,
            tool_call_retries=tool_call_retries,
        )
        self.api_key = api_key
        self.price_per_m_input = float(price_per_m_input)
        self.price_per_m_output = float(price_per_m_output)

    def _auth_headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    def _compute_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        if self.price_per_m_input == 0.0 and self.price_per_m_output == 0.0:
            return 0.0
        return (
            input_tokens * self.price_per_m_input / 1_000_000.0
            + output_tokens * self.price_per_m_output / 1_000_000.0
        )


class _MalformedToolCallError(Exception):
    pass


# ---------------------------------------------------------------------------
# Configuration resolution
# ---------------------------------------------------------------------------


_CONFIG_CACHE: dict[str, Any] | None = None


def _load_config(path: Path | None = None) -> dict[str, Any]:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    if yaml is None:
        raise RuntimeError("PyYAML not installed — cannot parse config/llm_routing.yaml")
    cfg_path = path or _DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        _CONFIG_CACHE = {}
        return _CONFIG_CACHE
    with cfg_path.open("r", encoding="utf-8") as f:
        _CONFIG_CACHE = yaml.safe_load(f) or {}
    return _CONFIG_CACHE


def _reset_config_cache() -> None:
    """For tests."""
    global _CONFIG_CACHE
    _CONFIG_CACHE = None


def resolve_routing(agent_slot: str) -> dict[str, Any]:
    """Return `{"backend": ..., "model": ...}` for the given slot.

    Precedence: env override > profile > explicit agents entry > default.
    """
    cfg = _load_config()
    # 1. Per-slot env overrides
    section, _, slot = agent_slot.partition(".")
    env_backend_key = f"MDK_LLM_{section.upper()}_{slot.upper()}_BACKEND"
    env_model_key = f"MDK_LLM_{section.upper()}_{slot.upper()}_MODEL"
    env_backend = os.environ.get(env_backend_key)
    env_model = os.environ.get(env_model_key)

    # 2. Profile
    profile_name = os.environ.get("MDK_LLM_PROFILE")
    profile_entry: dict[str, Any] = {}
    if profile_name:
        prof = (cfg.get("profiles") or {}).get(profile_name) or {}
        profile_entry = prof.get(agent_slot) or {}

    # 3. Explicit agents entry
    agents_cfg = cfg.get("agents") or {}
    explicit_entry: dict[str, Any] = {}
    if section in agents_cfg and slot in (agents_cfg[section] or {}):
        explicit_entry = agents_cfg[section][slot] or {}

    # 4. Default
    default_entry = cfg.get("default") or {"backend": "anthropic", "model": "claude-sonnet-4-6"}

    backend = env_backend or profile_entry.get("backend") or explicit_entry.get("backend") or default_entry.get("backend")
    model = env_model or profile_entry.get("model") or explicit_entry.get("model") or default_entry.get("model")
    return {"backend": str(backend), "model": str(model)}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


_BACKEND_INSTANCES: dict[str, LLMBackend] = {}


def get_backend(agent_slot: str) -> tuple[LLMBackend, str]:
    """Return `(backend, resolved_model)` for the given slot.

    Supports three backend names:
      - `anthropic`       → AnthropicBackend (native Claude API)
      - `standard_local`  → StandardLocalBackend (Ollama / LM Studio / etc.)
      - `standard_api`    → StandardAPIBackend (OpenAI / Groq / Together / ...)

    Backward compatibility: `ollama` is accepted and silently mapped to
    `standard_local` (with a deprecation warning on stderr).
    """
    routing = resolve_routing(agent_slot)
    backend_name = routing["backend"]
    model = routing["model"]

    # Deprecation shim — accept legacy `ollama` name, emit warning, use standard_local.
    if backend_name == "ollama":
        import sys as _sys
        print(
            "Warning: 'ollama' backend name deprecated, use 'standard_local'. "
            "Backward compatibility maintained.",
            file=_sys.stderr,
        )
        backend_name = "standard_local"

    if backend_name == "anthropic":
        if "anthropic" not in _BACKEND_INSTANCES:
            _BACKEND_INSTANCES["anthropic"] = AnthropicBackend()
        return _BACKEND_INSTANCES["anthropic"], model

    if backend_name == "standard_local":
        # Instance keyed on (host, model) would be too granular; one shared
        # instance per backend name is fine because `call()` is stateless.
        if "standard_local" not in _BACKEND_INSTANCES:
            cfg = _load_config()
            # Slot-specific config may override host/timeout; fall back to
            # global `ollama` block for compatibility with older YAMLs, or to
            # a new `standard_local` block if the user chose the new name.
            slot_cfg = _slot_backend_config(cfg, agent_slot)
            legacy = cfg.get("standard_local") or cfg.get("ollama") or {}
            host = (
                os.environ.get("MDK_LLM_STANDARD_LOCAL_HOST")
                or os.environ.get("MDK_LLM_OLLAMA_HOST")
                or slot_cfg.get("host")
                or legacy.get("host")
                or "http://127.0.0.1:11434"
            )
            host_fb = slot_cfg.get("host_fallback") or legacy.get("host_fallback")
            _BACKEND_INSTANCES["standard_local"] = StandardLocalBackend(
                host=host,
                host_fallback=host_fb,
                timeout_s=float(legacy.get("timeout_s", 120.0)),
                tool_call_retries=int(legacy.get("tool_call_retries", 2)),
            )
        return _BACKEND_INSTANCES["standard_local"], model

    if backend_name == "standard_api":
        # Instance depends on host+api_key, so key the cache on the slot-level
        # host for per-provider instances. Simplification: one instance, and
        # swap its host/auth per call would be cleaner — for v0.1 we let the
        # first slot win and the user expresses one provider per run.
        if "standard_api" not in _BACKEND_INSTANCES:
            cfg = _load_config()
            slot_cfg = _slot_backend_config(cfg, agent_slot)
            global_api = cfg.get("standard_api") or {}
            host = slot_cfg.get("host") or global_api.get("host")
            if not host:
                raise ValueError(
                    f"standard_api backend for slot '{agent_slot}' requires a "
                    f"'host' value in config/llm_routing.yaml"
                )
            api_key_env = slot_cfg.get("api_key_env") or global_api.get("api_key_env", "")
            api_key = os.environ.get(api_key_env) if api_key_env else None
            if not api_key:
                raise ValueError(
                    f"standard_api backend for slot '{agent_slot}' requires "
                    f"env var '{api_key_env}' to be set with an API key"
                )
            _BACKEND_INSTANCES["standard_api"] = StandardAPIBackend(
                host=host,
                api_key=api_key,
                timeout_s=float(global_api.get("timeout_s", 120.0)),
                tool_call_retries=int(global_api.get("tool_call_retries", 2)),
                price_per_m_input=float(slot_cfg.get("price_per_m_input", 0.0) or global_api.get("price_per_m_input", 0.0)),
                price_per_m_output=float(slot_cfg.get("price_per_m_output", 0.0) or global_api.get("price_per_m_output", 0.0)),
            )
        return _BACKEND_INSTANCES["standard_api"], model

    raise ValueError(f"unknown backend '{backend_name}' for slot '{agent_slot}'")


def _slot_backend_config(cfg: dict[str, Any], agent_slot: str) -> dict[str, Any]:
    """Return the per-slot config block (merging profile override and
    explicit agents entry). Used to read host/api_key_env/pricing."""
    section, _, slot = agent_slot.partition(".")
    profile_name = os.environ.get("MDK_LLM_PROFILE")
    profile_entry: dict[str, Any] = {}
    if profile_name:
        prof = (cfg.get("profiles") or {}).get(profile_name) or {}
        profile_entry = prof.get(agent_slot) or {}
    agents_cfg = cfg.get("agents") or {}
    explicit_entry: dict[str, Any] = {}
    if section in agents_cfg and slot in (agents_cfg[section] or {}):
        explicit_entry = agents_cfg[section][slot] or {}
    # Profile overrides explicit agents entry
    merged = dict(explicit_entry)
    merged.update(profile_entry)
    return merged


__all__ = [
    "AnthropicBackend",
    "BackendResult",
    "LLMBackend",
    "OllamaBackend",  # backwards-compat alias → StandardLocalBackend
    "StandardAPIBackend",
    "StandardLocalBackend",
    "get_backend",
    "resolve_routing",
]
