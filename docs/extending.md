# Extending MDK Orchestra — Adding Custom LLM Backends

MDK Orchestra ships with three built-in backend types, each handled by one class in `agents/llm_backend.py`:

- **`AnthropicBackend`** — for Anthropic's API, using its native format (prompt caching, native tool use).
- **`StandardLocalBackend`** — for local LLM servers using the de-facto industry-standard HTTP format (Ollama, LM Studio, llama.cpp server, vLLM, text-generation-webui, …).
- **`StandardAPIBackend`** — for remote API providers using the same standard format (OpenAI, Groq, Together, OpenRouter, DeepSeek, Mistral, Fireworks, Perplexity, …).

The first question to ask is *which* of these three covers your provider. Most popular providers expose a chat-completions endpoint with tool-calling, and any of them can be driven by `StandardAPIBackend` with zero code changes.

## Using existing backends with a non-default provider

The two Standard backends work with any compatible server/provider by editing the YAML. No code changes required.

### Example: use Groq instead of Anthropic

```yaml
agents:
  maestro:
    dispatch:
      backend: standard_api
      host: https://api.groq.com/openai
      api_key_env: GROQ_API_KEY
      model: llama-3.3-70b-versatile
      price_per_m_input: 0.59
      price_per_m_output: 0.79
```

Then:

```bash
export GROQ_API_KEY=gsk_...
mdk-orchestra run
```

### Example: use LM Studio instead of Ollama

```yaml
agents:
  specialists:
    voltage:
      backend: standard_local
      host: http://localhost:1234     # LM Studio default
      model: Llama-3.2-8B-Instruct-Q4_K_M
```

### Example: use OpenAI for the environment specialist only

```yaml
agents:
  specialists:
    environment:
      backend: standard_api
      host: https://api.openai.com
      api_key_env: OPENAI_API_KEY
      model: gpt-4o-mini
      price_per_m_input: 0.15
      price_per_m_output: 0.60
```

Every agent slot is independently configurable — you can mix Anthropic Maestro + OpenAI specialists + local Ollama curation with no code.

## Writing a custom backend

If your provider uses a proprietary format that doesn't match the industry-standard chat completions shape (Google Gemini's native SDK, Cohere, any non-standard protocol), you'll need a new backend class.

### The interface

All backends implement the `LLMBackend` protocol defined in `agents/llm_backend.py`:

```python
class LLMBackend(Protocol):
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
```

`BackendResult` is a plain dataclass:

```python
@dataclass(frozen=True)
class BackendResult:
    content: str | None                     # free-form text, if any
    tool_calls: list[dict[str, Any]]        # [{"name": str, "input": dict}, ...]
    cost_usd: float
    latency_ms: float
    model_used: str
    backend_used: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    is_mock: bool
```

### Template

Create `agents/my_custom_backend.py`:

```python
from __future__ import annotations
import os
import time
from typing import Any
from agents.llm_backend import BackendResult


class MyCustomBackend:
    """Backend for <provider name> using <protocol>."""

    name: str = "my_custom"

    def __init__(self, *, host: str, api_key_env: str, timeout_s: float = 120.0) -> None:
        self.host = host.rstrip("/")
        self.api_key = os.environ.get(api_key_env)
        if not self.api_key:
            raise ValueError(f"env var '{api_key_env}' is not set")
        self.timeout_s = timeout_s

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
        t0 = time.monotonic()

        # 1. Translate Anthropic-style tool schemas into your provider's shape
        provider_tools = self._translate_tools(tools) if tools else None

        # 2. Make the HTTP / SDK call
        response = ...  # your provider's SDK here

        # 3. Translate response back into the standard BackendResult shape
        return BackendResult(
            content=response.text,
            tool_calls=self._translate_tool_calls(response),
            cost_usd=self._compute_cost(response.usage),
            latency_ms=(time.monotonic() - t0) * 1000.0,
            model_used=model,
            backend_used=self.name,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            cache_read_tokens=0,
            is_mock=False,
        )

    def _translate_tools(self, tools):
        # Anthropic shape: {name, description, input_schema}
        # Translate into whatever your provider expects.
        ...

    def _translate_tool_calls(self, response):
        # Return: [{"name": str, "input": dict}, ...]
        ...

    def _compute_cost(self, usage):
        # Return USD cost based on token usage.
        ...
```

### Register in the factory

In `agents/llm_backend.py`, add a branch to `get_backend()`:

```python
if backend_name == "my_custom":
    if "my_custom" not in _BACKEND_INSTANCES:
        from agents.my_custom_backend import MyCustomBackend
        slot_cfg = _slot_backend_config(cfg, agent_slot)
        _BACKEND_INSTANCES["my_custom"] = MyCustomBackend(
            host=slot_cfg.get("host"),
            api_key_env=slot_cfg.get("api_key_env"),
        )
    return _BACKEND_INSTANCES["my_custom"], model
```

### Use it in YAML

```yaml
agents:
  maestro:
    dispatch:
      backend: my_custom
      host: https://api.myprovider.com
      api_key_env: MY_PROVIDER_KEY
      model: my-model-name
```

## Caveats

- **Tool calling** is where protocol differences bite hardest. Providers may differ on: whether `tools` is nested under `function`, how `tool_choice` is expressed, whether arguments come back as a JSON *string* or already-parsed dict. Read the provider's tool-calling docs carefully and mirror `StandardLocalBackend._parse_chat_completion` as the reference.
- **Cost reporting** depends on whether the provider exposes token usage in the response. If not, return `cost_usd=0.0` and flag it in your docs.
- **Response streaming** is not currently supported — all calls are blocking synchronous. A streaming interface is a v0.2 conversation.
- **Small models** occasionally return malformed tool-call JSON. The shipped backends retry up to N times (configurable via `tool_call_retries`). Implement similar logic in custom backends if you're targeting ≤7B local models.

## Backward-compatibility notes

- `OllamaBackend` is preserved as an alias for `StandardLocalBackend` so code importing the old name still works.
- `backend: ollama` in YAML is accepted with a deprecation warning — prefer `backend: standard_local`.

## Contributing a backend upstream

If you write a high-quality backend for a popular provider, please consider opening a PR. See [CONTRIBUTING.md](../CONTRIBUTING.md) for the workflow.
