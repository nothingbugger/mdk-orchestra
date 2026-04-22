# MDK Orchestra — Architecture

## One-paragraph summary

A multi-agent fleet for predictive maintenance of Bitcoin miners. A simulator streams real-time telemetry for 50 miners. A deterministic detection layer (rule engine + two XGBoost predictors) raises flags on pre-failure patterns. An LLM orchestrator ("Maestro") dispatches each flag to the relevant domain specialist(s), synthesizes their reasoning, and emits a decision. An executor enforces an autonomy ladder that caps which actions can be taken autonomously versus queued for human approval. A dashboard streams everything live via Server-Sent Events.

## Layered architecture

```
Layer 1 — Simulator (simulator/)
    50-miner fleet with realistic telemetry + scheduled fault injection.
    Writes telemetry.jsonl at 10× simulated speed by default.

Layer 2 — Ingest (ingest/)
    Computes per-miner KPIs (TE = Telemetry Efficiency,
    HSI = Hardware Stress Index) and fleet-level snapshots.
    Writes kpis.jsonl and snapshots.jsonl.

Layer 3 — Deterministic detection (deterministic_tools/)
    Rule engine (thermal runaway, fan anomaly, voltage drift)
    + XGBoost hashrate-degradation predictor (20%-drop, 30-min horizon)
    + XGBoost chip-instability precursor.
    Writes flag_raised events to flags.jsonl.

Layer 4 — Orchestra (agents/)
    Maestro reads flags, dispatches to the relevant specialist(s), waits
    for their verdicts, synthesizes a decision, and emits it.
    Four specialists (voltage / hashrate / environment / power) each with
    a personality .md file and a read-only memory .md file.
    Curator periodically (every 30 sim-min) distills the decision log
    into new patterns the specialists and Maestro will read on future
    calls — learning by writing, not by fine-tuning.
    Writes orchestrator_decision events to decisions.jsonl.

Layer 5 — Action (action/)
    Executor enforces the autonomy ladder (L1..L4). L3 (reversible,
    bounded) executes immediately with a rollback timestamp. L4 (human
    approval required) is queued. Writes action_taken to actions.jsonl.

Layer 6 — Dashboard (dashboard/)
    Flask + SSE live view of the fleet, with per-miner telemetry,
    KPI trajectories, and incoming flags/decisions/actions.
```

## Data flow

Every subsystem writes to JSONL files on a shared stream directory. No Redis, no Kafka — just append-only files tailed by consumers. Event envelope is identical for every event (see `shared/schemas/events.py`):

```json
{"event": "<name>", "ts": "<iso8601>", "source": "<producer>", "data": {...}}
```

Nine event types in total: `telemetry_tick`, `kpi_update`, `fleet_snapshot`, `flag_raised`, `reasoning_request`, `reasoning_response`, `orchestrator_decision`, `action_taken`, `episodic_memory_write`.

## The autonomy ladder

Maestro decisions carry one of four autonomy levels:

| Level | Name | Who acts | Example |
|---|---|---|---|
| **L1** | `observe` | system logs only | weak signal on healthy telemetry |
| **L2** | `suggest` | operator approves | flag that needs human sanity-check |
| **L3** | `bounded_auto` | system acts, reversibly | throttle to 70% with 4h rollback |
| **L4** | `human_only` | queued for approval | shutdown, retire, voltage change |

The executor re-enforces this — if Maestro somehow emits an L3 action whose concrete `action` field is consequential (e.g. `shutdown`), the executor rejects and re-routes to L4. The ladder is a contract, not a suggestion.

## Specialist dispatch

Maestro does not broadcast every flag to every specialist — that would be wasteful. A dispatch table (mirrored in `agents/maestro.md` and enforced in `agents/maestro.py::DISPATCH_TABLE`) maps each flag type to a `(primary, fallback)` specialist pair:

| Flag type | Primary | Fallback (consulted only if primary is inconclusive) |
|---|---|---|
| `voltage_drift` | voltage_agent | power_agent |
| `hashrate_degradation` | hashrate_agent | voltage_agent |
| `chip_instability_precursor` | hashrate_agent | voltage_agent |
| `thermal_runaway` | environment_agent | voltage_agent |
| `fan_anomaly` | environment_agent | — |
| `power_instability` | power_agent | voltage_agent |
| `chip_variance_high` | voltage_agent | hashrate_agent |
| `anomaly_composite` | **all four** in parallel | (Flow 3) |

Definitive primary verdicts (`real_signal` or `noise`) short-circuit the fallback — one specialist call is enough.

## Tiered synthesis

For cost efficiency, Maestro uses two models:

1. **First-pass** on a cheap model (Sonnet by default). Handles L1/L2 decisions directly.
2. **Second opinion** on Opus. Fires only when the first-pass proposes L3 or L4 — the consequential path.

This roughly halves the per-decision cost vs. always using Opus, without sacrificing quality on the consequential decisions where it matters.

## Memory curation

After each decision, Maestro buffers it in the curator state. Every 30 simulated minutes (configurable via `MDK_CURATION_INTERVAL_MIN`), the curator fires: reads the buffer, distills recurring decision shapes into patterns, and atomically writes them into `agents/maestro_memory.md` and (when relevant) the specialist memory files.

Specialists and Maestro re-read their memory files on every call as part of the system prompt, so a new pattern curated at T=30 min is active from T=30 min onwards.

This is *learning by writing*, not by fine-tuning. The weights never change. The state lives in the 5 plain-text memory files, which are:
- human-readable (operators can inspect / edit / veto patterns)
- versionable (git-trackable)
- portable (copy them between deployments, see [federated_memory_design.md](federated_memory_design.md))

## LLM backend abstraction

The `agents/llm_backend.py` module defines an `LLMBackend` protocol that any backend implements. Three built-ins:

- `AnthropicBackend` — native Anthropic API with prompt caching
- `StandardLocalBackend` — local LLM server via industry-standard chat-completions HTTP (Ollama, LM Studio, llama.cpp, vLLM, …)
- `StandardAPIBackend` — remote provider via the same standard format (OpenAI, Groq, Together, OpenRouter, DeepSeek, Mistral, …)

Routing (which backend for which agent slot, under which profile) is declared in `config/llm_routing.yaml`. See [configuration.md](configuration.md) for the full YAML schema and [extending.md](extending.md) for adding custom backends.

## Testing

The test suite (`tests/`) covers:

- Event schema round-trip validation
- Maestro dispatch logic
- Curator write path (atomic file replace, LRU eviction)
- Action executor autonomy enforcement
- End-to-end smoke with mock LLM backend

Run with:

```bash
pytest tests/
```

Integration tests that hit a real LLM are gated behind `ANTHROPIC_API_KEY` being set.
