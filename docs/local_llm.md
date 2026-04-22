# Local LLM / Inference rig

MDK Orchestra's agent layer talks to a pluggable **LLM backend**. The default
is Anthropic (cloud), but any or all agent slots can be routed to a local
LLM server running on the same machine or a second machine over the LAN.
This is useful for

- keeping specialist consultations cost-free while Maestro stays on Anthropic
  (hybrid mode),
- running the whole Orchestra locally with zero API spend (local mode, for
  ablation / cost-free ops tests),
- per-slot experiments — e.g. swap only the environment specialist to a
  local model while everything else is unchanged.

Compatible local servers: **Ollama**, **LM Studio**, **llama.cpp server**,
**vLLM**, **text-generation-webui**, or anything that exposes the
industry-standard `/v1/chat/completions` endpoint with tool-calling support.

## Topology (dual-machine example)

```
┌────────────────────────┐                    ┌────────────────────────┐
│  Primary machine       │                    │  Agent machine         │
│                        │                    │  (LAN reachable)       │
│  Orchestra (Python)    │ ── LAN HTTP ────▶  │  Ollama / LM Studio    │
│  agents/llm_backend.py │    :11434          │  Qwen 2.5 7B q4_K_M    │
│  config/llm_routing.yaml│                   │                        │
└────────────────────────┘                    └────────────────────────┘
```

The primary makes HTTP requests against the agent machine's
`/v1/chat/completions` endpoint. The tool-use contract is translated
inside `agents/llm_backend.py::StandardLocalBackend` so Maestro and
specialists stay backend-agnostic.

Single-machine setups work too — just leave `standard_local.host` at the
default `http://127.0.0.1:11434` and run the LLM server on the same box
as the Orchestra.

## Install Ollama on the agent machine via SSH

If the agent machine lacks passwordless sudo (typical on locked-down
remote boxes), Homebrew install won't work over SSH without a TTY. Use
the userspace binary install instead:

```bash
# From the primary machine (ssh'd to `agent` for this example):

# 1. Download the universal binary tarball to the agent machine
ssh agent 'curl -sL -o /tmp/ollama-darwin.tgz \
  https://github.com/ollama/ollama/releases/latest/download/ollama-darwin.tgz'

# 2. Extract into ~/ollama_bin
ssh agent 'mkdir -p ~/ollama_bin && cd ~/ollama_bin && tar xzf /tmp/ollama-darwin.tgz'

# 3. Put the binary on PATH
ssh agent 'mkdir -p ~/bin && ln -sf ~/ollama_bin/ollama ~/bin/ollama'

# 4. Launch the server in background, LAN-bound
ssh agent 'mkdir -p ~/ollama_data && \
  OLLAMA_HOST=0.0.0.0:11434 OLLAMA_MODELS=~/ollama_data \
  nohup ~/bin/ollama serve > ~/ollama_data/server.log 2>&1 & disown'

# 5. Pull a model (5 GB download)
ssh agent 'OLLAMA_HOST=127.0.0.1:11434 ~/bin/ollama pull qwen2.5:7b-instruct-q4_K_M'

# 6. Verify from the primary machine (replace `agent-machine.local` with
#    your agent machine's hostname or IP)
curl http://agent-machine.local:11434/api/tags
# {"models":[{"name":"qwen2.5:7b-instruct-q4_K_M",...}]}
```

**Known pitfalls**:

- `Ollama-darwin.zip` (GUI app bundle from github releases) ships an x86_64-only binary. On Apple Silicon without Rosetta 2 it hangs silently over SSH. Always use `ollama-darwin.tgz` (universal binary) instead.
- `brew install ollama` is the nicer path but requires interactive sudo over SSH — not possible without a TTY.
- The default `OLLAMA_HOST=127.0.0.1:11434` binds to localhost only. Set `0.0.0.0:11434` explicitly to listen on the LAN.

## Configuration

All routing decisions live in `config/llm_routing.yaml`. Each agent slot
has a default backend + model; **profiles** override entire blocks at
once; **env vars** override single slots.

### Agent slots

| Slot | Used by |
|---|---|
| `maestro.dispatch` | First-pass synthesis in `Maestro._synthesize` |
| `maestro.escalation` | Opus second opinion (fires when first-pass proposes L3/L4) |
| `maestro.curation` | Every-30-sim-min memory curation in `agents/curator.py` |
| `specialists.voltage` | VoltageAgent |
| `specialists.hashrate` | HashrateAgent |
| `specialists.environment` | EnvironmentAgent |
| `specialists.power` | PowerAgent |

### Predefined profiles

```bash
# Default — everything on Anthropic. Canonical baseline.
MDK_LLM_PROFILE=full_api   mdk-orchestra run

# Specialists on Qwen (local, free), Maestro on Anthropic.
MDK_LLM_PROFILE=hybrid_economic  mdk-orchestra run

# Everything on Qwen. Zero API spend. Longer latency, lower quality.
MDK_LLM_PROFILE=full_local  mdk-orchestra run
```

### Per-slot env override

The highest-precedence knob. Useful for one-off tests.

```bash
# Route ONLY the environment specialist to local Qwen; leave everything
# else on the full_api defaults.
MDK_LLM_SPECIALISTS_ENVIRONMENT_BACKEND=standard_local \
MDK_LLM_SPECIALISTS_ENVIRONMENT_MODEL=qwen2.5:7b-instruct-q4_K_M \
mdk-orchestra run
```

Env var key format: `MDK_LLM_<SECTION>_<SLOT>_BACKEND` / `_MODEL`
(section/slot uppercased, so `specialists.voltage` → `SPECIALISTS_VOLTAGE`).

## Tested models

| Model | Size | Ollama name | Tool-calling reliability | Notes |
|---|---|---|---|---|
| Qwen 2.5 7B Instruct (q4_K_M) | 4.7 GB | `qwen2.5:7b-instruct-q4_K_M` | ~85% | Verified on M4 16 GB. Occasional enum-violation tool_call (e.g. `assessment="step drop"` instead of `"real_signal"/"noise"/"inconclusive"`) — the StandardLocalBackend retries up to 2× on malformed JSON but does NOT retry on enum violations. The surviving specialist responses are usually enough for Maestro to synthesize. |

Models NOT yet tested: Llama 3.1 8B, Mistral 7B, larger quants. Add them
to `config/llm_routing.yaml` profiles as needed and smoke-test first.

## Performance expectations (M4 16 GB, Qwen 2.5 7B q4_K_M)

| Metric | full_api | hybrid_economic | full_local |
|---|---|---|---|
| Cost / decision | ~$0.023 | ~$0.006 | $0.000 |
| Latency / decision | 23 s | 44 s | 69 s |
| Specialist reply latency | ~7 s | ~16 s (Ollama) | ~20 s (Ollama) |
| Maestro synthesis latency | ~8 s (Sonnet) | ~8 s (Sonnet) | ~25 s (Qwen) |

The hybrid mode is the sweet spot — Maestro keeps the cache-warmed
Sonnet personality prompt, specialists are free but slower. full_local
is for runs where even $0.02 / decision is too much (research /
ablation) — expect lower reasoning-trace richness.

## Performance tuning (measured 2026-04-22)

Two knobs were evaluated against the default (q4_K_M, OLLAMA_NUM_PARALLEL=1):

### Q3_K_M quantisation vs Q4_K_M

Smaller model weights (3.8 GB vs 4.7 GB). In theory, lower memory bandwidth pressure = faster token generation.

| Profile | q4_K_M (pre) | q3_K_M (post) | Delta |
|---|---|---|---|
| `hybrid_economic` | 43.9 s total | 44.4 s total | **+0.5 s** (noise) |
| `full_local` | 69.5 s total | 45.3 s total | −24 s **but confounded** |

The `full_local` delta is NOT a real latency improvement. In the q4 run, Maestro's first-pass Qwen synthesis concluded `L3_throttle`, which triggered the Opus-escalation second pass. In the q3 run, Qwen first-pass concluded `L1_observe` directly, so no second synthesis fired — saving one Qwen call. Per-call latency for either a single specialist or a single synthesis is essentially unchanged between q3 and q4 on this hardware.

**Verdict**: on our dispatch pattern (Maestro consults primary, then fallback sequentially when primary is inconclusive — not in parallel), the quantisation knob does not move the needle. Both models are pulled on Mac B; we keep `q4_K_M` as the canonical default for its marginally better tool-call enum compliance. Run `scripts/smoke_test_backends.py` after switching if you want to re-measure.

### OLLAMA_NUM_PARALLEL

Setting `OLLAMA_NUM_PARALLEL=2` on the Ollama server allows two model-worker slots at once. This matters when multiple concurrent HTTP requests hit the server — e.g. if the dispatcher were to call the primary and fallback specialists in parallel threads.

Our current Maestro dispatch is **sequential** (fallback only fires when primary returns `inconclusive`, and only then), so `NUM_PARALLEL=2` has no measurable effect. Kept at 2 in the server config for future work (an A/B Track B mapper that batches decisions, or a refactor of Maestro to fire primary + fallback in parallel with an early-cancel on definitive primary). Verified active in the server log on restart.

## Known limitations

1. **Qwen 7B tool-call reliability ~85%** — occasional enum violations on
   forced tool_use. See StandardLocalBackend retry logic in `agents/llm_backend.py`.
2. **No prompt cache on Ollama**. Ephemeral cache is Anthropic-side only.
3. **Memory curation on local Qwen is untested at scale** — the curation
   schema has 8 required fields per tool call and Qwen 7B's structured
   output compliance on complex schemas is borderline. Profile `full_local`
   works in smoke but recommend `hybrid_economic` (Maestro curation stays
   on Opus) for any run longer than ~15 min.
4. **mDNS hostnames (`<name>.local`) may fail on some networks**.
   `config/llm_routing.yaml` accepts `host_fallback` pointing at a static
   IP; `StandardLocalBackend` switches to the fallback on first error.
5. **Ollama server is not auto-restarted**. If Mac B reboots, re-run
   step 4 of the install to relaunch `ollama serve`. A proper LaunchAgent
   plist is a future-work item.

## Smoke-test script

`scripts/smoke_test_backends.py` runs a single synthetic flag under any
profile and prints cost / latency / routing-per-slot. Use it to verify a
new profile or per-slot override before committing to a full A/B run.

```bash
MDK_LLM_PROFILE=hybrid_economic python scripts/smoke_test_backends.py
```
