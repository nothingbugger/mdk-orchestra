# MDK Orchestra

Predictive maintenance for Bitcoin mining — a multi-agent AI system built to catch pre-failure patterns before they become downtime.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![Status](https://img.shields.io/badge/status-v0.1%20beta-orange.svg)]()

## What is this?

A demonstration software implementing a multi-agent AI system for predictive maintenance of Bitcoin mining farms. It contains:

- **50-miner fleet simulator** with realistic telemetry and fault injection
- **Deterministic detection layer** — rule engine + 2 pre-trained XGBoost predictors
- **Orchestra multi-agent reasoning layer** — 4 domain specialists (voltage / hashrate / environment / power) + a Maestro conductor + periodic memory curator
- **Memory system** — agents learn by writing patterns, not by updating weights
- **Live dashboard** — terminal-style UI streaming flags, decisions, and actions via SSE
- **LLM backend abstraction** — Anthropic API, plus industry-standard HTTP backends for local servers (Ollama, LM Studio, llama.cpp, vLLM) and remote providers (OpenAI, Groq, Together, OpenRouter, DeepSeek, Mistral, …)

Originally developed as the DEV Track thesis for Plan B Academy 2026 (Tether MDK assignment). This repository is the demonstration software — for experimental results and analysis, see the companion [mdk-orchestra-fullreport](https://github.com/nothingbugger/mdk-orchestra-fullreport) repo.

## Quick start (reviewers)

One command to try it, courtesy of [pipx](https://pypa.github.io/pipx/):

```bash
pipx install git+https://github.com/nothingbugger/mdk-orchestra.git
mdk-orchestra
```

That's it — `pipx` creates an isolated venv for MDK Orchestra, wires the
`mdk-orchestra` binary onto your `$PATH`, and hands control to the
interactive wizard. No system-Python pollution, no `pip install -e .`
ceremony.

The wizard opens with four options:

- **Demo** — replay the canonical validation run (no API, no LLM required)
- **API** — full system on Anthropic or any OpenAI-compatible provider
- **Local LLM** — full system on Ollama or a compatible local server
- **Explore** — simulator + flag detection, no agents

Runs are stored in **`~/.mdk-orchestra/runs/`** by default — this
directory survives `pipx uninstall` / `pipx upgrade`, so your history
is never lost to a package-lifecycle operation. Override with
`MDK_RUNS_DIR=/path` (env var) or `--output-dir /path` (CLI flag).

When you're done:

```bash
pipx uninstall mdk-orchestra
```

Requires **Python 3.11+** and **pipx**. Install pipx with
`brew install pipx` on macOS, or follow
<https://pypa.github.io/pipx/installation/> for other platforms.

---

## Quick start (developers)

If you want to modify the code:

```bash
git clone https://github.com/nothingbugger/mdk-orchestra
cd mdk-orchestra
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
mdk-orchestra
```

> **macOS / Homebrew users**: the `python3 -m venv .venv` step is not optional.
> Recent Homebrew Python enforces [PEP 668](https://peps.python.org/pep-0668/) and
> blocks `pip install -e .` at the system level with an "externally-managed-environment"
> error. Create the venv first and install inside it — otherwise the `mdk-orchestra`
> command will never land on your `$PATH`.

Running `mdk-orchestra` with no arguments opens an interactive wizard with four options:

- **Demo** — replay of the canonical API validation run (38 flags, 28 Orchestra decisions, autonomy L1 → L4 all exercised). Plays back at 1× real-time (~12 min wall, matching the original run) so you can read the reasoning traces as they appear. **No API key, no LLM required.** Pass `--speed 4` (or higher) to the `replay` subcommand if you want accelerated playback.
- **API** — full system on a remote LLM. Anthropic native, or any OpenAI-compatible provider (OpenAI, Groq, Together, OpenRouter, DeepSeek, Mistral, …). You can paste the API key inline or fall back to the env var.
- **Local LLM** — full system on a local inference server. Autodetects Ollama on `localhost:11434`, also supports LM Studio / llama.cpp server / vLLM.
- **Explore** — simulator + flag detection, no agents. 50 miners with balanced fault mix, 10× accelerated, rule engine + XGBoost predictors emitting flags into the dashboard. Maestro/specialists/executor stay idle — the decision and action panels remain empty. Use it to see what the detection layer produces before committing to a full Orchestra run.

You'll see the simulator, deterministic detection, and Orchestra reasoning all running live. Open `http://127.0.0.1:8000` in a browser to watch the dashboard — the active backend is shown in the header (e.g. `Backend: Anthropic API · Sonnet 4.6 + Opus 4.7`).

Every run creates its own `runs/<timestamp>/` directory — dashboard events start clean each session.

## Install

**Prerequisites:** Python 3.11+, `git`, and either an Anthropic API key or a compatible local LLM server installed.

```bash
git clone https://github.com/nothingbugger/mdk-orchestra
cd mdk-orchestra
pip install -e .
```

For screenshot capture (optional):

```bash
pip install -e ".[screenshots]"
playwright install chromium
```

### LLM backend options

MDK Orchestra needs *some* LLM backend reachable to drive the Orchestra layer. You have three choices:

**Option A — Anthropic API (recommended quality, small cost).**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
mdk-orchestra demo --profile full_api
```

**Option B — Local LLM (free, private, requires a compatible server).**

Install [Ollama](https://ollama.ai/download) (simplest option — LM Studio / llama.cpp server / vLLM also work):

```bash
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama serve
mdk-orchestra demo --profile full_local
```

**Option C — Third-party API (Groq, OpenAI, Together, OpenRouter, DeepSeek, Mistral, …).**

Uncomment the `groq_specialists` profile in `config/llm_routing.yaml`, set the corresponding API key env var, and run with that profile. See [docs/extending.md](docs/extending.md) for details.

## Configuration

### Profiles

Three pre-configured routing profiles ship out of the box:

| Profile | Maestro | Specialists | Cost/decision | Notes |
|---|---|---|---|---|
| `full_api` | Anthropic | Anthropic | medium | Highest quality, requires `ANTHROPIC_API_KEY`. Default. |
| `hybrid_economic` | Anthropic | Local (Ollama) | lower | Specialists free, Maestro on cloud for caching + curation. |
| `full_local` | Local | Local | $0.00 | Zero-cost, full-privacy. Needs 8–16 GB RAM for 7B model. |

Switch with `--profile <name>` on the CLI, or via env var:

```bash
export MDK_LLM_PROFILE=full_local
mdk-orchestra run
```

### Per-agent routing

Edit `config/llm_routing.yaml` to customize backend and model for each agent individually. Every slot — `maestro.dispatch`, `maestro.escalation`, `maestro.curation`, `specialists.voltage`, `specialists.hashrate`, `specialists.environment`, `specialists.power` — can be independently configured.

### Agent memories

The five `agents/*_memory.md` files start empty on first install. Two mechanisms populate them:

- **Live curation** during API runs — the Maestro writes patterns it distills from decision history every 30 simulated minutes (default).
- **Seed from a prior pilot** — copy `examples/memories/sample_*_memory.md` into `agents/` to start from a known-good knowledge base. See [examples/memories/README.md](examples/memories/README.md).

## CLI reference

| Command | What it does |
|---|---|
| `mdk-orchestra` | Interactive wizard (Demo / API / Local LLM / Explore). |
| `mdk-orchestra demo [--speed N]` | Replay the canonical API run at 1× real-time (no API, no LLM). |
| `mdk-orchestra replay --speed N` | Alias for `demo`; replays at configurable speed. |
| `mdk-orchestra run [--profile P] [--duration MIN]` | Full A/B run — Track A (Orchestra) vs Track B (deterministic baseline). |
| `mdk-orchestra simulator [--duration MIN]` | Run only the simulator (dataset generation, ML training). |
| `mdk-orchestra train [--data-dir D]` | Retrain the XGBoost predictors on a new dataset. |
| `mdk-orchestra discover [--hours H]` | Run pattern-discovery scripts on simulator data. |
| `mdk-orchestra --help` | Full help with examples. |

All run artifacts land under `~/.mdk-orchestra/runs/<run_id>/` by default (override via `MDK_RUNS_DIR` env or `--output-dir`). Runs survive `pipx uninstall`.

## What's in the dashboard

Once a session is live at `http://127.0.0.1:8000/`:

- **Miners Map** — 50 cells. Green = full operative. Yellow pulse = active warn flag on that miner. Red pulse = active crit flag. Orange = throttling (persistent after an L3 decision). Grey = shut. Coloring is **flag-driven** — a miner doesn't turn yellow from HSI noise.
- **Live Flags Feed** — each flag card shows its type (e.g. `hashrate_degradation`) colored by severity. While the Orchestra is still reasoning, a pulsing mint dot narrates the phase ("Maestro dispatching → hashrate_agent…", "hashrate_agent reasoning…", "Maestro synthesizing decision…"). When the decision lands, the card fades and a `✓ L3_bounded_auto` tag appears (white / yellow / orange / red by level).
- **Reasoning Trace Log** — each decision rendered collapsed to 2 lines; click to expand the full reasoning trace. Scroll to see older entries.
- **Backend badge** (header) — `Anthropic API · Sonnet 4.6 + Opus 4.7`, `Demo Replay · Anthropic API run (1×)`, `Local · …`, `Explore mode · simulator + flag detection · no agents`.
- **Fleet TE / HSI bars** — smoothed over a 5 s rolling window so fleet-relative TE doesn't jitter when the fleet is at rest. HSI shown with a visual `−3` penalty per throttled miner so operator-visible fleet state reacts to Orchestra actions.

Every invocation creates a fresh run dir — the dashboard starts empty and populates live. No stale events from prior sessions.

## Architecture

```
  ┌──────────┐   telemetry   ┌──────────┐   kpis/snaps   ┌──────────┐
  │ Simulator├──────────────►│  Ingest  ├───────────────►│ Dashboard│
  │ (50 min) │               └──────────┘                └────▲─────┘
  └────┬─────┘                     │                          │
       │ telemetry                 │                    flags │ decisions
       ▼                           ▼                          │ actions
  ┌──────────────────────┐   ┌─────────────────────────────┐  │
  │ Deterministic layer  │   │      Orchestra (LLM)        │──┘
  │   rule engine        │   │  Maestro ⇄ 4 specialists    │
  │   XGB hashrate pred  ├──►│  + curated memory           │
  │   XGB chip instab.   │   │  + executor (autonomy lad.) │
  └──────────────────────┘   └─────────────────────────────┘
```

See [docs/architecture.md](docs/architecture.md) for the full architecture, data flow, and event schemas.

## Documentation

- [docs/architecture.md](docs/architecture.md) — how the system works end to end
- [docs/configuration.md](docs/configuration.md) — all YAMLs, env vars, and CLI flags
- [docs/extending.md](docs/extending.md) — adding custom LLM backends
- [docs/local_llm.md](docs/local_llm.md) — local inference rig setup
- [docs/federated_memory_design.md](docs/federated_memory_design.md) — future direction: pattern sharing across deployments via Nostr/BitTorrent over Tor
- [examples/memories/README.md](examples/memories/README.md) — sample agent memories

## Troubleshooting

**`mdk-orchestra: command not found` after `pip install -e .`**

You almost certainly hit PEP 668. On macOS with Homebrew Python (and most
recent Linux distros) `pip install -e .` at the system level is blocked with an
`externally-managed-environment` error — often scrolled off-screen. If the install
partially succeeded via `--user`, the console script lands in a bin dir that's
not on `$PATH` by default.

**Fix:** always install inside a venv:

```bash
cd mdk-orchestra
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
mdk-orchestra      # now resolves to .venv/bin/mdk-orchestra
```

Re-activate the venv every new shell (`source .venv/bin/activate`) or add
`.venv/bin` to your `$PATH`.

**Wizard shows no output / hangs**

The wizard uses `input()` with the default stdin. If you're running under a
harness that redirects stdin (e.g. `nohup`), feed explicit input or use the
explicit subcommands (`mdk-orchestra demo`, `mdk-orchestra run …`) instead.

**`No LLM backend available` fail-fast error**

Expected when neither `ANTHROPIC_API_KEY` is set nor a local LLM server is
reachable on `localhost:11434`. See [Install → LLM backend options](#install)
for setup.

## Contributing

Bug reports, feature requests, and pull requests welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

If you add a backend for a provider not yet supported, please consider opening a PR to share it upstream.

## License

Apache 2.0 — see [LICENSE](LICENSE). Copyright 2026 Daniele Serlenga.

## Credits

Created by Daniele Serlenga as the DEV Track thesis for Plan B Academy 2026 / Tether MDK assignment. Built with Claude Code as the development partner.

For detailed experimental results, architectural comparisons, and the original pitch, see the [mdk-orchestra-fullreport](https://github.com/nothingbugger/mdk-orchestra-fullreport) repo.
