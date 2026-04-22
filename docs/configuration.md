# MDK Orchestra — Configuration Reference

All configuration files, environment variables, and CLI flags.

## Interactive wizard

Running `mdk-orchestra` with no subcommand launches an interactive wizard. It offers four options and exits on `Ctrl+C`:

1. **Demo** — replay of the canonical API validation run. The wizard plays back a pre-recorded event stream (`examples/demo_replay/`) at **1× real-time** against the real dashboard, producing 38 flags / 28 Orchestra decisions / 28 actions / 559 fleet snapshots in ~12 minutes wall time — matching the original run's pace so you can read the reasoning traces as they arrive. **No API key, no LLM required — cost $0.** Wraps the `replay` subcommand; pass `--speed 4` (or higher) for accelerated playback.
2. **API** — full system on a remote LLM. Sub-menu:
   - Anthropic (native) — prompts for the key inline or uses `ANTHROPIC_API_KEY` if already set. Sonnet for specialists + Opus for Maestro escalation.
   - Custom OpenAI-compatible — prompts for base URL, API key env var name, model name, and the key itself (inline or via the named env var). Applies one model to every agent slot.
3. **Local LLM** — autodetects Ollama at `localhost:11434` and lists its models. Falls back to a prompt for a custom local host (LM Studio, llama.cpp server, vLLM, remote Ollama) if autodetect fails.
4. **Explore** — simulator + flag detection, no agents. 50 miners, 10× accelerated, balanced fault mix (faults DO get injected). Rule engine + XGBoost predictors emit flags live into the dashboard; Maestro, specialists, and the executor stay idle. The alert log fills; the decision and action panels remain empty. Open-ended — `Ctrl+C` stops it.

The wizard writes the effective routing into `runs/<timestamp>/config_used.yaml` for audit. It also writes `runs/<timestamp>/backend_summary.json`, which the dashboard reads to render the active-backend badge in its header (e.g. `Backend: Demo Replay · Canonical API Run (1×)` or `Backend: Explore mode · simulator + flag detection · no agents`).

For scripting (CI, reproducibility), use the explicit subcommands — they behave the same as before the wizard was added:

- `mdk-orchestra demo [--speed 4] [--source DIR]` — replay (default source: `examples/demo_replay/`)
- `mdk-orchestra replay` — alias for `demo` with the same flags
- `mdk-orchestra run --profile <name>` — live A/B run with LLM backend
- `mdk-orchestra simulator --duration 60` — simulator alone
- `mdk-orchestra train` / `discover`

## Demo replay

The Demo option is powered by `mdk-orchestra replay`, which:

- Reads `flags.jsonl`, `decisions.jsonl`, `actions.jsonl`, `snapshots.jsonl` from a source directory (default: `examples/demo_replay/`)
- Merges all events into a single chronological timeline
- Streams them one-by-one into a fresh `runs/demo_<timestamp>/` directory, spacing them by `original_delta_s / speed` of wall-clock
- Launches the dashboard pointed at the new run dir — it starts empty and populates live as the replay progresses
- Writes `replay_meta.json` when the last event lands, which the wizard's demo flow reads to show its completion summary

Source event stream in `examples/demo_replay/` comes from the canonical API run (`ab_short_complete_20260422_0958`) — 38 flags, 28 decisions, autonomy ladder fully exercised, hero case `m040 chip_instability → L4 human_review`. Telemetry + KPIs (too large for the repo) are excluded; the dashboard's fleet grid reads from `snapshots.jsonl` which is included.

Configure with:

- `--speed N` (default 4) — playback speed multiplier
- `--source DIR` — alternate source directory
- `--run-id ID` — deterministic run-dir name (default `demo_<timestamp>`)
- `--no-dashboard` — just write the replayed stream to disk, skip the Flask server

## Output directories

All run artifacts land under a single root, defaulting to
**`~/.mdk-orchestra/runs/`**. Each invocation (demo / explore / run)
creates its own timestamped subdirectory inside that root.

The home-relative default is deliberate: it survives
`pipx uninstall mdk-orchestra` and `pipx upgrade`, respects POSIX
application-state conventions, and doesn't require `$PATH` or working
directory to be in any particular place.

**Priority order** (highest first) when resolving the root:

1. CLI flag: `mdk-orchestra replay --output-dir /custom/path ...`
2. Environment variable: `MDK_RUNS_DIR=/custom/path mdk-orchestra ...`
3. Default: `~/.mdk-orchestra/runs/`

The resolver is centralised in `shared/paths.py::get_runs_dir` and
used by the CLI wizard, the `replay` subcommand, and the
`ab_experiment` A/B runner. Any new code that writes run data should
go through the same helper.

## Session isolation

Every invocation of `demo` or `run` creates a fresh directory under
the runs root:

```
runs/20260422_083045/
├── config_used.yaml           ← routing snapshot for this run
├── backend_summary.json       ← backend label shown by the dashboard
├── flags.jsonl                ← append-only flags stream (starts empty)
├── decisions.jsonl            ← append-only decisions (starts empty)
├── actions.jsonl
├── telemetry.jsonl · kpis.jsonl · snapshots.jsonl
├── memory_snapshot_start/     ← agents/*_memory.md at run start
└── logs/
    ├── simulator_main.log
    ├── agents_main.log
    └── dashboard_main.log
```

The dashboard (`MDK_STREAM_DIR` pointed at the run dir) starts with zero flags and zero decisions; it populates live as the Orchestra produces them. Previous runs remain untouched — nothing bleeds across sessions.

`runs/` is gitignored by default.

## File-based configuration

### `config/llm_routing.yaml` — LLM backend + model per agent slot

Defines which backend (`anthropic` / `standard_local` / `standard_api`) and which model handles each agent slot. Organized into four sections:

**`default`** — catch-all fallback when a slot has no explicit entry.

**`agents`** — explicit per-slot defaults. Every agent slot can be individually configured:
- `maestro.dispatch` — first-pass synthesis
- `maestro.escalation` — second-opinion on L3/L4 decisions
- `maestro.curation` — periodic memory curation
- `specialists.voltage`
- `specialists.hashrate`
- `specialists.environment`
- `specialists.power`

**`profiles`** — named bundles that override entire slot sets at once. Three shipped:
- `full_api` — all Anthropic (default)
- `hybrid_economic` — specialists on local LLM, Maestro on Anthropic
- `full_local` — everything on local LLM

**Backend-specific blocks** (`standard_local`, `standard_api`) — shared defaults (host, timeout, retries) that slots inherit.

Resolution precedence (highest first):
1. Per-slot env override: `MDK_LLM_<SECTION>_<SLOT>_BACKEND` and `_MODEL`
2. Profile override: `MDK_LLM_PROFILE=<profile_name>`
3. Explicit `agents` section entry
4. `default` section

### `config/ave_calibration.yaml` — AVE metric calibration

The AVE (Agent Value-added Efficiency) formula measures decision quality per dollar-second. Calibration knobs live here:
- `value_by_severity` — USD value of a correct decision, by flag severity (info / warn / crit)
- `miscalibration_penalty` — USD penalty when wrong, by error class (correct / action_only / adjacent_under / adjacent_over / distant_under / distant_over)

See `report/compute_ave.py` for the full formula and scoring logic.

## Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `ANTHROPIC_API_KEY` | Enables `AnthropicBackend`. Required for any profile using Anthropic. | — |
| `MDK_LLM_PROFILE` | Switch the LLM routing profile globally (overrides YAML default). | unset |
| `MDK_LLM_<SECTION>_<SLOT>_BACKEND` | Override a single agent slot's backend. E.g. `MDK_LLM_SPECIALISTS_VOLTAGE_BACKEND=standard_local`. | unset |
| `MDK_LLM_<SECTION>_<SLOT>_MODEL` | Override a single slot's model. | unset |
| `MDK_LLM_STANDARD_LOCAL_HOST` | Override the `standard_local` host (e.g. point to LM Studio on :1234). | YAML value |
| `MDK_LLM_OLLAMA_HOST` | Deprecated alias of the above. | — |
| `MDK_STREAM_DIR` | Where JSONL event streams are written. | repo `events/` |
| `MDK_MEMORY_DIR` | Where per-track episodic memory writes go (A/B only). | auto per run |
| `MDK_DASH_HOST` | Dashboard bind host. | `127.0.0.1` |
| `MDK_DASH_PORT` | Dashboard port. | `8000` |
| `MDK_CURATION_INTERVAL_MIN` | Curator firing interval in simulated minutes. | `30.0` |
| `MDK_AGENT_MOCK` | Force-mock the agent layer (no real LLM calls). Useful for smoke tests. | unset |

## CLI flags

All `mdk-orchestra` subcommands accept `--help` for a full listing. The most-used flags:

### `mdk-orchestra demo`

| Flag | Default | Purpose |
|---|---|---|
| `--duration MIN` | 10 | Simulated duration in minutes |
| `--miners N` | 50 | Number of miners in the simulated fleet |
| `--profile NAME` | `full_api` | LLM routing profile |
| `--fault-mix {random,balanced}` | `balanced` | Fault scheduling strategy |
| `--dashboard-port PORT` | 8000 | Dashboard HTTP port |

### `mdk-orchestra run`

| Flag | Default | Purpose |
|---|---|---|
| `--profile NAME` | `full_api` | LLM routing profile |
| `--duration MIN` | 60 | Simulated duration in minutes |
| `--miners N` | 50 | Miners per track |
| `--seed N` | 42 | RNG seed |
| `--fault-mix {random,balanced}` | `balanced` | Fault scheduling |
| `--run-id ID` | auto | Run identifier |
| `--output DIR` | `runs/` | Output root directory |

### `mdk-orchestra simulator`

Simulator-only mode (for dataset generation / ML training). Skips the entire LLM layer.

| Flag | Default | Purpose |
|---|---|---|
| `--duration MIN` | 60 | Simulated duration |
| `--miners N` | 50 | Number of miners |
| `--seed N` | 42 | RNG seed |
| `--speed RATIO` | 10.0 | Simulated-to-wall time ratio |
| `--fault-mix {random,balanced}` | `balanced` | Fault scheduling |
| `--output PATH` | `runs/telemetry.jsonl` | Output JSONL |

## Memory files — where state lives

The 5 files under `agents/` (`maestro_memory.md`, `voltage_memory.md`, `hashrate_memory.md`, `environment_memory.md`, `power_memory.md`) carry the *only* LLM-layer state that persists across runs. Everything else is derivable from code + config + input data.

They ship **empty** in this repo. Two ways to populate them:
1. Run the Orchestra with memory curation enabled (default ON for API runs) — the curator will write patterns it distills from the decision log.
2. Seed from a prior pilot: copy files from `examples/memories/` into `agents/` (see [examples/memories/README.md](../examples/memories/README.md)).

## Model weights

XGBoost predictors ship as pre-trained pickles under `models/`:
- `xgb_predictor.pkl` — hashrate-degradation 30-min-ahead (active)
- `xgb_chip_instability.pkl` — chip-instability precursor (active)

Other candidate models (hashboard_failure, isolation_forest_v2) were trained during development but failed validation and are not included. See `models/ensemble_summary.md` for the ablation rationale.

To retrain on your own dataset:

```bash
mdk-orchestra train --data-dir path/to/telemetry
```

This runs `scripts/retrain_xgb_miner_wise.py` with the miner-wise 80/20 split that gives an honest OOD-style evaluation.
