# MDK Fleet Dashboard

Live web dashboard for the MDK Fleet multi-agent predictive maintenance system.

## Overview

A Flask-based TUI-style dashboard that reads JSONL event streams and renders:
- **Fleet map** — 50-miner status grid, color-coded by `ok/warn/imm/shut`
- **Fleet KPIs** — live TE and HSI bars (fleet-wide and per-miner)
- **Live flags feed** — streaming flag_raised events from deterministic tools
- **Maestro reasoning trace** — orchestrator_decision events with full reasoning trace
- **Per-miner detail** — telemetry history, TE/HSI sparklines, flags, decisions
- **Decisions log** — searchable orchestrator reasoning trace, filterable by miner/action/autonomy

## Running

```bash
# Make sure you're in the project root
python -m dashboard.main                          # 127.0.0.1:8000
python -m dashboard.main --host 0.0.0.0 --port 9000
MDK_STREAM_DIR=/custom/path python -m dashboard.main
```

Or via Flask app factory:

```python
from dashboard.app import create_app
app = create_app(stream_dir="/run/mdk_fleet/stream/")
app.run()
```

## CSS build step

The stylesheet is derived from `shared/design/tokens.json` — never hand-edited:

```bash
python -m dashboard.build_css
# → dashboard/static/css/dashboard.css
```

Run this whenever `tokens.json` changes.

## URL routes

| Route | Description |
|---|---|
| `/` | Main dashboard: fleet map, KPIs, flags feed, reasoning trace |
| `/miner/<id>` | Per-miner detail: telemetry table, TE/HSI sparklines, flags + decisions |
| `/decisions` | Searchable reasoning trace log |
| `/api/stream/<channel>` | SSE streams: `telemetry`, `kpi`, `flag`, `decision`, `action`, `snapshot`, `live` |
| `/api/fleet` | JSON fleet snapshot (polling alternative to SSE) |
| `/api/flags` | JSON recent flags |

## Architecture

```
dashboard/
├── app.py          Flask app factory + all routes
├── sse.py          SSE generator helpers (tail_jsonl_sse, replay_jsonl_sse)
├── build_css.py    CSS build step — reads tokens.json, emits dashboard.css
├── main.py         CLI entry point
├── templates/
│   ├── base.html   Shared layout (header bar, nav, scripts)
│   ├── index.html  Main dashboard grid
│   ├── miner_detail.html  Per-miner view
│   └── decisions.html     Searchable log
└── static/
    ├── css/dashboard.css   Generated from tokens.json
    └── js/
        ├── event_stream.js  MDKEventStream class + utilities
        └── dashboard.js     Fleet map, flags feed, trace log live updaters
```

## Design tokens

All colors, fonts, and spacing come from `shared/design/tokens.json`.
Zero hardcoded values in templates or semantic CSS classes — everything
goes through CSS custom properties emitted by `build_css.py`.

## Stream channels (SSE)

| Channel | File | Events |
|---|---|---|
| `telemetry` | `telemetry.jsonl` | `telemetry_tick` |
| `kpi` | `kpis.jsonl` | `kpi_update` |
| `snapshot` | `snapshots.jsonl` | `fleet_snapshot` |
| `flag` | `flags.jsonl` | `flag_raised` |
| `decision` | `decisions.jsonl` | `orchestrator_decision` |
| `action` | `actions.jsonl` | `action_taken` |
| `live` | `live.jsonl` | all events |
