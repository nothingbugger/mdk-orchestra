# Dashboard — Module Notes

**Branch:** `feat/dashboard`
**Session:** Claude Sonnet 4.6 (dashboard subagent)
**Approx token spend:** ~75k

## What was built

### Files created

| File | Purpose |
|---|---|
| `dashboard/app.py` | Flask app factory (`create_app`) + all routes |
| `dashboard/sse.py` | SSE generator helpers: `tail_jsonl_sse`, `replay_jsonl_sse` |
| `dashboard/build_css.py` | CSS build step: reads tokens.json → emits dashboard.css |
| `dashboard/main.py` | CLI entry point (`python -m dashboard.main`) |
| `dashboard/templates/base.html` | Shared layout: header bar, nav, live clock |
| `dashboard/templates/index.html` | Main dashboard: fleet map + KPIs + flags + trace |
| `dashboard/templates/miner_detail.html` | Per-miner: telemetry table + sparklines + flags + decisions |
| `dashboard/templates/decisions.html` | Searchable orchestrator reasoning trace log |
| `dashboard/static/css/dashboard.css` | Generated CSS (committed, derived from tokens.json) |
| `dashboard/static/js/event_stream.js` | `MDKEventStream` class + shared utilities |
| `dashboard/static/js/dashboard.js` | `FleetMap`, `FlagsFeed`, `TraceFeed`, `EnvStrip`, `Sparkline` |
| `tests/test_dashboard_app.py` | 34 tests: factory, routes, SSE, CSS checks |

### Architecture decisions

- **No Redis / no WebSocket** — SSE on plain JSONL tails, consistent with event_bus.py design.
- **Server-side initial render** — Flask reads recent JSONL history and renders it via Jinja on first page load, so the dashboard is useful even with JS disabled or before SSE connects.
- **Client SSE for live updates** — three `MDKEventStream` instances on the main page (snapshot, flag, decision channels) patch the DOM incrementally.
- **CSS build step** — `build_css.py` reads `tokens.json`, emits `:root { --var: value; }` for every token, then writes semantic classes using only `var(--...)` references. Zero hardcoded hex in semantic rules.
- **Replay then follow** — on SSE connect, `replay_jsonl_sse` first sends the last N lines (for context), then `tail_jsonl_sse` takes over.

## Deviations from spec

1. **`/api/stream/snapshot`** added — not in interfaces.md §5 but required by the live fleet map. The spec listed `telemetry|kpi|flag|decision` as channels; `snapshot` and `action` were added to support the main dashboard and completeness.

2. **`/api/fleet` and `/api/flags`** JSON poll endpoints added — useful for clients that prefer polling over SSE (e.g., health-check scripts, the A/B experiment runner wanting fleet state without a stream).

3. **`from_start` SSE param** — optional `?from_start=true` query param added for debugging/replay. Not in spec, no conflict.

4. **miner_detail.html uses inline `<script>` for sparkline data injection** — the per-miner KPI sparkline is seeded from server-side data via Jinja `tojson`. This is safe (output-escaped) and avoids a separate API call.

## Token additions to tokens.json / tokens.py

None. All required tokens were present in the existing token files.

## Runnable command

```bash
python -m dashboard.main
# → serving on http://127.0.0.1:8000
```

Or with stream directory override:

```bash
MDK_STREAM_DIR=events/ python -m dashboard.main
```

## Test status

```
34 passed in 0.31s
```

All routes tested: 200/404 correctness, fixture data rendering, SSE content-type,
SSE generator correctness (empty file, limit, malformed JSON skip),
CSS build step, CSS var-only semantic classes, API JSON responses.
