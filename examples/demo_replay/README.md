# Demo Replay Assets

JSONL event streams from the **canonical API run** of our validation
experiments. See `mdk-orchestra-fullreport` for full experimental context
and reproducibility details.

These files are consumed by:
- `mdk-orchestra replay` — the subcommand that replays them live in
  the dashboard
- The **Demo** option of the `mdk-orchestra` wizard — which wraps
  `replay` with a friendlier UX

Running the demo replays these events at **1× real-time wall-clock speed
by default** (matching the ~12-minute cadence of the original run) against
the real dashboard, with **zero API cost and no LLM required**. Pass
`--speed 4` or higher to `mdk-orchestra replay` for accelerated playback.

## Files

| File | Lines | Purpose |
|---|---:|---|
| `flags.jsonl` | 38 | `flag_raised` events from the rule-engine + XGBoost flaggers |
| `decisions.jsonl` | 28 | `orchestrator_decision` events with Maestro reasoning traces |
| `actions.jsonl` | 28 | `action_taken` events dispatched by the executor |
| `snapshots.jsonl` | 559 | `fleet_snapshot` events for the dashboard's fleet grid |
| `run_stats.txt` | — | Summary counts (flags by type/severity, action mix, etc.) |
| `hero_trace.md` | — | The `m040` chip-instability → L4 human_review trace used in the pitch |
| `ab_results.json` | — | A/B metrics from the source run |

## Source run metadata

- **Run ID:** `ab_short_complete_20260422_0958`
- **Duration (simulated):** 120 min
- **Duration (wall, real):** ~11.5 min at 10× speed
- **Miners:** 50
- **Seed:** 42
- **Fault mix:** balanced
- **Flags consumed:** 38
- **Orchestra decisions:** 28
- **Autonomy distribution:** L1_observe × 13, L2_suggest × 6, L3_bounded_auto × 5, L4_human_only × 4
- **Hero case:** miner `m040`, chip_instability_precursor → L4 `human_review`

## What's NOT included

- `telemetry.jsonl` (34 MB) and `kpis.jsonl` (29 MB) — too large for the
  repo. The dashboard works fine without them during replay; the fleet
  grid reads from `snapshots.jsonl`.
- `live.jsonl` (66 MB) — derived log, not needed for replay.

If you want the full raw dataset, pull `ab_canonical_run.tar.gz` from
the `mdk-orchestra-fullreport` repo.
