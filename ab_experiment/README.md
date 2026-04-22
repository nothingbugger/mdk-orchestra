# ab_experiment — A/B Validation Module

Validates whether the multi-agent fleet layer adds measurable value over
deterministic tools alone, by running both configurations against identical
fault scenarios.

## Quick start

```bash
# Mock smoke (1 min simulated = 6 real seconds, no API key needed)
MDK_AGENT_MOCK=1 python -m ab_experiment.runner --scenario smoke --duration-min 1

# Full 60-min run (real Claude API, requires ANTHROPIC_API_KEY)
python -m ab_experiment.runner --scenario full --duration-min 60 --api-mode

# Custom seed and output
python -m ab_experiment.runner --scenario prod --seed 1234 --n-miners 50 \
    --duration-min 60 --output /tmp/ab_runs/
```

## Output

Each run creates a directory under `--output/<run_id>/`:

```
<run_id>/
  ab_results.json          # structured metrics (machine-readable)
  summary.md               # narrative comparison
  figures/
    action_timeline.png    # action counts per track
    catch_rate.png         # faults caught A vs B
    cost_vs_catch.png      # cost vs catch-rate scatter
    reasoning_trace.png    # example reasoning trace card (Track A)
  track_a/
    stream/                # all JSONL event streams for Track A
    logs/                  # subprocess logs per component
  track_b/
    stream/                # all JSONL event streams for Track B
    logs/
```

## Design

**Track A — Fleet ON**: simulator → ingest → deterministic_tools → Maestro → action executor.
Full multi-agent stack. LLM consultation per flag.

**Track B — Deterministic only**: same simulator seed → same deterministic_tools flags → rule-based
action mapping (no Maestro):
- `info` severity → `observe`
- `warn` severity → `alert_operator`
- `crit` severity → `throttle` (80%, 60 min)

Both tracks share the same simulator seed (same fault injection schedule) but run as
independent subprocesses with separate stream directories. Fault scenarios are identical;
only the action strategy differs.

## Metrics computed

1. **Detection**: faults injected, caught A, caught B (catch = action taken within 1h pre-fault window)
2. **Action profile**: counts by action type
3. **False positives**: actions on miners with no fault within 24h
4. **Cost**: Track A Claude API tokens + USD; Track B = $0
5. **Latency**: median flag → action time
6. **Qualitative**: reasoning trace snapshots from Track A

## Public API

```python
from ab_experiment.runner import run_ab_experiment, ABResults

results: ABResults = run_ab_experiment(
    scenario="full",
    duration_min=60,
    output_dir="/tmp/ab_runs/",
    seed=42,
)
print(results.run_a_faults_caught_pre, results.run_b_faults_caught_pre)
```

## Module structure

| File | Purpose |
|---|---|
| `runner.py` | `run_ab_experiment()` + `ABResults` dataclass — public API |
| `metrics.py` | Post-hoc metric computation from JSONL event logs |
| `report_ab.py` | Write `ab_results.json` + markdown + 4 figures |
| `track_b.py` | Rule-based action mapper logic |
| `track_b_runner.py` | CLI entry point for Track B mapper subprocess |
| `fleet_adapter.py` | `FleetStateAdapter` — connects executor to in-process FleetState |
| `main.py` | CLI entry point (`python -m ab_experiment.runner`) |
