# Event Schemas — MDK Fleet

**This file is the data contract between every module in MDK Fleet.**
If a field is not in this file, it does not exist.
If a field is in this file, every producer must emit it and every consumer must handle it.

## Transport

All events are **JSON objects, one per line (JSONL)**, written to disk or streamed over an HTTP SSE channel.

Base path for live event streams: `/run/mdk_fleet/stream/`
Base path for persistent event logs: `/run/mdk_fleet/log/`

Every event has a common envelope:

```json
{
  "event": "<event_name>",
  "ts": "2026-04-20T15:30:45.123Z",
  "source": "<producer_module>",
  "data": { ... }
}
```

- `ts` is ISO 8601 UTC with millisecond precision
- `source` is one of: `simulator`, `ingest`, `detector`, `orchestrator`, `voltage_agent`, `hashrate_agent`, `environment_agent`, `power_agent`, `action`
- `data` is event-specific payload, defined below

---

## telemetry_tick

**Producer**: `simulator`
**Consumers**: `ingest`, `dashboard`
**Frequency**: every 5 seconds per miner (configurable)

```json
{
  "event": "telemetry_tick",
  "ts": "2026-04-20T15:30:45.123Z",
  "source": "simulator",
  "data": {
    "miner_id": "m042",
    "miner_model": "S19j Pro",
    "hashrate_th": 95.3,
    "hashrate_expected_th": 104.0,
    "temp_chip_c": 78.5,
    "temp_amb_c": 24.1,
    "power_w": 3250,
    "voltage_v": 12.08,
    "fan_rpm": [5800, 5820, 5790, 5810],
    "operating_mode": "balanced",
    "uptime_s": 432150,
    "env": {
      "site_temp_c": 23.5,
      "site_humidity_pct": 42,
      "elec_price_usd_kwh": 0.072,
      "hashprice_usd_per_th_day": 0.058
    },
    "fault_injected": null
  }
}
```

- `fault_injected`: `null` during normal operation. During a fault it is a string identifying the fault type (e.g., `"chip_instability"`, `"cooling_degradation"`, `"power_sag"`, `"hashboard_failure"`). **Only the simulator knows ground truth** — other modules must never read this field except for A/B labeling.
- `operating_mode`: one of `turbo`, `balanced`, `eco`
- `fan_rpm`: list of 4 fans (ASIC standard); individual fan failure detectable

---

## kpi_update

**Producer**: `ingest`
**Consumers**: `dashboard`, `deterministic_tools`
**Frequency**: computed on every `telemetry_tick` → emitted at same rate

```json
{
  "event": "kpi_update",
  "ts": "2026-04-20T15:30:45.123Z",
  "source": "ingest",
  "data": {
    "miner_id": "m042",
    "te": 47.3,
    "hsi": 82.1,
    "te_components": {
      "value_usd_day": 5.53,
      "cost_usd_day": 11.70,
      "h_eff_th": 95.3,
      "p_hashprice": 0.058,
      "p_asic_w": 3250,
      "p_cool_w": 520,
      "rho_elec": 0.072,
      "sigma_hash": 0.21,
      "alpha": 0.5
    },
    "hsi_components": {
      "thermal_stress": 0.18,
      "hashrate_variability": 0.11,
      "hot_time_frac": 0.05
    }
  }
}
```

---

## fleet_snapshot

**Producer**: `ingest`
**Consumers**: `dashboard`
**Frequency**: every 1 second (dashboard-only aggregate)

```json
{
  "event": "fleet_snapshot",
  "ts": "2026-04-20T15:30:45.123Z",
  "source": "ingest",
  "data": {
    "miner_count": 50,
    "miners": {
      "m001": { "te": 48.2, "hsi": 85.1, "status": "ok",   "hashrate_th": 102.1, "temp_chip_c": 76.2 },
      "m002": { "te": 52.0, "hsi": 79.3, "status": "warn", "hashrate_th":  98.5, "temp_chip_c": 82.1 }
    },
    "fleet_te": 49.8,
    "fleet_hsi": 81.6,
    "env": {
      "site_temp_c": 23.5,
      "site_humidity_pct": 42,
      "elec_price_usd_kwh": 0.072,
      "hashprice_usd_per_th_day": 0.058
    }
  }
}
```

- `status`: one of `ok`, `warn`, `imm`, `shut` (see `shared/design/tokens.json` → `semantic.miner_status`)
- `miners` is a dict keyed by miner_id for O(1) dashboard updates

---

## flag_raised

**Producer**: `deterministic_tools`
**Consumers**: `orchestrator`, `dashboard`
**Frequency**: whenever a flagger raises a pre-failure signal

```json
{
  "event": "flag_raised",
  "ts": "2026-04-20T15:30:45.123Z",
  "source": "detector",
  "data": {
    "flag_id": "flg_00237",
    "miner_id": "m042",
    "flag_type": "voltage_drift",
    "severity": "warn",
    "confidence": 0.71,
    "source_tool": "xgboost_predictor",
    "evidence": {
      "window_min": 30,
      "metric": "voltage_v",
      "recent_mean": 11.83,
      "miner_baseline_mean": 12.05,
      "miner_baseline_std": 0.04,
      "z_score": -5.5
    },
    "raw_score": 0.82
  }
}
```

Possible values:
- `flag_type`: `voltage_drift`, `hashrate_degradation`, `thermal_runaway`, `fan_anomaly`, `power_instability`, `chip_variance_high`, `anomaly_composite`
- `severity`: `info`, `warn`, `crit`
- `source_tool`: `xgboost_predictor`, `isolation_forest_v2`, `rule_engine`

---

## reasoning_request

**Producer**: `orchestrator`
**Consumer**: specific sub-agent (`voltage_agent`, `hashrate_agent`, etc.)
**Frequency**: on each sub-agent dispatch

```json
{
  "event": "reasoning_request",
  "ts": "2026-04-20T15:30:46.511Z",
  "source": "orchestrator",
  "data": {
    "request_id": "req_00891",
    "flag_id": "flg_00237",
    "target_agent": "voltage_agent",
    "miner_id": "m042",
    "question": "Assess whether the voltage drift is a real pre-failure signal or noise, given the miner's own history.",
    "context": {
      "flag": { /* the original flag_raised.data object */ },
      "miner_history_voltage_7d": [ /* 7 days of 5-min voltage samples */ ],
      "miner_recent_telemetry_30min": [ /* last 30 min of full telemetry */ ],
      "similar_past_events": [ /* top 3 from agent's episodic memory, if any */ ]
    }
  }
}
```

---

## reasoning_response

**Producer**: sub-agent
**Consumer**: `orchestrator`
**Frequency**: one per `reasoning_request`

```json
{
  "event": "reasoning_response",
  "ts": "2026-04-20T15:30:48.902Z",
  "source": "voltage_agent",
  "data": {
    "request_id": "req_00891",
    "miner_id": "m042",
    "assessment": "real_signal",
    "confidence": 0.84,
    "severity_estimate": "warn",
    "reasoning": "Voltage has drifted from 12.05 (miner's 7-day baseline) to 11.83 (-5.5σ). In the past 7 days this miner has had 0 similar excursions, so this is not noise. The drift magnitude is consistent with a failing capacitor on the PSU side, based on a matching pattern in episodic memory (event evt_00182, 6 days ago, same miner, preceded a power_sag within 4h).",
    "recommended_action_hint": "throttle",
    "cost_usd": 0.0023,
    "model_used": "claude-sonnet-4-6",
    "latency_ms": 2180
  }
}
```

- `assessment`: `real_signal`, `noise`, `inconclusive`
- `severity_estimate`: `info`, `warn`, `crit`
- `recommended_action_hint`: `observe`, `alert_operator`, `throttle`, `migrate_workload`, `schedule_maintenance`, `human_review`, `shutdown`
- `cost_usd`, `model_used`, `latency_ms` are operational metadata for A/B analysis

---

## orchestrator_decision

**Producer**: `orchestrator`
**Consumers**: `action`, `dashboard`, memory (persistent log)
**Frequency**: one per flag, after all sub-agent responses are aggregated (or timed out)

```json
{
  "event": "orchestrator_decision",
  "ts": "2026-04-20T15:30:51.337Z",
  "source": "orchestrator",
  "data": {
    "decision_id": "dec_00714",
    "flag_id": "flg_00237",
    "miner_id": "m042",
    "action": "throttle",
    "action_params": { "target_hashrate_pct": 0.80, "duration_min": 60 },
    "autonomy_level": "L3_bounded_auto",
    "confidence": 0.78,
    "reasoning_trace": "Flag was voltage_drift severity=warn confidence=0.71. Dispatched to voltage_agent and power_agent. Voltage agent: real_signal conf=0.84, matches past event evt_00182 which preceded power_sag within 4h. Power agent: site supply stable, no upstream issue. Combined: high likelihood this miner's PSU is degrading. Throttling to 80% reduces thermal stress on capacitors, buys time for scheduled maintenance. Reversible, rate-limited, L3 approved.",
    "consulted_agents": ["voltage_agent", "power_agent"],
    "total_cost_usd": 0.0184,
    "total_latency_ms": 4820
  }
}
```

- `autonomy_level`: `L1_observe`, `L2_suggest`, `L3_bounded_auto`, `L4_human_only`
- `action`: matches sub-agent `recommended_action_hint` vocabulary
- `action_params`: action-specific parameters
- L4 actions are emitted as `orchestrator_decision` but action module only queues them for operator approval; they don't auto-execute

---

## action_taken

**Producer**: `action`
**Consumers**: `dashboard`, memory
**Frequency**: one per `orchestrator_decision` that triggered an execution (L3) or one per operator approval (L4)

```json
{
  "event": "action_taken",
  "ts": "2026-04-20T15:30:51.889Z",
  "source": "action",
  "data": {
    "action_id": "act_00301",
    "decision_id": "dec_00714",
    "miner_id": "m042",
    "action": "throttle",
    "status": "executed",
    "outcome_expected": "hashrate drops to 80%, temp_chip drops 5-8°C, voltage stabilizes within 10 min",
    "outcome_observed": null,
    "rollback_ts_scheduled": "2026-04-20T16:30:51.889Z"
  }
}
```

- `status`: `executed`, `queued_for_human`, `rejected`, `failed`
- `outcome_observed` is filled in later by a background loop that compares reality to expectation

---

## episodic_memory_write

**Producer**: any sub-agent (via memory manager)
**Consumer**: persistent storage only (not emitted over stream)
**Frequency**: one per sub-agent response (append to that agent's event log)

```json
{
  "event": "episodic_memory_write",
  "ts": "2026-04-20T15:30:48.902Z",
  "source": "voltage_agent",
  "data": {
    "memory_id": "mem_voltage_00451",
    "miner_id": "m042",
    "trigger_flag_id": "flg_00237",
    "request_id": "req_00891",
    "snapshot": { /* condensed flag context */ },
    "assessment": "real_signal",
    "reasoning": "Voltage has drifted ...",
    "outcome_followup": null
  }
}
```

Each agent has its own JSONL log: `/run/mdk_fleet/memory/<agent_name>/events.jsonl`.
`outcome_followup` is set later if the episode is followed by a confirming outcome (fault / successful intervention / false alarm), enabling retrospective learning.

---

## Event stream files (by default)

| Stream | File |
|---|---|
| All events (live) | `/run/mdk_fleet/stream/live.jsonl` |
| Telemetry only | `/run/mdk_fleet/stream/telemetry.jsonl` |
| KPIs only | `/run/mdk_fleet/stream/kpis.jsonl` |
| Flags | `/run/mdk_fleet/stream/flags.jsonl` |
| Orchestrator decisions | `/run/mdk_fleet/stream/decisions.jsonl` |
| Per-agent episodic | `/run/mdk_fleet/memory/<agent>/events.jsonl` |

A/B experiment runs use prefix `/run/mdk_fleet/ab_run_<timestamp>/` instead of the live path.

## Changes to this file

This is a contract. **Do not change it unilaterally.** Propose changes via issue or PR description and wait for explicit approval.
