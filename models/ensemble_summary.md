# Detection Ensemble Summary

MDK Fleet ships a **conservative-by-default detection pipeline**: only
flaggers with documented OOD validation metrics (or hard-physics rules)
are ACTIVE at startup. Classifiers without validation — or those whose
validation revealed under-sampled training populations — stay in the
repo as deliverables but are opted-out at runtime. The pkl artefacts
are preserved for future re-activation once the validation gate is
passed.

This discipline matters because Maestro spends real API budget on each
flag it receives, especially for `anomaly_composite` (Flow 3, all four
specialists). Shipping an unvalidated detector into that path would
burn credits on signals whose quality nobody has measured.

## Active / Inactive status

| Flagger | Status at startup | Flag type(s) | Opt-in CLI flag |
|---|---|---|---|
| `RuleEngineFlagger` | **ACTIVE** | voltage_drift, hashrate_degradation, thermal_runaway, fan_anomaly | — |
| `XGBoostFlagger` (canonical hashrate) | **ACTIVE** | hashrate_degradation | — |
| `ChipInstabilityFlagger` | **ACTIVE** | chip_instability_precursor | — |
| `HashboardFailureFlagger` | **INACTIVE** | hashboard_failure_precursor | `--enable-hashboard-failure` |
| `IsolationForestFlagger` | **INACTIVE** | anomaly_composite | `--enable-isolation-forest` |

Both inactive flaggers keep their pkl / metrics files in `models/` so a
future run (with more data or a dedicated validation study) can flip
the default back to enabled.

## Why IsolationForestFlagger is deactivated

- **Zero flags emitted** across three real runs — `realistic_b3b723d8`
  (A/B seed 42), `realistic_550baf33` (A/B seed 43), and
  `rich_run_20260421_1650` (30-min balanced-mix). In every run, 100%
  of raised flags came from `rule_engine` or `xgboost_predictor`.
- **No validation metrics at all.** Unlike the three XGBoost models,
  IF has no ROC AUC, no PR AUC, no precision-at-top-k. Only protocol
  tests (load, save, init) exist in `tests/test_detector_protocol.py`;
  they verify nothing about detection quality.
- **Dispatch cost risk.** If IF did fire, its `anomaly_composite` flag
  maps in Maestro's dispatch table to **all four specialists**
  (`voltage`, `hashrate`, `environment`, `power`) — roughly
  $0.35–0.50 per decision. Firing an unvalidated classifier into the
  most expensive dispatch path is the wrong default.
- **Training lineage unclear.** The pkl (`models/if_v2.pkl`) was
  bootstrapped from a live stream rather than a controlled dataset,
  so we cannot reproduce the exact training population post-hoc.

### What it takes to re-activate IsolationForest

1. A dedicated **validation study**: hold out telemetry windows with
   known fault labels, score the IF anomaly score against ground truth,
   compute ROC AUC + precision-at-top-k, and compare across sensitivity
   profiles (low / medium / high).
2. Tune `anomaly_score_warn` / `anomaly_score_crit` thresholds based
   on the ROC curve rather than the arbitrary 0.70 / 0.88 defaults.
3. Verify that enabling IF does not flood the Maestro dispatch — run a
   30-min smoke and count emitted flags per minute.
4. If all three pass, flip the `runner.py` default to
   `disable_isolation_forest: bool = False` and document the
   validation in a `models/if_v2_metrics.json`.

## Why HashboardFailureFlagger is deactivated

```bash
# To re-enable when you have more data:
python -m deterministic_tools.main --enable-hashboard-failure
```

### Why hashboard_failure is deactivated

- Cross-miner OOD AUC **0.745**, under the 0.80 target
- Pre-fault samples came from only **2 miners** (m019, m036) — too
  narrow to claim OOD generalization
- Feature importance of the trained model fell on raw hashrate mean
  instead of the `temp_per_power` / `voltage_per_power` signal that
  the pattern discovery had identified — a sign that the thermal-
  electrical signature doesn't yet generalize across miners

### What it takes to re-activate

1. Run the simulator for **≥12 hours** simulated, balanced fault-mix,
   more miners if possible. Target: **≥5 distinct miners** with a
   hashboard_failure episode so a proper miner-wise 80/20 split has
   positives on both sides.
2. Regenerate `pattern_discovery/features.parquet`
3. Re-run `scripts/train_xgb_ensemble.py` — will retrain both
   pattern-specific models
4. Inspect the new OOD AUC. If it crosses 0.80 with the top features
   still belonging to the thermal/electrical ratios (not raw
   hashrate), flip the default to enabled in `runner.py`.

## Comparative Metrics

| Model | Pattern Name | Flag Type | AUC | PR-AUC | P@10% | R@10% | n_train | n_test_positives | Top-3 Features |
|---|---|---|---|---|---|---|---|---|---|
| xgb_predictor (canonical) | hashrate_degradation_canonical | hashrate_degradation | 0.9283 | 0.8368 | 0.9968 | 0.4984 | 100,360 | 5,018 | power_mean_1m, fan_mean, hr_mean_1m |
| xgb_chip_instability | rolling_variance_spike | chip_instability_precursor | 0.8343 | 0.8191 | 0.6508 | 0.1139 | 97,442 | 360 | power_w_30m_std, fan_rpm_mean_30m_std, power_w_5m_std |
| xgb_hashboard_failure | thermal_electrical_decoupling | hashboard_failure_precursor | 0.7453 | 0.8045 | 0.8852 | 0.1500 | 97,082 | 360 | hashrate_th_1m_mean, hashrate_th, temp_per_power |

## Targets vs Actuals

| Model | AUC Target | AUC Actual | AUC Met? | P@10% Target | P@10% Actual | P@10% Met? |
|---|---|---|---|---|---|---|
| xgb_chip_instability | ≥0.92 | 0.8343 | NO | ≥0.90 | 0.6508 | NO |
| xgb_hashboard_failure | ≥0.80 | 0.7453 | NO | ≥0.70 | 0.8852 | YES |

## Honest Assessment

**xgb_chip_instability** missed targets: AUC=0.8343 (target 0.92), P@10%=0.6508 (target 0.9). Cause: the pattern has weaker Cohen's d (rolling_variance_spike), fewer pre_fault samples in train/test, and the XGBoost signal may not generalize perfectly across held-out miners.

**xgb_hashboard_failure** missed targets: AUC=0.7453 (target 0.8). Cause: the pattern has weaker Cohen's d (thermal_electrical_decoupling), fewer pre_fault samples in train/test, and the XGBoost signal may not generalize perfectly across held-out miners.

### Sample Size Reality Check — chip_instability

Fault miners (all): `['m005', 'm020', 'm038']`
  - In strict train side (m001-m040): `['m005', 'm020', 'm038']`
  - In strict test side (m041-m050): `[]`
  - Strict test positives: **0** (ALL fault miners on train side — evaluation used internal leave-one-miner-out hold-out)

Pre_fault ticks by miner on **train** side: `{'m005': 360, 'm020': 360, 'm038': 360}`
Pre_fault ticks by miner on **test** side (strict): `{}`

Internal eval held-out miner: `m038` (360 positive eval ticks used for metric estimation).

### Sample Size Reality Check — hashboard_failure

Fault miners (all): `['m019', 'm036']`
  - In strict train side (m001-m040): `['m019', 'm036']`
  - In strict test side (m041-m050): `[]`
  - Strict test positives: **0** (ALL fault miners on train side — evaluation used internal leave-one-miner-out hold-out)

Pre_fault ticks by miner on **train** side: `{'m019': 360, 'm036': 360}`
Pre_fault ticks by miner on **test** side (strict): `{}`

Internal eval held-out miner: `m036` (360 positive eval ticks used for metric estimation).
