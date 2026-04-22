# Pitch canonical run — hero trace

**Build ID:** `ab_short_complete_20260422_0958`

**Run dir:** `/tmp/mdk_pitch_run/ab_short_complete_20260422_0958_23e965fb/`


## Miner & flag

- **Miner:** `m040`
- **Flag type:** `chip_instability_precursor` (severity `warn`)
- **Source tool:** `xgboost_rolling_variance`
- **Confidence:** 0.731
- **Top feature:** `hashrate_th_30m_std` = 10.5177

### Full flag envelope

```json
{
  "flag_id": "flg_00027",
  "miner_id": "m040",
  "flag_type": "chip_instability_precursor",
  "severity": "warn",
  "confidence": 0.731,
  "source_tool": "xgboost_rolling_variance",
  "evidence": {
    "metric": "hashrate_th_30m_std",
    "window_min": 30.0,
    "predicted_prob": 0.731,
    "top_feature_name": "hashrate_th_30m_std",
    "top_feature_value": 10.5177,
    "features": [
      "hashrate_th_30m_std",
      "power_w_30m_std",
      "hashrate_th_5m_std",
      "temp_chip_c_30m_std",
      "power_w_5m_std",
      "hashrate_th_1m_std",
      "power_w_1m_std",
      "temp_chip_c_5m_std",
      "fan_rpm_mean_30m_std",
      "hashrate_th_1m_mean",
      "hashrate_th_30m_mean",
      "temp_chip_c_1m_mean",
      "power_w_1m_mean",
      "fan_rpm_mean_1m_std"
    ],
    "feature_values": [
      10.5177,
      341.6098,
      15.9143,
      4.9522,
      519.4683,
      17.954,
      580.9585,
      7.5591,
      122.5403,
      90.3342,
      104.8642,
      67.1125,
      2923.75,
      132.858
    ]
  },
  "raw_score": 0.731
}
```

## Specialist consultation

- **Agent:** `hashrate_agent`
- **Assessment:** `real_signal`  ·  confidence: 0.91
- **Cost:** $0.1007  ·  latency: 14.8s

## Maestro synthesis

- **Action:** `human_review`  ·  **Autonomy:** `L4_human_only`
- **Confidence:** 0.90
- **Cost total:** $0.1641  ·  **Latency total:** 30.8s
- **Consulted:** voltage_agent
- **Pending human approval:** True

### Reasoning trace

> m040 chip_instability_precursor warn/0.73 (xgboost_rolling_variance, top feature hashrate_th_30m_std=10.52). Hashrate specialist: real_signal conf=0.91, severity upgraded warn→crit. Bilateral oscillation 43.9–116.2 TH/s vs 115 TH/s nameplate, 30m mean ~84 TH/s (−26.9%); chip temp and power co-oscillate in lockstep with hashrate (43.9 TH/s@46°C ↔ 92.7 TH/s@65°C), ruling out thermal sawtooth or pool stall — this is chip substrate intermittent dropout. HSI collapsed from ~92 to 69–74; sigma_hash ~17 TH/s is ~10x normal envelope. Unambiguous verdict, no fallback warranted. Throttle contraindicated: reducing target hashrate won't arrest intermittent substrate failure and may mask progression. Shutdown is the correct protective action but is L4-mandatory per autonomy ladder (never auto-shutdown). Queuing for human approval with high-urgency alert; operator should expect possible hashboard retirement. Confirming first-pass Sonnet decision.
