# Maestro Memory

*Agent-curated cross-domain patterns. Written by Maestro during curation
cycles (every 30 simulated minutes). Read by Maestro itself at every
dispatch as operational context.*

*Pattern format: each entry lives under a `## Pattern: <snake_case>`
heading with a fixed field block. Do not edit the structure manually —
the curator rewrites this file in place.*

---

## Pattern: xgboost_hashrate_fp_unanimous_noise
- First seen: 2026-04-21T20:47:47.817543+00:00
- Last seen: 2026-04-21T20:47:47.817543+00:00
- Occurrences: 1
- Signature: XGBoost hashrate_degradation flag (often crit severity, p_drop 0.5–0.99) where both required specialists (hashrate, voltage) return noise at confidence ≥0.80 with tight agreement (stddev <0.05). Telemetry shows miner running at or above nominal with flat thermals and clean PSU — model tripped on 1-min window variance that doesn't persist at 30-min.
- Learned verdict/action: L1_observe
- Confidence: 0.85
- Reasoning: When both required specialists unanimously call noise above 0.80 on an XGBoost hashrate flag, the deterministic model is almost certainly firing on short-window feature noise. Do NOT escalate to L2 despite crit severity — that path leads to alert fatigue. Downgrade to L1 observe, log for predictor calibration. Only escalate if the flag re-fires on the same miner within the next cycle or telemetry actually begins to slope within the prediction window.
- Example reference: dec_758229dd1506
