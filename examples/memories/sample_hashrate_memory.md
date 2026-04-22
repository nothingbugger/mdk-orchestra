# Hashrate Memory

*Hashrate-domain patterns learned from past consultations. Written by
Maestro during curation cycles. Read by the hashrate specialist at
every invocation.*

*Pattern format: each entry lives under a `## Pattern: <snake_case>`
heading with a fixed field block. The hashrate agent reads these
entries as prior lessons and may cite them in its reasoning.*

---

## Pattern: short_window_variance_false_positive
- First seen: 2026-04-21T20:47:47.820280+00:00
- Last seen: 2026-04-21T20:47:47.820280+00:00
- Occurrences: 1
- Signature: XGBoost predicts -20% hashrate drop in 30min but 30-min realized trajectory is stationary within ±2% of nominal, HSI ≥99, no step/ramp/sawtooth, temp and power steady. The model fired on 1-min window variance or momentary low sample while the sustained window shows nothing above noise floor.
- Learned verdict/action: noise
- Confidence: 0.87
- Reasoning: The predictor's 1-min feature sensitivity generates false positives when the 30-min trajectory is flat. Check: is mean within ±2% of nominal? Is σ consistent with this miner's baseline variance? Is HSI >99 with no hot_time_frac? If yes to all, classify as predictor false positive with high confidence — acting on it would throttle a healthy miner on model artifact alone.
- Example reference: dec_6d5fc2d09f4c

## Pattern: eco_mode_subnominal_offset_fp
- First seen: 2026-04-21T20:47:47.823384+00:00
- Last seen: 2026-04-21T20:47:47.823384+00:00
- Occurrences: 1
- Signature: Miner runs sustained 1-1.5% below nameplate (eco-mode baseline) with stationary band, no ramp/step, temp/power/voltage decoupled. XGBoost trips on the sustained sub-nominal offset combined with normal power variance, but the offset is the miner's operating baseline, not degradation.
- Learned verdict/action: noise
- Confidence: 0.82
- Reasoning: Eco-mode or de-rated miners produce a persistent sub-nominal hashrate that the predictor can misread as early degradation. Distinguish from real decay by: absence of slope, HSI high, thermal decoupling, and the offset matching the miner's known operating profile. A sustained −1% that is stationary ≠ a −20% predicted drop.
- Example reference: dec_02ccbb42fb30
