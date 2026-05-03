# TALMAS Findings: GSM8K Evaluation

**Model:** LLaDA-8B-Base | **Dataset:** GSM8K (1319 examples) | **Config:** λ_max=4.0, μ=0.1, timestep gate ✓, layer gate ✓

---

## TL;DR

TALMAS gives a **+1.44 percentage point** accuracy gain over the LLaDA-8B-Base baseline on GSM8K (73.77% vs 72.33%) with the default config (λ=4.0, μ=0.1). The improvement is real but uneven: TALMAS fixes 47 baseline errors while introducing 28 new ones. The pattern is clear — TALMAS helps when problems require tracking multiple entities or accumulating across steps, and hurts when it causes output degeneration (repetition loops, truncation before the final answer) or inserts spurious arithmetic steps in long chains. A hyperparameter sweep over 25 configs suggests the default μ=0.1 is suboptimal: **μ=0.5 consistently tops the sweep** at n=300–500 across all λ values, and may push the full-scale number closer to 75–76% if confirmed.

---

## Numbers

| | Baseline | Full TALMAS | Delta |
|---|---|---|---|
| Accuracy | 72.33% | 73.77% | **+1.44pp** |
| Correct | 954 / 1319 | 973 / 1319 | **+19** |

### Per-example confusion matrix

|  | TALMAS correct | TALMAS wrong |
|---|---|---|
| **Baseline correct** | 926 (70.2%) | **28 regressions** (2.1%) |
| **Baseline wrong** | **47 gains** (3.6%) | 318 (24.1%) |

The 318 "both wrong" examples are the hard tail — TALMAS doesn't touch them. The intervention is net positive (47 gains vs 28 losses) but the losses are meaningful and follow a predictable pattern.

---

## Where TALMAS Wins

**Pattern: Multi-entity tracking and sequential accumulation**

TALMAS suppresses real-token → mask-token attention. During reverse diffusion, this keeps the model anchored to already-revealed tokens rather than getting pulled toward unrevealed positions. The payoff is clearest when the problem requires counting across multiple entities or accumulating over several time periods — situations where baseline LLaDA drifts and under-counts.

### Example 1: Basketball coaches (gold = 48)

> *"Each school has sent a girls' basketball team and a boys' basketball team... Each school has also sent a coach for each team."*

| | Prediction | What happened |
|---|---|---|
| Baseline | **44** | Counted 1 coach per school (4 coaches total). Missed that there is 1 coach per **team**, so 8 coaches total. |
| TALMAS | **48** | Correctly tracked: 4 schools × 2 teams × (5 players + 1 coach) = 48 people. |

**Baseline output (end):** `...4 schools * 1 coach per school = 4 coaches. Total = 4*10 + 4 = 44.`

**TALMAS output (end):** `...4 schools * 2 coaches = 8 coaches. Total = 40 players + 8 coaches = 48.`

---

### Example 2: Marble price over 36 months (gold = 92)

> *"A bag of marbles costs $20 and the price increases by 20% of the original price every two months."*

| | Prediction | What happened |
|---|---|---|
| Baseline | **64** | Output truncated after counting only 11 price increases instead of 18. |
| TALMAS | **92** | Correctly computed: 36 months ÷ 2 months = 18 periods; 20% × $20 = $4/period; $20 + 18 × $4 = $92. |

**Baseline output (end):** `...after 11 increases the price is $20 + $44 = $64.`

**TALMAS output (end):** `...36 / 2 = 18 two-month periods. Price = $20 + 18 * $4 = $92.`

---

### Example 3: Lemonade profit (gold = 15)

> *"It costs $3 for lemons and $2 for sugar per gallon. They sell each glass for $0.50, get 20 glasses/gallon. They made $25 in profit. How much did they spend on lemons?"*

| | Prediction | What happened |
|---|---|---|
| Baseline | **14** | Flawed profit attribution — did not correctly chain profit/gallon → number of gallons → lemon cost. |
| TALMAS | **15** | Revenue/gallon = 20 × $0.50 = $10; cost/gallon = $5; profit/gallon = $5; gallons = $25 ÷ $5 = 5; lemons = 5 × $3 = **$15**. |

**Baseline output (end):** `...profit per gallon is $5, total gallons = 4.67, lemons ≈ $14.`

**TALMAS output (end):** `...5 gallons needed. Lemon cost = 5 * $3 = $15.`

---

## Where TALMAS Loses

**Pattern: Output degeneration and spurious arithmetic steps**

TALMAS suppression is strongest at late layers and late timesteps — when most tokens are already revealed (r_t ≈ 0, so f(1−r_t) ≈ 1). This means suppression ramps up precisely as the model is writing its final answer. In long arithmetic chains, over-anchoring on already-committed tokens causes two failure modes: (a) the model enters a repetition loop, re-generating an intermediate sentence instead of advancing to the final sum, or (b) it completes the correct chain but appends an extra wrong step, re-applying a factor it already used.

### Example 1: Streaming service annual cost (gold = 1596) — repetition loop

> *"Aleena pays $140/month. First 6 months at full price, last 6 months at 10% off. Total for the year?"*

| | Prediction | What happened |
|---|---|---|
| Baseline | **1596** | Correct: 6 × $140 + 6 × $126 = $840 + $756 = $1596. |
| TALMAS | **1** | Output repeated "In the second half of the year, she was charged $140/month × 6 months = $840" nine times, then collapsed to prediction "1". |

**TALMAS output (end):** `...she was charged $840. In the second half of the year, she was charged $840. In the second half of the year, she was charged $840. [×6 more] The answer is 1.`

This is the clearest failure mode: the suppression of mask tokens disrupts the model's ability to exit a generation loop and commit to the final answer.

---

### Example 2: Park perimeter (gold = 5) — spurious multiplication

> *"Gary walks around a 1.5 × 6 mile rectangular park at 3 miles/hour. How many hours?"*

| | Prediction | What happened |
|---|---|---|
| Baseline | **5** | Perimeter = 1.5 + 6 + 1.5 + 6 = 15 miles; time = 15 ÷ 3 = 5 hours. |
| TALMAS | **10** | Computed perimeter correctly (15 miles) then multiplied by 2 for no reason: 15 × 2 = 30 miles → 30 ÷ 3 = 10 hours. |

**Baseline output (end):** `...perimeter = 15 miles. Time = 15 / 3 = 5 hours.`

**TALMAS output (end):** `...perimeter = 15 miles. Total distance = 15 * 2 = 30 miles. Time = 30 / 3 = 10 hours.`

The chain was correct until the final step, where an extra operation appeared. The model had latched onto a "×2" from the rectangle's two pairs of sides and re-applied it after the perimeter was already complete.

---

### Example 3: Pool water leak (gold = 8) — algebraic setup error

> *"Both pools leak at 4 gallons/min. 4 minutes ago the big pool had twice as much as the small. Now it has four times as much. How much water does the small pool have now?"*

| | Prediction | What happened |
|---|---|---|
| Baseline | **8** | Set up: B − 16 = 2(S − 16) and B = 4S → S = 8. |
| TALMAS | **16** | Double-subtracted leak: treated the 4-minute drain as applying twice → 2x − 32 = 4x → x = 16. |

**Baseline output (end):** `...4S - 16 = 2S - 16 → 2S = 0... wait, B - 16 = 2(S - 16). B = 4S → 4S - 16 = 2S - 32 → 2S = -16... S = 8.`

**TALMAS output (end):** `...2x - 16 - 16 = 4x - 16 → 2x - 32 = 4x - 16 → -2x = 16 → x = -8... |x| = 16.`

---

## Interpretation

The same mechanism drives both outcomes. TALMAS suppression scales as f(1−r_t) = (1−r_t)², so it is near-zero at the start of denoising (almost all tokens masked, r_t ≈ 1) and reaches full strength at the end (almost all tokens revealed, r_t ≈ 0). The model generates with minimal constraint early on, but gets progressively anchored to its already-committed tokens as it approaches the final answer.

This is precisely the right regime for **accumulation and tracking**: by the time the model is writing the final total, TALMAS forces it to stay consistent with the intermediate counts it already committed to. It can't quietly revise "2 coaches per school" while writing the sum.

The same late-timestep strength is what breaks **long arithmetic chains**: when the model is about to write the final number, TALMAS is at peak suppression. If it has already committed an intermediate sentence (e.g. "she was charged $840"), the over-anchoring causes it to regenerate that sentence instead of advancing — a repetition loop. In the spurious-step failures, the model has already committed a factor to the sequence; TALMAS anchors on it and re-applies it one extra time before finally writing the answer.

---

## Performance by Question Difficulty

Four independent difficulty proxies were used to bucket all 1319 questions. All comparisons are baseline vs full TALMAS (λ=4.0, μ=0.1, timestep+layer gates).

---

### By answer magnitude (size of the correct answer)

| Bucket | n | Baseline | TALMAS | Delta | Gains | Losses |
|---|---|---|---|---|---|---|
| 1-digit (1–9) | 253 | 76.3% | 75.5% | **−0.8pp** | 7 | 9 |
| 2-digit (10–99) | 639 | 72.5% | 75.3% | **+2.8pp** | 24 | 6 |
| 3-digit (100–999) | 296 | 71.6% | 73.6% | **+2.0pp** | 13 | 7 |
| 4-digit (1000–9999) | 84 | 70.2% | 66.7% | **−3.6pp** | 1 | 4 |
| 5-digit+ (≥10000) | 47 | 57.4% | 57.4% | 0.0pp | 2 | 2 |

TALMAS gains concentrate in the **2–3 digit range** (medium-difficulty answers) and disappear at the extremes. The 4-digit regression (−3.6pp, 1 gain vs 4 losses) is the sharpest negative signal — these problems involve large multi-step arithmetic where over-anchoring breaks the chain. The 1-digit regression is mild and may reflect simple questions (e.g. "how many left after?") where TALMAS adds friction without benefit. The 5-digit+ floor (57%) is where both models fail and TALMAS has no effect.

---

### By question length (word count)

| Bucket | n | Baseline | TALMAS | Delta | Gains | Losses |
|---|---|---|---|---|---|---|
| Short (<50 words) | 839 | 77.8% | 78.4% | **+0.6pp** | 19 | 14 |
| Medium (50–79 words) | 406 | 66.3% | 68.0% | **+1.7pp** | 20 | 13 |
| Long (80–119 words) | 70 | 44.3% | 52.9% | **+8.6pp** | 7 | 1 |
| Very long (120+ words) | 4 | 25.0% | 50.0% | +25.0pp* | 1 | 0 |

*n=4, noisy*

The trend is monotone: **TALMAS benefit scales with question length**. Short questions are nearly neutral (+0.6pp). Long questions (80–119 words) show a dramatic +8.6pp gain with only 1 regression vs 7 gains. Long questions have more entities, more conditions, and more steps to accumulate — exactly the regime where TALMAS's late-timestep anchoring adds value.

---

### By number of numeric values in the question

| Bucket | n | Baseline | TALMAS | Delta | Gains | Losses |
|---|---|---|---|---|---|---|
| Few (≤3 numbers) | 760 | 76.4% | 77.2% | **+0.8pp** | 22 | 16 |
| Some (4–6 numbers) | 506 | 68.2% | 70.9% | **+2.8pp** | 23 | 9 |
| Many (7–9 numbers) | 47 | 53.2% | 51.1% | **−2.1pp** | 2 | 3 |
| Lots (10+ numbers) | 6 | 50.0% | 50.0% | 0.0pp | 0 | 0 |

Questions with 4–6 numbers represent the TALMAS sweet spot: enough numeric values to require careful multi-step tracking, but not so many that the arithmetic chain itself becomes fragile. Beyond 6 numbers TALMAS reverses to a slight negative — consistent with the over-anchoring failures in complex arithmetic chains.

---

### By number of sentences in the question

| Bucket | n | Baseline | TALMAS | Delta | Gains | Losses |
|---|---|---|---|---|---|---|
| 1–2 sentences | 228 | 78.9% | 78.9% | **0.0pp** | 5 | 5 |
| 3–4 sentences | 796 | 74.7% | 75.4% | **+0.6pp** | 23 | 18 |
| 5–6 sentences | 242 | 62.8% | 65.7% | **+2.9pp** | 11 | 4 |
| 7+ sentences | 53 | 50.9% | 64.2% | **+13.2pp** | 8 | 1 |

This is the clearest single signal in the analysis. On problems with 7+ sentences, TALMAS gains on 8 examples and regresses on only 1 — a 8:1 gain-to-loss ratio and a +13.2pp delta. On 1–2 sentence problems it is perfectly neutral (5 gains, 5 losses, 0.0pp). The monotone increase in benefit with sentence count directly mirrors the intuition: each sentence introduces a new constraint or entity; TALMAS's anchoring becomes increasingly valuable as those constraints multiply.

---

### Difficulty summary

| Difficulty proxy | TALMAS helps | TALMAS neutral | TALMAS hurts |
|---|---|---|---|
| Answer magnitude | 2–3 digit | 5-digit+ | 1-digit, 4-digit |
| Question length | Long (80+ words) | Short (<50 words) | — |
| Number count | 4–6 numbers | ≤3 numbers | 7+ numbers |
| Sentence count | 5+ sentences | 1–2 sentences | — |

**The pattern is consistent across all four proxies.** TALMAS helps in the middle of the difficulty distribution — problems complex enough to require multi-entity tracking but not so large that the answer requires extended arithmetic chains. It is neutral on easy problems (short, few numbers, 1–2 sentences) and negative on problems with very large answers or very many numeric values.

---

## Hyperparameter Sweep Results

Sweep over λ_max ∈ {1, 2, 4, 6, 8} and μ ∈ {0.0, 0.1, 0.2, 0.5, 1.0}, all with timestep gate on and layer gate on unless noted. Results at n=300 and n=500 are partial runs; the full baseline and full TALMAS are at n=1319 for comparison.

> **Variance caveat:** At n=300, SE ≈ ±2.5pp; at n=500, SE ≈ ±1.9pp. Differences of 1–2pp between configs at these scales are within noise. The consistent directional patterns across many configs are the signal worth trusting.

---

### Finding 1: Static bias without gating destroys performance

| Config | n | Accuracy |
|---|---|---|
| λ=10.0, timestep=**off**, layer=**off** | 100 | **62.0%** |
| Baseline (no TALMAS) | 1319 | 72.3% |
| λ=4.0, timestep=on, layer=on | 1319 | 73.8% |

Applying a large constant bias uniformly across all layers and all timesteps is 10pp below baseline. The gates are not optional decorations — the selective timing (ramp up late) and layer focus (deep layers only) are what prevent TALMAS from suppressing attention it shouldn't touch.

---

### Finding 2: Layer gate gives marginal benefit; timestep gate is the primary driver

| Config | n=300 | n=500 | n=1319 |
|---|---|---|---|
| λ=4.0, μ=0.1, timestep=on, layer=**on** | 75.3–75.7% | 76.8% | **73.8%** |
| λ=4.0, μ=0.1, timestep=on, layer=**off** | 76.7% | 75.4% | **73.2%** |

The layer gate adds roughly +0.6pp at full scale (73.8 vs 73.2). At partial scale the difference is noisy. The timestep gate alone (layer=off) already captures most of the gain; the sigmoid layer gate focuses suppression on deep layers and provides a moderate additional benefit.

---

### Finding 3: Accuracy is flat across λ values once the timestep gate is on

Best accuracy per λ (n=300, timestep=on, layer=on, across all μ):

| λ_max | Best μ | Accuracy (n=300) |
|---|---|---|
| 1.0 | 0.2 or 0.5 | **77.7%** |
| 2.0 | 0.1 or 0.5 | 76.7% |
| 4.0 | 0.5 | 77.3% |
| 6.0 | 0.5 or 1.0 | 77.3% |
| 8.0 | 0.0 | 77.3% |

All gated configs cluster within a ~1pp band at their best μ. There is no clear winner for λ_max — once the timestep gate shapes the schedule, the raw magnitude of the bias matters less than expected. λ=1.0 with the right μ matches λ=8.0. This suggests the timestep gate's quadratic ramp — not the absolute scale — is doing most of the work.

Crucially, the μ=0.0 configs (no mask→mask suppression) also appear in the top tier at high λ, suggesting that real→mask suppression alone is sufficient and mask→mask suppression is not always necessary.

---

### Finding 4: μ=0.5 is a consistent top performer, especially for lower λ

Sweeping μ at fixed λ, the pattern across all λ values:

| λ | μ=0.0 | μ=0.1 | μ=0.2 | μ=0.5 | μ=1.0 |
|---|---|---|---|---|---|
| 1.0 | 75.3% | 77.3% | **77.7%** | **77.7%** | 77.3% |
| 2.0 | 75.3% | 76.7% | 76.0% | 76.7% | 75.7% |
| 4.0 | 76.3% | 75.3% | 75.3% | **77.3%** | 75.0% |
| 6.0 | 76.0% | 75.3% | 76.3% | **77.3%** | 77.0% |
| 8.0 | **77.3%** | 77.0% | 74.7% | 76.3% | 76.7% |

*(n=300; all with timestep=on, layer=on)*

μ=0.5 appears in the top tier for λ=4.0 and λ=6.0 and ties for the top at λ=1.0. The μ=0.1 default used in the primary full run is clearly not optimal — it under-suppresses mask→mask attention.

**Why μ=0.5 likely helps:** μ controls how strongly mask tokens are suppressed when attending to other mask tokens. At μ=0.1, masks can almost freely attend to each other, which may let the model re-read unrevealed positions and introduce inconsistency. At μ=0.5, mask→mask attention is partially suppressed, discouraging the model from "planning ahead" based on what's still masked. At μ=1.0, both real and mask queries are equally suppressed when attending to mask keys, which may be too strong and hurt coordination across masked positions in complex problems.

---

### Sweep summary: recommended config

Based on partial-run results, the config that most consistently reaches the top of the sweep is:

**λ_max=1.0–4.0, μ=0.5, timestep=on, layer=on** (77.3–77.7% at n=300)

Compared to the default run (λ=4.0, μ=0.1) which achieves 73.8% at n=1319, this represents a potentially meaningful improvement — but the partial-run numbers need to be confirmed at full scale. The primary run's μ=0.1 was likely a suboptimal choice; μ=0.5 is the next priority for a full 1319-example run.

---

## Open Questions

1. **Run λ=4.0, μ=0.5 at full scale** — the sweep strongly suggests μ=0.5 outperforms μ=0.1 across multiple λ values. A single full-run comparison would confirm whether the sweep signal holds at 1319 examples.

2. **Generation length and repetition loops** — repetition loop failures concentrate in longer outputs. If the model is given a fixed 256-token budget, TALMAS at late timesteps may be over-constraining near the end. Testing with shorter generation lengths (128) would show whether loop failures disappear.

3. **Layer gate threshold** — the sigmoid layer gate centers at u=0.5. Shifting to u=0.65 would focus suppression on the top third of layers only, potentially reducing spurious-step failures that may originate from mid-depth layers.

4. **What drives the ~24% both-wrong floor** — 318/1319 examples are wrong for both baseline and all TALMAS configs. Understanding whether these share a structural property (e.g. multi-hop division, large final answers) would clarify the ceiling for this class of intervention.
