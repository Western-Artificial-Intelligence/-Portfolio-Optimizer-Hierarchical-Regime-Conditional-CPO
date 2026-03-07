# SP_Mirror: Architecture & Robustness Audit

## 1. System Architecture

SP_Mirror is a derivative-overlay extension that tracks the S&P 500 using Canadian-listed equities. It operates as a 3-layer system where each layer addresses a distinct source of tracking error.

```
+-------------------------------------------------------------+
|  Layer 3 - AI Supervisor (optional)                         |
|  Regime-aware allocation dial / cash blend                  |
+--------------------------+----------------------------------+
|  Layer 2 - Derivative Overlay + FX Hedge                    |
|  ES1 futures overlay to close beta/vol gap vs benchmark     |
|  Spot USDCAD hedge to neutralize currency exposure          |
+--------------------------+----------------------------------+
|  Layer 1 - Canadian Clone (QP Worker)                       |
|  Monthly rolling QP: min tracking error vs SPY              |
|  Universe: ~32 Canadian equities across all GICS sectors    |
+-------------------------------------------------------------+
```

### Layer 1 - Canadian Clone

A long-only portfolio of ~32 Canadian equities, optimized monthly via quadratic programming over a 5-year rolling window. The objective is to minimize tracking error vs the S&P 500. The clone achieves beta ~0.55-0.70 vs the benchmark, with annualized tracking error around 10%.

### Layer 2 - Derivative Overlay

A calibrated S&P 500 futures position (ES1) scales the clone's effective exposure to match the benchmark. The scaling factor `alpha` determines how much extra futures exposure to add. When `alpha = 1.2`, the portfolio holds 20% additional S&P futures notional. USD exposure from the overlay is hedged via spot USDCAD.

Daily return decomposition:

```
SP_mirror_return = clone_return + (alpha - 1) * r_ES - max(0, alpha - 1) * r_USDCAD - cost
```

### Layer 3 - Supervisor (optional)

An XGBoost meta-labeling classifier that scales the overlay based on macro conditions. Outputs probability P in [0, 1]; overlay is scaled by P in "dial" mode.

---

## 2. Alpha Estimation Modes

SP_Mirror supports four overlay sizing modes:

| Mode | How alpha is determined | Parameters |
|------|------------------------|------------|
| `beta` | `alpha = shrinkage * (1/rolling_beta) + (1-shrinkage) * 1.0`, clipped to bounds | L, bounds, shrinkage_w |
| `vol` | `alpha = shrinkage * (sigma_bench/sigma_clone) + (1-shrinkage) * 1.0`, clipped | L, bounds, shrinkage_w |
| `hybrid` | Weighted blend of beta and vol modes | L, bounds, shrinkage_w, lambda |
| `regime` | Fixed alpha per volatility regime (no estimation) | 3 alphas, 2 thresholds |

### Regime-Conditional Mode (recommended)

The regime mode classifies each rebalance date into one of three volatility regimes based on the trailing 20-day annualized realized vol of the benchmark:

| Regime | Condition | Alpha | Rationale |
|--------|-----------|-------|-----------|
| Normal | vol < 15% | 1.20 | Clone beta is stable; full overlay compensates structural gap |
| Elevated | 15% <= vol < 25% | 1.10 | Relationships noisier; reduce overlay |
| Crisis | vol >= 25% | 1.00 | Kill overlay to protect capital |

This mode uses no rolling estimation, no shrinkage, and no vol dampening (the regime IS the dampening). Only 3 alpha values and 2 vol thresholds, with thresholds fixed a priori from long-run vol distribution percentiles.

### Protective Mechanisms

Two safety mechanisms apply across all modes:

**Drawdown circuit-breaker:** If the clone's 60-day rolling drawdown exceeds -15%, alpha is forced to 1.0 (overlay off). Re-enables when drawdown recovers past -10%.

**Vol-regime dampening** (non-regime modes only): When trailing 20-day realized vol exceeds its 90th percentile, alpha is scaled toward 1.0 linearly, with a floor of 0.3x at extreme vol levels.

---

## 3. Robustness & Overfitting Audit

### Purpose

The robustness audit detects overfitting, parameter instability, look-ahead bias, and false improvements due to noise. It is not a performance optimization tool. It answers: "Is the model's apparent performance real, or an artifact of in-sample fitting?"

### Diagnostic Sections

The audit comprises 8 independent sections:

#### Section 1: Walk-Forward Testing

Rolling evaluation: 5-year train (1260 days), 1-year test (252 days), roll by 1 year. Config is held fixed across all folds (no re-tuning). Computes tracking error, beta, correlation, information ratio, and max drawdown per fold.

**What it detects:** Performance collapse in later folds indicates instability or regime sensitivity.

**Flag threshold:** Walk-forward TE std > 3.0%, or TE deteriorates by 50%+ from first to last fold.

**Current result:** SP_mirror WF TE mean = 9.04%, std = 2.19%. Clone WF TE mean = 10.18%, std = 2.22%. SP_mirror wins 9/10 folds. No flag triggered.

#### Section 2: Parameter Sensitivity

Varies three parameters independently across a grid:
- Lookback L: {200, 252, 500}
- Alpha bounds: {[0.6,1.4], [0.8,1.2], [0.9,1.1]}
- Rebalance frequency: {Monthly, Weekly}

Computes full-sample and OOS (last 30%) tracking error for each of 18 configurations.

**What it detects:** Fragility (small param changes cause large performance swings) and boundary optima (best config at grid edge).

**Flag threshold:** OOS TE range > 5pp (fragile) or > 2pp (moderate).

**Current result:** OOS TE range across grid = 0.33pp. Model is parameter-insensitive. No flag triggered.

#### Section 3: Naive Baseline Comparison

Compares the dynamic model against:
- Clone (alpha = 1, no overlay)
- Constant alpha = sigma_bench / sigma_clone (vol ratio)
- Constant alpha = 1 / beta

All baselines are routed through the same pipeline (with dampening and circuit-breaker) for fair comparison.

**What it detects:** Whether dynamic calibration adds value over a fixed constant. If not, the estimation is fitting to noise.

**Current result:** Evaluated via composite score (see Section 8). Dynamic model is marginal vs constant.

#### Section 4: Placebo Test (Supervisor Signal)

When a supervisor is configured, replaces the supervisor's probability series P(t) with:
- Shuffled P(t) (time-permuted)
- 5 independent random uniform P(t) series

If performance is similar to placebo, the supervisor adds no real signal.

**Skipped when:** No supervisor is configured (supervisor_mode = "none").

#### Section 5: Look-Ahead Bias Check

Verifies:
- Alpha changes only on rebalance dates (or is adjusted daily by trailing-data mechanisms like dampening)
- All rebalance dates are actual trading days
- First non-trivial alpha respects the lookback warm-up period
- Vol dampening uses trailing (backward-looking) windows only

**Current result:** All checks pass.

#### Section 6: Stability Diagnostics

Analyzes:
- Alpha distribution (mean, std, skew, autocorrelation at rebalance dates)
- Rolling 252-day beta (mean, std, min, max) computed after warm-up period

**Flag thresholds:** Alpha std > 0.3 (excessive swings), rolling beta std > 0.3 (regime instability).

**Current result:** Alpha std = 0.09, rolling beta std = 0.11. Both well within thresholds.

#### Section 7: Crisis Stress Test

Isolates four crisis subperiods and compares SP_mirror vs Clone:

| Crisis | SP_mirror TE | Clone TE | SP_mirror DD | Clone DD |
|--------|-------------|----------|-------------|----------|
| 2011 Euro Crisis | 30.1% | 30.1% | 0% | 0% |
| 2018 Vol Shock | 9.7% | 13.8% | -8.7% | -6.3% |
| 2020 COVID Crash | 22.2% | 23.1% | -39.1% | -37.3% |
| 2022 Rate Shock | 9.4% | 11.2% | -24.4% | -18.4% |

**Flag threshold:** SP_mirror TE > Clone TE in more than half of crisis periods.

**Current result:** SP_mirror improves TE in 2 of 3 active crises (2018, 2022). COVID is roughly neutral. No flag triggered.

#### Section 8: Composite Robustness Score

The core innovation of the audit framework. Instead of binary pass/fail flags, each model variant is scored on a single composite metric that formalizes the complexity-performance tradeoff:

```
Composite Score = mean(fold_TE) + lambda * std(fold_TE) + kappa * n_params
```

Where:
- `mean(fold_TE)` rewards tracking accuracy across walk-forward folds
- `lambda * std(fold_TE)` penalizes instability (lambda = 0.5)
- `kappa * n_params` penalizes model complexity (kappa = 0.05 per parameter)

A model is only justified if its composite score beats simpler alternatives. This prevents both overfitting (complex model with high variance) and meta-overfitting (selecting config by lowest OOS TE alone).

**Current ranking:**

| Rank | Model | Composite | TE mean | TE std | Params |
|------|-------|-----------|---------|--------|--------|
| 1 | SP_mirror (regime) | 10.575 | 9.26% | 2.13% | 5 |
| 2 | SP_mirror (dynamic) | 10.581 | 9.04% | 2.19% | 9 |
| 3 | Clone (alpha=1) | 11.296 | 10.18% | 2.22% | 0 |
| 4 | Const alpha=1.0 | 11.346 | 10.18% | 2.22% | 1 |

The regime model wins because it achieves similar fold stability with fewer parameters than the dynamic model, and substantially better tracking than constant alpha.

### Scoring Rubric

The final overfitting risk score aggregates all section flags on a 0-10 scale:

| Score Range | Verdict | Interpretation |
|-------------|---------|----------------|
| 0-2 | LOW | Model appears stable and not overfit |
| 3-5 | MEDIUM | Some sensitivity; interpret results cautiously |
| 6-10 | HIGH | Significant fragility; high risk of overfitting |

Penalty points by section:

| Check | Max Penalty | Condition |
|-------|-------------|-----------|
| S1: WF TE deterioration | +2 | Last fold TE > 1.5x first fold TE |
| S1: WF TE variance | +1 | TE std > 3.0% |
| S2: Parameter sensitivity | +2 | OOS TE range > 5pp |
| S3/S8: Dynamic vs simpler | +2 | Dynamic composite > constant composite |
| S4: Supervisor placebo | +2 | Supervisor TE ~= placebo TE |
| S5: Look-ahead bias | +2 | Any warning detected |
| S6: Alpha instability | +1 | Alpha std > 0.3 |
| S6: Beta instability | +1 | Rolling beta std > 0.3 |
| S7: Crisis degradation | +1 | Overlay hurts in >50% of crises |

**Current score: 1/10 (LOW)**

---

## 4. Evolution History

The model went through a systematic audit-driven refinement:

| Stage | Config | Score | Key Change |
|-------|--------|-------|------------|
| Original | bounds [0.6,1.4], no protections | 7/10 HIGH | Alpha pinned at 1.4 ceiling, COVID DD = -50% |
| + Tighter bounds | bounds [0.8,1.2] | - | Reduced overlay aggression |
| + Shrinkage | w=0.5 toward alpha=1.0 | - | Alpha distributed, not pinned at ceiling |
| + Vol dampening | 90th percentile trigger | - | Reduces overlay in extreme vol |
| + Circuit breaker | -15% DD trigger | 3/10 MEDIUM | Prevents catastrophic drawdown amplification |
| + Composite scoring | Mean-variance + complexity penalty | 1/10 LOW | Principled complexity-performance tradeoff |
| + Regime-conditional | 3 fixed alphas by vol regime | 1/10 LOW | Best composite score; structured adaptiveness |

Key metrics improvement:

| Metric | Original | Final |
|--------|----------|-------|
| Walk-forward TE std | 3.29% | 2.13% |
| COVID fold TE | 16.9% | 11.6% |
| Worst drawdown | -50.3% | -39.5% |
| Alpha std | 0.180 | 0.066 |
| Rolling beta std | 0.507 | 0.105 |
| Overfitting risk | 7/10 HIGH | 1/10 LOW |

---

## 5. How to Run

### Robustness audit

```bash
python run_sp_mirror_robustness.py
```

Runs the full 8-section audit for both old and new configs, with side-by-side comparison. Outputs to `results/Derivatives_cloning/robustness_tests/`.

### Main SP_mirror

```bash
python run_sp_mirror.py
```

Runs all overlay variants (beta, vol, hybrid, regime) against both SPY Total Returns and SPY benchmarks. Outputs to `results/Derivatives_cloning/`.

### Output structure

```
results/Derivatives_cloning/
+-- robustness_tests/
|   +-- old_config/          # Pre-fix baseline audit
|   |   +-- s1_walk_forward_results.csv
|   |   +-- s1_walk_forward_te.png
|   |   +-- s2_param_sensitivity.csv
|   |   +-- s2_param_heatmap.png
|   |   +-- s3_naive_baselines.csv
|   |   +-- s4_placebo_test.csv
|   |   +-- s5_look_ahead_checks.csv
|   |   +-- s6_stability_stats.csv
|   |   +-- s6_alpha_distribution.png
|   |   +-- s6_rolling_beta_stability.png
|   |   +-- s7_crisis_stress.csv
|   |   +-- s7_crisis_te.png
|   |   +-- s8_composite_scores.csv
|   |   +-- s8_composite_scores.png
|   |   +-- robustness_report.txt
|   +-- new_config/          # Post-fix audit (same files)
+-- sp_mirror_SPY_Total_Returns_comparison.png
+-- sp_mirror_SPY_Total_Returns_metrics.csv
+-- sp_mirror_SPY_comparison.png
+-- sp_mirror_SPY_metrics.csv
+-- rolling_beta_vs_*.png
+-- semiannual_beta_vs_*.csv / .png
```

---

## 6. Key Design Principles

1. **Complexity must earn its keep.** Every parameter is justified by improvement on the composite score (mean TE + 0.5 * TE std + 0.05 * n_params). A simpler model wins by default.

2. **Regime awareness over continuous estimation.** The regime-conditional approach outperforms rolling beta estimation because it captures the main source of variation (vol regimes) with far fewer parameters.

3. **Protective mechanisms are separate from alpha estimation.** The drawdown circuit-breaker operates independently of the alpha mode. It provides a hard floor against catastrophic losses regardless of which estimation method is used.

4. **Evaluation is strictly out-of-sample.** Walk-forward folds never re-tune on test data. The composite score penalizes fold variance, not just mean performance. The audit compares against naive baselines through the same pipeline for fairness.

5. **The audit is a tool, not a score to optimize.** The goal is to detect real fragility, not to game the scoring rubric. False positives (penalizing legitimate features) are fixed in the audit, not in the model.
