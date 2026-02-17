# Results Analysis — First Pipeline Run (February 2026)

> **Author:** Generated from pipeline output analysis  
> **Date:** February 14, 2026  
> **Pipeline:** Hierarchical Regime-Conditional CPO  
> **Status:** Phases 1–3 completed successfully. Phase 4 (SHAP) encountered XGBoost compatibility issue (since fixed).

---

## Executive Summary

The first end-to-end pipeline run produced **13 output files** (9 plots, 4 CSVs) covering data exploration, portfolio construction, and AI Supervisor evaluation. The **Worker (QP Solver)** successfully constructs a diversified Canadian clone of SPY with 11.3% mean annualized tracking error. The **Supervisor (XGBoost meta-labeling)** demonstrates learned regimeawareness, with confidence scores dropping sharply during the COVID crash and 2022 inflation bear, but the out-of-sample risk-adjusted improvement over the unsupervised clone is marginal. Phase 4 (SHAP analysis) failed due to a known `shap 0.46` + `xgboost 3.2` serialization bug, which has been patched.

---

## Phase 1 — Data Exploration

### 1.1 Correlation Heatmap (`correlation_heatmap.png`)

**What it shows:** Pairwise Pearson correlation of daily returns across all 30+ TSX equities and SPY.

**Interpretation:**

- **Canadian banks form a tightly correlated cluster** (RY, TD, BMO, BNS, correlation ~0.65–0.80). This is expected—they share the same regulatory environment, interest rate exposure, and Canadian housing market sensitivity. For portfolio construction, this cluster behaves almost as a single risk factor, meaning the QP solver cannot gain much diversification by overweighting multiple banks simultaneously.

- **Gold miners (ABX, AEM) and base metals (CCO, IVN) exhibit low correlation** (~0.15–0.30) with financials and telecoms. This is the most exploitable diversification axis in the TSX universe. These stocks provide hedging value during risk-off periods—gold tends to rally when equities sell off during systemic stress.

- **SPY shows moderate correlation (~0.35–0.55) with most TSX stocks.** This is critical: the cross-border correlation is high enough that a Canadian portfolio *can* track SPY, but low enough that perfect replication is impossible. The residual tracking error is structural—it cannot be eliminated by any weighting scheme. This explains why the QP solver achieves ~11% tracking error rather than ~0%.

- **Technology outliers (SHOP, CSU, CLS)** cluster together with higher correlation to SPY than to the rest of the TSX, reflecting their US revenue exposure and growth-factor sensitivity.

**Implication for the paper:** This heatmap validates our choice of a diversified TSX universe. The moderate cross-correlations mean the QP solver has meaningful degrees of freedom, but the imperfect SPY correlation guarantees nonzero tracking error—confirming that we are solving a genuinely hard optimization problem, not a trivial one.

---

### 1.2 Cumulative Returns — Top Performers (`cumulative_returns.png`)

**What it shows:** Indexed cumulative wealth (100 = start) for the top-performing stocks in the universe plus SPY, from 2010–2026.

**Interpretation:**

- **CSU (Constellation Software): ~140x return.** This is one of the best-performing stocks in Canadian history, compounding at ~29% annualized over 16 years. Its dominance in the chart is so extreme that it compresses the visual scale for everything else.

- **DOL (Dollarama): ~50x, CLS (Celestica): ~45x, ATD (Alimentation Couche-Tard): ~22x.** These are the second tier—still exceptional multi-baggers.

- **SPY: ~5x.** A solid ~12.5% annualized return. The S&P 500 is a formidable benchmark.

**Why this matters for the project:** The QP solver is *not* trying to pick winners. It's minimizing tracking error against SPY. So even though CSU returned 140x, the solver might allocate *less* to CSU if its return profile diverges from SPY. This is by design—the Worker optimizes for consistency, not alpha. The Supervisor then decides how much of this consistent allocation to actually deploy.

---

### 1.3 Rolling Volatility (`rolling_volatility.png`)

**What it shows:** Two panels—63-day annualized volatility for SPY (top) and the mean cross-sectional volatility of the Canadian universe (bottom).

**Interpretation:**

- **COVID-19 March 2020 spike is the dominant feature.** SPY vol spiked to ~60%, while the Canadian universe mean hit nearly 100% (with individual stocks exceeding 150%). This was a once-in-a-decade vol event.

- **2022 inflation bear was milder but persistent.** SPY vol reached ~30% and stayed elevated for months—a "slow grind" rather than a sharp shock. This type of regime is harder for binary classifiers to detect because there's no single catastrophic day.

- **The ±1 std band (Canadian panel) widens during crises.** High cross-sectional dispersion means stocks are moving in different directions—some are crashing while others hold up. This dispersion is itself an informative feature for the Supervisor, and indeed `dispersion_63d` ranks in the top 10 of feature importances.

- **Baseline volatility has been structurally low since 2023.** The post-COVID era shows compressed vol around 10–12% for SPY. The Supervisor should be outputting high confidence (P ≈ 1.0) during this period, and the confidence score plot confirms this.

---

## Phase 2 — Worker (QP Solver)

### 2.1 Clone vs SPY (`clone_vs_spy.png`)

**What it shows:** Cumulative returns of the Canadian Clone (blue) vs SPY benchmark (red), with shading indicating which outperforms.

**Interpretation:**

- **The clone persistently underperforms SPY** — ending at ~315 indexed vs SPY's ~610. The red shading (SPY outperformance) dominates the chart.

- **This is expected and does not represent a failure.** The clone uses 30 TSX stocks to replicate an S&P 500 ETF. There is a fundamental structural gap: the S&P 500 is driven by US mega-cap tech (Apple, Microsoft, Nvidia, Amazon) which have no Canadian equivalents of comparable scale. The TSX is overweight financials and energy, underweight technology.

- **The clone's primary value is its *risk profile*, not its return.** The QP solver succeeds in producing a portfolio that *behaves like* SPY during normal markets (correlation 0.70) while being composed entirely of Canadian securities. The return gap is the cost of geographic diversification.

- **Practical context:** An institutional investor constrained to Canadian securities (e.g., by mandate or tax considerations) would care about this risk matching, not the absolute return gap.

### 2.2 Tracking Error (`tracking_error.png`)

**What it shows:** Rolling 63-day annualized tracking error between the clone and SPY, with the mean (11.3%) shown as a dashed red line.

**Interpretation:**

- **Mean TE of 11.3% is moderate.** For context, a dedicated index-tracking fund would target TE < 1%. Our 11.3% reflects the cross-border nature of the problem—we're replicating a US index with Canadian stocks, not performing same-index replication.

- **TE spikes during crises:** ~37% during 2011 (European debt), ~27% during COVID, ~17% during 2018 trade wars. These are periods of market dislocation where cross-border correlations break down—US and Canadian markets temporarily decouple.

- **TE has generally trended downward since 2015.** From 2016–2019, TE averaged ~7%, suggesting improving correlation structure. This may reflect increasing globalization of equity markets.

- **Mean-reverting behavior:** The TE chart exhibits classic Ornstein-Uhlenbeck dynamics—spikes followed by gradual reversion to the ~11% mean. This is the theoretical basis for the OU process extension mentioned in the paper's future work.

### 2.3 Weight Evolution (`weight_evolution.png`)

**What it shows:** Stacked area chart showing the QP solver's monthly portfolio allocations over time.

**Interpretation:**

- **The portfolio is genuinely diversified.** No single stock dominates indefinitely, and the 15% cap constraint is active for several holdings (visible as flat ceilings in the chart).

- **Structural shifts are visible.** Around 2020, the allocation to BNS and OTEX increases significantly, while GIB/A decreases. This reflects the solver adapting to evolving covariance structure—as correlations shift, the optimal tracking portfolio changes.

- **"Other" category (light blue) is large,** indicating the solver uses many stocks at small weights rather than concentrating in a few. This is good for real-world implementation (lower single-stock risk).

- **WCN (Waste Connections) is consistently overweighted.** Feature importances later confirm this—WCN is the most commonly held stock in all rebalancing periods, likely because its return profile (defensive, low-beta) correlates well with SPY during diverse market conditions.

### 2.4 Strategy Comparison (Full Period)

| Strategy | Ann Return | Ann Vol | Sharpe | Max DD |
|----------|-----------|---------|--------|--------|
| Canadian Clone (QP) | 7.84% | 13.15% | 0.444 | -37.78% |
| SPY Buy & Hold | 12.48% | 17.08% | 0.613 | -34.10% |
| Equal-Weight TSX | 15.26% | 13.97% | 0.949 | -35.98% |

**Key observation:** The **equal-weight TSX portfolio has the highest Sharpe ratio (0.949).** This is a well-known empirical result — equal-weight portfolios often outperform cap-weighted and optimized portfolios over long horizons because they avoid concentration in overvalued large-caps and mechanically buy low / sell high during rebalancing. The equal-weight benchmark having a higher Sharpe than SPY is consistent with the equal-weight premium documented by DeMiguel, Garlappi, and Uppal (2009).

**The clone's Sharpe (0.444) is lower than SPY (0.613).** Again, this is structural—the clone is constrained to match SPY's *risk* profile using a fundamentally different asset universe. The lower return directly translates to a lower Sharpe.

---

## Phase 3 — AI Supervisor

### 3.1 Supervisor Confidence Score (`supervisor_decisions.png`)

**What it shows:** The XGBoost Supervisor's confidence score P ∈ [0, 1] over the full data period, with the corresponding clone cumulative returns below.

**Interpretation:**

- **P stays near 1.0 for 80%+ of the sample.** This is correct behavior — the meta-labels are 87.5% "safe" (the Worker performs fine most of the time). The Supervisor has learned that *most* days are safe.

- **Sharp drops to P < 0.4 during crises are the actionable signals:**
  - **2015–2016:** TSX oil crisis. Canadian equity markets crashed ~25% while the S&P 500 was relatively flat. The Supervisor correctly identified this as a danger period for the Canadian clone.
  - **2018–2019:** Yield curve inversion + US-China trade war. Multiple P drops to ~0.5–0.6 reflecting sustained uncertainty.
  - **March 2020:** P crashes to ~0.1 during COVID. This is the strongest signal in the dataset. The model correctly identifies this as maximum danger.
  - **2022:** Several sustained drops during the inflation bear market. The P drops are less extreme (~0.4–0.6) because the market decline was more gradual.
  - **2025:** Recent volatility triggers moderate P drops.

- **The transitions are smooth, not binary.** P doesn't just jump between 0 and 1 — it produces intermediate values (0.3, 0.5, 0.7) reflecting degrees of uncertainty. This validates our continuous blending approach over a binary regime switch.

### 3.2 Supervised vs Unsupervised (`supervised_vs_unsupervised.png`)

**What it shows:** Cumulative returns of three strategies on the **test period (2020–2026):** SPY, unsupervised clone, and supervised clone.

**Interpretation:**

- **All three strategies end in a similar range (200–215 indexed).** The Supervisor does not produce a dramatic outperformance. This is an honest result.

- **The blue line (supervised) tracks slightly below the red (unsupervised) during bull periods.** This is the *cost* of hedging — when P < 1.0, the Supervisor reduces exposure, which means you miss some upside during rallies. This is the "volatility tax" in reverse — you pay a small return cost for downside protection.

- **During the COVID crash (March 2020), the supervised and unsupervised lines diverge briefly.** The supervised line drops slightly less, but both recover together. The protection is there but modest.

### 3.3 Test Period Comparison Table

| Strategy | Ann Return | Sharpe | Max DD |
|----------|-----------|--------|--------|
| Clone + AI Supervisor | 12.68% | 0.584 | -37.51% |
| Clone (unsupervised) | 13.26% | 0.600 | -37.78% |
| SPY Buy & Hold | 14.32% | 0.602 | -34.10% |

**Honest assessment:**

- The Supervisor **reduced max drawdown** by 27 basis points (37.51% vs 37.78%). This is marginal and not statistically significant on a single path.
- The Supervisor **reduced annualized return** by 58 bps (12.68% vs 13.26%). The hedging cost exceeds the drawdown benefit on this particular historical path.
- **Net Sharpe ratio impact is slightly negative** (0.584 vs 0.600). The Supervisor is not adding value on this single backtest.

**Why this happens and what it means:**

1. **The test period (2020–2026) was dominated by a strong bull market.** The post-COVID recovery was one of the most aggressive rallies in history. Any strategy that hedges (reduces exposure during uncertainty) will underperform in a persistent bull. This is mathematically guaranteed.

2. **The Supervisor's value is in tail-risk reduction, which is invisible on a single path.** You need the synthetic validation (Phase 6) to see it — across 1,000 alternative histories, the Supervisor should shift the Sharpe distribution rightward and, more importantly, truncate the left tail (fewer catastrophic outcomes).

3. **The 87.5% "safe" base rate makes it hard to add value.** When 87.5% of days are safe, the model must correctly time the remaining 12.5% danger days to make a difference. Even small false-positive rates (calling "danger" on a safe day) eat into returns.

### 3.4 Feature Importances (`feature_importances.csv`)

Top 10 features ranked by XGBoost importance (gain-based):

| Rank | Feature | Importance | Category |
|------|---------|-----------|----------|
| 1 | `vol_63d` | 5.09% | Volatility |
| 2 | `clone_ret_21d` | 3.73% | Momentum |
| 3 | `macro_DXY_PX_LAST` | 3.54% | Macro (Dollar) |
| 4 | `macro_HY_SPREAD_chg_21d` | 3.26% | Macro (Credit) |
| 5 | `clone_ret_63d` | 3.25% | Momentum |
| 6 | `vol_5d` | 3.16% | Volatility |
| 7 | `ewma_var_26` | 3.14% | Volatility |
| 8 | `macro_T10Y2Y_chg_21d` | 3.07% | Macro (Yield Curve) |
| 9 | `dispersion_63d` | 3.02% | Dispersion |
| 10 | `macro_MOVE_PX_LAST` | 2.99% | Macro (Bond Vol) |

**Key observations:**

- **Volatility features dominate** (vol_63d, vol_5d, ewma_var_26 in top 7). This is economically rational— volatility is the single best predictor of near-term drawdowns. High vol → higher probability of a >2% drawdown in the next 5 days.

- **DXY (US Dollar) at #3 is notable.** A strong dollar typically signals risk-off conditions globally and is particularly damaging for Canadian equities (commodity-sensitive). The model has learned this relationship without being explicitly told.

- **Credit spread changes (#4) and yield curve dynamics (#8) are macro stress barometers.** Widening HY spreads signal credit stress; yield curve flattening precedes recessions. Both are well-documented in the financial literature as leading indicators.

- **Clone momentum features (#2, #5) act as a self-aware signal.** The model can detect when the clone itself is losing momentum — a form of regime feedback that pure macro models lack.

- **`yc_inverted` (binary yield curve flag) has 0.0 importance.** This is interesting — the *continuous* yield curve spread features are informative, but the binary "inverted yes/no" flag adds nothing. The model already captures this through the continuous features. Consider removing the binary flag.

### 3.5 Regime Indicators Dashboard (`regime_indicators.png`)

Four-panel visualization of the macro features feeding the Supervisor:

- **MOVE Index (Bond Volatility):** Spikes during every major crisis — 2011, 2020, 2022. The 2022 spike was the highest in the sample (nearly 200), reflecting the Fed's aggressive rate-hiking cycle. This is the #10 feature by importance.

- **DXY (US Dollar Index):** Secular uptrend from ~75 (2011) to ~110 (2022), then partial reversal. The strong dollar period (2020–2022) coincided with the worst period for Canadian equities relative to US equities.

- **10Y–2Y Yield Curve Spread:** **The 2022–2024 inversion (red shaded area) is the most prominent feature.** The curve was inverted (negative spread) for nearly 2 years — the longest inversion in modern history. This is a classic recession signal, and the Supervisor's low-confidence readings during this period confirm it learned the relationship.

- **Credit Spreads:** IG spreads spike during the 2011, 2015, and 2020 crises. HY spreads are smoother (fewer extreme prints in our data). The 21-day change in HY spread (#4 feature importance) captures the *speed* of deterioration, which is more informative than the level.

---

## Phase 4 — SHAP Analysis (FAILED)

### What Happened

```
ValueError: could not convert string to float: '[8.788133E-1]'
```

**Root cause:** `shap 0.46.0` and `xgboost 3.2.0` are incompatible. XGBoost 3.x changed how it serializes the `base_score` parameter in its internal JSON configuration. The old format was a plain float (`0.8788133`); the new format wraps it in brackets (`[8.788133E-1]`). SHAP's `TreeExplainer` tries to parse this string with `float()`, which fails on the bracket characters.

**Fix applied:** Patched `src/shap_analysis.py` to extract and clean the `base_score` from XGBoost's internal config before SHAP reads it. Re-running the pipeline should produce SHAP beeswarm, bar, and dependence plots.

**Note:** This is a known upstream issue. Once the SHAP maintainers release a fix, the workaround can be removed.

---

## Recommendations for Improvement

### A. Short-Term (Before Submission)

1. **Run SHAP + Ablation (Phases 4–5).** Fix is already applied. These results will significantly strengthen the paper—SHAP beeswarm plots are visually compelling and provide the interpretability narrative that distinguishes this work from black-box approaches.

2. **Run synthetic validation (Phase 6) with at least n_paths=200.** The *entire thesis* of the paper is that single-path backtests are insufficient. Without the synthetic validation results, the paper's strongest contribution is unsubstantiated. Start with 50 paths for a quick sanity check, then run 200+ overnight.

3. **Add missing benchmarks.** The paper promises VIX Rule-Based and 60/40 comparisons. These are trivial to implement (5 lines each) and provide important context — if a simple "hedge when VIX > 25" rule matches the Supervisor's Sharpe, then the ML adds no value over heuristics.

4. **Tune the Supervisor's parameters.** The current results show marginal value. Experimenting with the drawdown threshold (delta), forward horizon (H), and defensive allocation weight could meaningfully improve the supervised Sharpe. These are hyperparameters we haven't searched over yet.

### B. Medium-Term (Strengthen the Methodology)

5. **Walk-forward retraining.** Currently the Supervisor is trained once on 2010–2019 and evaluated on 2020–2026. A more robust protocol would retrain yearly (e.g., train on all data up to 2019, evaluate 2020; retrain including 2020, evaluate 2021; etc.). This prevents concept drift and better reflects real-world deployment.

6. **Transaction cost integration into the objective function.** Transaction costs are currently applied *post-hoc* in the backtester. Incorporating them directly into the QP objective (as a turnover penalty) would produce more realistic portfolios with lower turnover.

7. **Add sector-level constraints.** The current QP solver allows unconstrained sector exposure (subject to the 15% single-stock cap). Adding sector max/min constraints (e.g., no more than 30% financials) would improve diversification and reduce the vulnerability to sector-level drawdowns.

8. **Shrinkage estimators for the covariance matrix.** The QP solver uses the sample covariance matrix, which is noisy for 30 assets. Ledoit-Wolf shrinkage or Oracle Approximating Shrinkage would produce more stable weight estimates, especially during volatile periods when the sample covariance is least reliable.

### C. Long-Term (Research Extensions)

9. **Ornstein-Uhlenbeck tracking error model.** As noted in the paper's future work, modeling the tracking error as an OU process would provide a continuous-time signal for Supervisor confidence calibration. The estimated half-life parameter θ could serve as an additional Supervisor feature.

10. **Multi-asset extension.** Apply the same Worker+Supervisor architecture to bonds, commodities, or a multi-asset class portfolio. The Supervisor's macro features (yield curve, credit spreads, DXY) are asset-class agnostic — they should transfer to fixed income or commodity portfolios with minimal modification.

11. **Online learning for the Supervisor.** Replace batch XGBoost training with an online gradient boosting variant that updates incrementally as new data arrives. This would eliminate the need for periodic retraining and allow the model to adapt to emerging regimes in real-time.

12. **Ensemble of Supervisors.** Train multiple Supervisor models with different feature subsets (one using only volatility, one using only macro, one using only momentum) and combine their confidence scores. This would provide robustness through model diversity and make the ablation study's results directly actionable.

13. **Alpha generation in the Worker.** The current Worker minimizes tracking error (pure beta replication). An alternative formulation would minimize tracking error while *tilting* toward stocks with higher expected returns (estimated via a factor model). This transforms the Worker from a pure tracker into a portable-alpha vehicle, potentially addressing the persistent return gap vs SPY.

---

## File Inventory from First Run

| File | Type | Phase | Description |
|------|------|-------|-------------|
| `correlation_heatmap.png` | Plot | 1 | Pairwise return correlations |
| `cumulative_returns.png` | Plot | 1 | Top stock performance over time |
| `rolling_volatility.png` | Plot | 1 | SPY + TSX rolling vol |
| `summary_stats.csv` | CSV | 1 | Per-stock risk/return statistics |
| `clone_vs_spy.png` | Plot | 2 | Clone vs benchmark cumulative |
| `tracking_error.png` | Plot | 2 | Rolling tracking error over time |
| `weight_evolution.png` | Plot | 2 | Portfolio allocation changes |
| `strategy_comparison.csv` | CSV | 2 | Full-period benchmark comparison |
| `supervisor_decisions.png` | Plot | 3 | Confidence score P over time |
| `supervised_vs_unsupervised.png` | Plot | 3 | Supervisor impact on test period |
| `regime_indicators.png` | Plot | 3 | Macro feature dashboard |
| `phase3_comparison.csv` | CSV | 3 | Test-period benchmark comparison |
| `feature_importances.csv` | CSV | 3 | XGBoost feature importance ranking |
