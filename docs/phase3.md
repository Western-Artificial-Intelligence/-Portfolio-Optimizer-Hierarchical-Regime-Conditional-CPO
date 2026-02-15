# Phase 3: Layer 2 — AI Supervisor (Continuous Meta-Labeling)

## What We Built

Phase 3 adds the **AI Supervisor** layer. It doesn't pick stocks — it monitors the Worker's portfolio and **continuously adjusts exposure** based on macroeconomic regime signals. When the Supervisor is confident the market is safe, it lets the Worker run aggressively. When it detects danger, it smoothly transitions to a defensive allocation.

This implements the meta-labeling framework from López de Prado (2018), extended with continuous output inspired by Chan's Conditional Portfolio Optimization (2023).

---

## Architecture

```
  Volatility Features ──────┐
  Vol-of-Vol ───────────────┤
  Cross-Sectional Dispersion┤
  EWMA Variance ────────────┤
  MOVE Index ───────────────┤──→ XGBoost ──→ P ∈ [0,1] ──→ Continuous Blending
  Yield Curve Spread ───────┤     Classifier     confidence
  Credit Spreads ───────────┤                        │
  DXY ──────────────────────┤                        ▼
  Downside Risk ────────────┤           W = P × W_agg + (1-P) × W_def
  Portfolio Momentum ───────┘
```

**Execution logic** (continuous blending):
- **P = 1.0** → Full aggressive allocation (Worker's min-tracking-error weights)
- **P = 0.0** → Full defensive allocation (min-volatility or cash)
- **P = 0.5** → 50/50 blend between aggressive and defensive
- Smooth transitions → less turnover, proportional hedging

---

## Files Created / Modified

| File | What |
|------|------|
| `src/forecaster.py` | **Uncertainty estimator** — rolling vol, vol-of-vol, dispersion, EWMA variance, downside risk (~15 features) |
| `src/supervisor.py` | **AI Supervisor** — meta-label generation, super-state feature builder, XGBoost with TimeSeriesSplit CV, execution logic |
| `src/eda.py` | 2 new plots: supervisor confidence score, supervised vs unsupervised |
| `main.py` | Updated to run Phase 1 → Phase 2 → Phase 3 |

---

## How It Works Step-by-Step

### 1. Meta-Label Generation
For each trading day, we look ahead 5 days:
- **Label = 1** (safe): Portfolio didn't suffer > 2% drawdown in next week
- **Label = 0** (danger): Portfolio did suffer a drawdown > 2%

### 2. Super-State Features (~40 features)
| Category | Features |
|----------|----------|
| Uncertainty | Rolling vol (5/21/63d), vol-of-vol, cross-sectional dispersion, EWMA variance, downside vol, rolling max drawdown |
| Macro | MOVE level + changes, DXY level + changes, IG/HY spreads + changes |
| Yield Curve | Spread level + change, inversion flag |
| Momentum | Portfolio 5/21/63-day cumulative returns |
| Benchmark | SPY vol (21/63d), relative vol ratio |

### 3. XGBoost Classifier
- Trained with **TimeSeriesSplit** cross-validation (no data leakage)
- **Walk-forward split**: train on 2010–2019, test on 2020–2026
- Handles class imbalance via `scale_pos_weight`
- Outputs continuous probability `P = predict_proba()`

### 4. Continuous Blending Execution
The final allocation smoothly blends between aggressive and defensive:
```
W_final = P × W_aggressive + (1-P) × W_defensive
```
This replaces the old step-function approach (binary bull/bear switching), producing smoother transitions and less turnover.

---

## How to Run

```bash
python main.py
```

**Expected outputs in `results/`:**
- `supervisor_decisions.png` — confidence score over time with regime shading
- `supervised_vs_unsupervised.png` — triple overlay: supervised vs unsupervised vs SPY
- `phase3_comparison.csv` — metrics comparison table
- `feature_importances.csv` — ranked XGBoost feature importances

**Expected runtime:** ~60–90 seconds total (Phase 1 + Phase 2 + Phase 3).

---

## What to Expect

- **Test period (2020–2026)** includes COVID crash, recovery, 2022 rate hikes, 2023 rally
- The Supervisor should **reduce max drawdown** vs the unsupervised Worker
- Sharpe should be ≥ the unsupervised portfolio (better risk-adjusted returns)
- XGBoost CV accuracy ~55–70% is normal for meta-labeling (financial data is noisy)
- **Top features** will likely be macro indicators (MOVE, credit spreads) and rolling drawdown

---

## What's Next: Phase 4 (Synthetic Validation)

Phase 4 implements the Chan (2018) synthetic validation framework — generating 1,000 bootstrapped market histories and running the full Worker + Supervisor pipeline on each. This proves the strategy's outperformance isn't an artifact of overfitting to a single historical path.
