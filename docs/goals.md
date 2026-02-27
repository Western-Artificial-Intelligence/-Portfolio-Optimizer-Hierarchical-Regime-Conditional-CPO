# Portfolio Optimizer — Project Goals & To-Do List

> **Project:** Hierarchical Regime-Conditional Portfolio Optimization (CPO)
> **Paper:** `docs/paper.tex`
> **Conference:** CUCAI 2026
> **Last Updated:** February 27, 2026 — 3:30 AM EST

---

## Available Data (✅ Ready)

| File | Contents |
|------|----------|
| `stock_history.csv/json` | Daily price history for TSX 60 + SPY (PX_LAST, volume, OHLC, etc.) |
| `stock_profiles.csv/json` | Static profiles (GICS sector, beta, analyst rating, currency) |
| `economic_indicators.csv/json` | Macro indicators (VIX, yield curves, credit spreads, DXY, MOVE, etc.) |
| `yield_curve_spread.csv/json` | 10Y-2Y yield curve spread time series |
| `data/derivatives/*.csv` | Bloomberg BDH CSVs: ES1 futures, USDCAD, SPXT, SPX, VIX, MOVE, DXY, sector indices |

---

## Phase 1: Data Preparation & Exploration ✅

- [x] Load and inspect all CSV/JSON datasets — confirm date ranges, tickers, and column coverage
- [x] Clean data: handle missing values, align dates across datasets, remove delisted tickers
- [x] Compute daily returns for all assets + SPY benchmark
- [x] Merge macro indicators with price data on date index
- [x] Exploratory data analysis (EDA): correlation heatmap, rolling stats, regime visual inspection

---

## Phase 2: Layer 1 — The Worker (Rolling QP Solver) ✅

- [x] Implement tracking error minimization via Quadratic Programming (`cvxpy`)
- [x] Apply constraints: fully invested (`Σwᵢ = 1`), diversification (`0 ≤ wᵢ ≤ 0.15`)
- [x] Run optimization on training period with 5-year rolling lookback
- [x] Evaluate base portfolio: tracking error, correlation, and cumulative returns vs. SPY
- [x] Implement rolling/monthly rebalancing of base weights

---

## Phase 3: Layer 2 — AI Supervisor (Continuous Meta-Labeling) ✅

### 3a. Uncertainty Feature Engineering
- [x] Implement rolling uncertainty features (vol, vol-of-vol, dispersion, EWMA variance)
- [x] Extract forecast uncertainty as risk features (~15 features)

### 3b. Regime Detection & Feature Engineering
- [x] Build the "super-state" feature set: uncertainty + MOVE, yield curve, credit spreads, DXY + momentum
- [x] Create binary meta-labels: `1` if portfolio drawdown ≤ 2% next week, else `0`
- [x] Walk-forward split: train 2010–2019, test 2020–2026 (no look-ahead)

### 3c. XGBoost Classifier
- [x] Train XGBoost classifier on meta-labels with super-state features
- [x] Tune with `TimeSeriesSplit` cross-validation
- [x] Output continuous confidence score `P ∈ [0, 1]`

### 3d. Execution Logic ✅
- [x] **Update to continuous blending**: `W_final = P × W_aggressive + (1-P) × W_defensive`
- [x] Define defensive allocation (min-volatility or cash)
- [x] Remove old step-function logic (bull/bear thresholds)

### 3e. Super-State Bug Fix (see `docs/superstate_fix_roadmap.md`)
- [ ] Chunk 1: Add diagnostics (log index metadata, overlap counts, NaN counts)
- [ ] Chunk 2: Normalize indices to canonical form (fix tz/type mismatches)
- [x] Chunk 3: Make `dropna()` less aggressive
- [x] Chunk 4: Align uncertainty to clone dates explicitly
- [x] Chunk 5: Data loading and Phase 1/2 consistency
- [x] Chunk 6: Tests and regression guard

---

## Phase 3.5: SP_Mirror — Derivative Overlay System ✅

> Added via `SP_Clone_Derivatives` PR (merged Feb 2026). See `docs/SP_Mirror.md` and `docs/SP_Mirror_Robustness.md`.

- [x] Implement Layer 2 derivative overlay engine (`src/sp_mirror_SPY_Total_Returns.py`)
- [x] ES1 S&P 500 futures overlay with calibrated alpha scaling
- [x] Spot USDCAD FX hedge for overlay USD exposure
- [x] 4 overlay modes: beta, vol, hybrid, regime-conditional
- [x] Regime-conditional mode with 3 fixed alphas by volatility regime (recommended)
- [x] Drawdown circuit-breaker (-15% DD → overlay off)
- [x] Vol-regime dampening for non-regime modes
- [x] SPY convenience wrapper (`src/sp_mirror_SPY.py`)
- [x] Single entrypoint runner (`run_sp_mirror.py`) — runs both SPY_Total_Returns + SPY benchmarks
- [x] 8-section robustness audit framework (`run_sp_mirror_robustness.py`)
- [x] Walk-forward testing, parameter sensitivity, placebo test, crisis stress test
- [x] Composite robustness score: **1/10 (LOW overfitting risk)**
- [x] Bloomberg derivatives data loader

---

## Phase 4: Synthetic Validation (Chan 2018)

- [ ] Implement stationary block bootstrap for synthetic market generation
- [ ] Generate 1,000 synthetic price histories preserving cross-correlations and vol clustering
- [ ] Run full Worker pipeline on each synthetic history
- [ ] Train + test Supervisor on each synthetic history (60/40 train/test split)
- [ ] Record Sharpe ratio, max drawdown, tracking error for each simulation
- [ ] Generate histogram plot: distribution of Sharpe ratios (Benchmark vs. Worker-only vs. CPO)
- [ ] Statistical test: confirm CPO distribution is shifted right vs. benchmarks

---

## Phase 5: Feature Importance & Interpretability (SHAP)

- [x] Fix SHAP 0.46 + XGBoost 3.2 `base_score` compatibility bug (patched in `src/shap_analysis.py`)
- [ ] Generate SHAP beeswarm plot for XGBoost — rank feature importance
- [ ] Generate SHAP dependence plots (e.g., VIX > 25 disproportionate impact)
- [ ] Ablation study: Sharpe ratio when removing each feature group:
  - Full model (baseline)
  - Remove volatility features → expect significant degradation
  - Remove yield curve / macro features → expect moderate degradation
  - Remove credit spreads → expect minor degradation
  - Remove downside risk features → expect moderate degradation

---

## Phase 6: Backtesting & Final Results

- [ ] Backtest all benchmarks on test period (2020–2025):
  - SPY Buy & Hold
  - Worker Only (no Supervisor) — always 100% invested
  - Equal-weight
  - VIX rule-based (VIX > 25 → 50%)
  - Static 60/40 portfolio
- [ ] Report: Sharpe, Sortino, max drawdown, Calmar, tracking error
- [ ] Conditional metrics: high VIX vs. low VIX, inverted vs. normal yield curve
- [ ] Generate final comparison table and cumulative return plots

---

## Phase 7: Paper Finalization

- [x] Rewrite paper.tex with new framing (regime-conditional, not "Canadian Clone")
- [ ] Fill in results tables and figures from Phase 5 & 6
- [ ] Add SHAP plots and synthetic validation histogram
- [ ] Finalize bibliography
- [ ] Review and polish for CUCAI submission

---

## Phase 8: Project Infrastructure

- [x] Set up project structure (`src/`, `docs/`, `data/`, `results/`)
- [x] Create `requirements.txt` with dependencies
- [x] Write `architecture.md` documenting system design
- [x] Write SP_Mirror documentation (`SP_Mirror.md`, `SP_Mirror_Robustness.md`, `sp_mirror_README.md`)
- [x] Write `results_analysis.md` — full first-run analysis with interpretations
- [x] Write `superstate_fix_roadmap.md` — chunked bug fix plan
- [ ] Write proper `README.md` with setup instructions
- [ ] Ensure `.env` and data files are in `.gitignore`
