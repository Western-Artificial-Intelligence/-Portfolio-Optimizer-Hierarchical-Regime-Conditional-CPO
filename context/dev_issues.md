# Dev Issues — Portfolio Optimizer

Assignable issues for 3 developers. Each is self-contained and non-blocking.

---

## Issue #1 — Run SHAP + Ablation + Synthetic Validation (Phases 4-6)

**Assignee:** Dev 1  
**Priority:** 🔴 Critical  
**Estimated Time:** 2–3 hours (plus overnight for synthetic)

### Context
Phases 1-3 work. Phase 4 (SHAP) crashed on a `shap` + `xgboost` version conflict. A fix has already been applied to `src/shap_analysis.py`. Phases 5 (ablation) and 6 (synthetic) have never been run.

### Tasks

- [ ] Pull latest code (the SHAP fix is in `src/shap_analysis.py`)
- [ ] Run `uv run python main.py` and confirm Phase 4 (SHAP) now completes
- [ ] Verify these files are generated:
  - `results/shap_beeswarm.png`
  - `results/shap_bar.png`
  - `results/shap_dependence.png`
  - `results/shap_importance.csv`
- [ ] Verify Phase 5 (ablation) completes and generates:
  - `results/ablation_results.csv`
  - `results/ablation_sharpe_drop.png`
- [ ] Enable Phase 6 (synthetic validation) in `main.py` — currently the call may be commented or set to low `n_paths`. Run with `n_paths=50` first as a test (~5 min), verify output:
  - `results/synthetic_sharpe_distribution.png`
  - `results/synthetic_summary.csv`
- [ ] Once confirmed working, run overnight with `n_paths=200` or `n_paths=500`
- [ ] If SHAP still fails, fallback: pin `xgboost==2.1.0` in `pyproject.toml`, run `uv sync`, re-run

### Acceptance Criteria
All 6 phases complete without error. All result files exist and contain valid data.

### Files You'll Touch
`main.py`, `src/shap_analysis.py` (read only, fix already applied), `pyproject.toml` (only if fallback needed)

---

## Issue #2 — Tune Supervisor to Beat Unsupervised Baseline

**Assignee:** Dev 2  
**Priority:** 🟡 High  
**Estimated Time:** 3–4 hours

### Context
The current Supervisor produces a **lower** Sharpe ratio (0.584) than the unsupervised clone (0.600) on the test period. This is the #1 weakness in the paper. The Supervisor's parameters have never been tuned — they're all defaults.

### Background
The Supervisor's key parameters are in `src/supervisor.py`:
- `delta` (drawdown threshold for meta-labels): currently 2%. Controls how many "danger" labels are generated. Lower δ → more cautious labels → more hedging.
- `H` (forward horizon for meta-labels): currently 5 days. How far ahead to look for max drawdown.
- `bear_allocation` in `apply_supervisor()`: currently 0.5 (50% allocation during bear signals).
- XGBoost hyperparameters: `max_depth`, `n_estimators`, `learning_rate`, `subsample`.

### Tasks

- [ ] Create a tuning script (`src/tuning.py` or a notebook) that:
  1. Iterates over parameter grids:
     - `delta`: [0.01, 0.015, 0.02, 0.03, 0.05]
     - `H`: [3, 5, 10, 15]
     - `bear_allocation`: [0.3, 0.5, 0.7]
  2. For each combination, runs the Supervisor pipeline on the train/val split
  3. Records: Sharpe ratio, max drawdown, annualized return, number of "danger" days triggered
- [ ] Find the parameter combination that maximizes Sharpe **on the validation set** (not test)
- [ ] Re-run the full pipeline with the best parameters on the test set
- [ ] Save tuning results to `results/tuning_results.csv`
- [ ] Update `src/supervisor.py` defaults to the best parameters (with a comment explaining why)

### Acceptance Criteria
Supervised Sharpe on test set is **≥ unsupervised Sharpe** (currently 0.600). Document the best parameters found.

### Target Numbers
- Supervised Sharpe ≥ 0.60 (match unsupervised)
- Ideally: Supervised Sharpe ≥ 0.62 with lower max drawdown
- If you can't beat unsupervised on Sharpe, focus on **max drawdown reduction** > 1% (currently only 27 bps better)

### Files You'll Touch
`src/supervisor.py`, new file `src/tuning.py`, `results/tuning_results.csv`

---

## Issue #3 — Add Missing Benchmarks + Write Results Section

**Assignee:** Dev 3  
**Priority:** 🟡 High  
**Estimated Time:** 3–4 hours

### Context
The paper promises several benchmarks that don't exist yet, and the Results section of `paper.tex` is not written. The raw numbers are available in `results/` CSVs.

### Part A: Implement Missing Benchmarks

- [x] **VIX Rule-Based Benchmark**: In `src/benchmarks.py`:
  - If VIX (use SPY vol_21d as proxy, or download VIX data) > 25: reduce clone allocation to 50% (rest to cash)
  - If VIX > 35: reduce to 25%
  - Otherwise: 100% clone allocation
  - Compute Sharpe, max DD, annualized return on test period
- [x] **Static 60/40 Benchmark**:
  - 60% clone returns + 40% risk-free (use 0% or 2% annualized for cash)
  - Compute same metrics
- [x] **Equal-Weight TSX** (in `strategy_comparison.csv` and in test-period comparison via `run_equal_weight_benchmark`)
- [x] Add all benchmarks to `phase3_comparison.csv` output

### Part B: Write Results Section in Paper

- [ ] Open `docs/paper.tex`
- [ ] Write Section 7 (Results) covering:
  1. **Phase 2 results**: Clone vs SPY tracking error, weight stability, full-period Sharpe table
  2. **Phase 3 results**: Supervised vs unsupervised comparison, regime detection quality, feature importance ranking
  3. **Benchmark comparison**: Table with ALL strategies (Clone, Supervised Clone, SPY, VIX Rule-Based, 60/40, Equal-Weight)
  4. If SHAP/ablation results are available from Dev 1, include those too
- [ ] Reference figures with `\includegraphics` for: `clone_vs_spy.png`, `supervisor_decisions.png`, `supervised_vs_unsupervised.png`, `regime_indicators.png`
- [ ] Add figure captions explaining what each plot shows
- [ ] Update the abstract with actual numbers (replace any placeholder text)

### Acceptance Criteria
- At least 4 benchmark strategies in the final comparison table
- Results section is ≥ 1.5 pages with at least 3 figures
- Paper compiles without LaTeX errors

### Files You'll Touch
`src/backtester.py` or new `src/benchmarks.py`, `docs/paper.tex`, `results/phase3_comparison.csv`

---

## Coordination Notes

| Issue | Depends On | Blocks |
|-------|-----------|--------|
| #1 (SHAP/Ablation/Synthetic) | Nothing | #3 (SHAP plots for paper) |
| #2 (Tuning) | Nothing | #1 (re-run with tuned params), #3 (updated numbers) |
| #3 (Benchmarks + Paper) | Best if #1 and #2 finish first, but can start Part B now |  Nothing |

**Recommended order:**
1. Dev 2 starts tuning immediately (most impactful)
2. Dev 1 runs SHAP/ablation in parallel
3. Dev 3 starts benchmarks (Part A) now, writes paper (Part B) once #1 and #2 deliver updated results
