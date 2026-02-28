# Bootstrapping & Backtesting Framework

This document explains the two statistical validation methods used to confirm that our GNN supervisor's performance is real and not an artifact of overfitting.

---

## 1. Chan 2018 — Synthetic Validation (Are we overfit to one history?)

### The Problem
History only happened once. A strategy that achieved Sharpe 1.052 on 2020–2026 might have simply gotten lucky — maybe the specific sequence of COVID → recovery → inflation → AI rally is the one timeline where it works. On a slightly different market path, it could fail.

### The Solution: Generate Alternate Realities
We use **stationary block bootstrap** (Chan 2018) to create 50–100 synthetic market histories:

1. Take the real daily returns (4,121 days × 32 stocks)
2. Slice them into random blocks of ~21 days each (preserving short-term correlation structure)
3. Shuffle the blocks into a new random sequence
4. That's one synthetic history — repeat 50–100 times

```
Real history:    [Jan-Feb] [Mar-Apr] [May-Jun] ...  (chronological)
Synthetic #1:    [May-Jun] [Nov-Dec] [Mar-Apr] ...  (shuffled blocks)
Synthetic #2:    [Aug-Sep] [Jan-Feb] [Jul-Aug] ...  (different shuffle)
```

Each synthetic history has the **same statistical properties** (volatility, correlations, fat tails) but a **different sequence**. If your strategy only worked because of the exact order of events in 2010–2026, it will fail on synthetic histories.

### What We Run on Each Synthetic Path

For **each** of the 50+ synthetic histories:

| Strategy | What happens |
|---|---|
| **XGBoost CPO** | Train XGBoost supervisor on 60% of synthetic path, test on 40% |
| **GNN CPO** | Apply the real GNN checkpoint's vol-scaled α to synthetic worker returns |
| **Worker Only** | Raw QP clone with α ≡ 1 (no supervisor) |
| **Benchmark** | SPY buy & hold on the synthetic path |

### Key Result
```
XGBoost CPO:  0.127 ± 0.602   ← FAILS (worse than doing nothing)
GNN CPO:      X.XXX ± X.XXX   ← run main.py to get this
Worker Only:  0.677 ± 0.547
Benchmark:    0.501 ± 0.462
```

The XGBoost CPO **massively underperforms** the Worker alone on synthetic data. This proves it memorized the exact feature signatures of COVID/2022 inflation rather than learning generalizable regime structure. This is the primary motivation for replacing it with the GNN.

### Implementation
- **File:** `src/synthetic_validation.py`
- **Key functions:**
  - `stationary_block_bootstrap()` — generates synthetic paths
  - `_run_pipeline_on_path()` — runs XGBoost + GNN on one synthetic history
  - `run_synthetic_validation()` — orchestrates the full validation
  - `plot_synthetic_validation()` — histogram of Sharpe distributions
- **Called from:** `main.py → phase6_synthetic()`
- **Output:** `results2/synthetic_validation.png`, `results2/synthetic_validation_results.csv`

---

## 2. Bootstrap Confidence Intervals (Is the Sharpe statistically significant?)

### The Problem
GNN achieves Sharpe 1.052 vs SPY's 0.745. But is that difference statistically significant, or could it be sampling noise?

### The Solution: Resample the Real Returns
Unlike Chan's synthetic validation (which tests generalization), bootstrap CIs test **statistical significance** of a single number:

1. Take GNN's 1,818 real daily returns
2. Draw 21-day blocks with replacement, 10,000 times
3. Compute Sharpe on each resample
4. Report the 2.5th and 97.5th percentile → 95% confidence interval

```
GNN Supervisor:  Sharpe = 1.052  [95% CI: X.XX – X.XX]
Clone (Worker):  Sharpe = 0.691  [95% CI: X.XX – X.XX]
SPY Buy & Hold:  Sharpe = 0.745  [95% CI: X.XX – X.XX]
```

If the GNN's CI doesn't overlap with SPY's CI → **statistically significant outperformance**.

### Implementation
- **File:** `bootstrap_ci.py` (standalone script)
- **Key function:** `block_bootstrap_sharpe()` — stationary block bootstrap with 10k resamples
- **Output:** `results6/bootstrap_ci.csv` + LaTeX-ready table for `paper2.tex`

### How to Run
```bash
python bootstrap_ci.py
```
Takes ~5 min (QP solver runs once, then 10k numpy resamples).

---

## 3. How They Differ

| | Chan 2018 Synthetic | Bootstrap CI |
|---|---|---|
| **What it creates** | Fake market histories | Resampled real returns |
| **Question answered** | Does the strategy generalize to other possible timelines? | Is the Sharpe number statistically significant? |
| **Tests for** | Overfitting to one historical path | Sampling variance in the point estimate |
| **Requires retraining?** | XGBoost: yes (per path). GNN: no (transfer) | No |
| **Output** | Sharpe distribution across 50+ paths | 95% CI on a single Sharpe |
| **Run time** | ~25 min (50 paths) | ~5 min |

### Why We Need Both
- A strategy can **pass** synthetic validation but have a **wide CI** → works in many timelines but the exact Sharpe is uncertain
- A strategy can have a **tight CI** but **fail** synthetic validation → the Sharpe is precisely measured on one history that it overfit to
- We want both: works across timelines (Chan) AND the measured Sharpe is precise (bootstrap)

---

## 4. How to Run Everything

### Step 1: Full Pipeline (includes Chan synthetic with GNN)
```bash
python main.py
```
This runs all 6 phases including Phase 6 (synthetic validation). Takes ~25–30 min total.  
The synthetic phase will now output GNN CPO alongside XGBoost CPO.

### Step 2: Bootstrap CIs
```bash
python bootstrap_ci.py
```
Runs after `main.py`. Takes ~5 min. Outputs CI table to paste into `paper2.tex` Table VI.

### Step 3: Update Paper
After both scripts finish:
1. Copy the GNN synthetic Sharpe from terminal output → paste into `paper2.tex` Table V
2. Copy the bootstrap CIs from terminal output → paste into `paper2.tex` Table VI

---

## 5. Reference

- Chan, E. (2018). *"Optimizing Trading Strategies without Overfitting."* QuantCon 2018.
- Summary: Standard backtesting on a single historical path leads to overfitting. Use block bootstrap to generate alternate market histories and test if strategy parameters remain profitable across thousands of simulated realities.
