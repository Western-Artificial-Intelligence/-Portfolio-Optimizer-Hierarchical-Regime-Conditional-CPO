# Phase 2: Layer 1 — The Worker (Rolling QP Solver)

## What We Built

The Worker constructs a diversified equity portfolio that tracks the S&P 500 (SPY) by solving a **Quadratic Programming (QP)** problem on a rolling basis.

### How It Works

We solve:

```
minimize    (1/T) Σ (r_portfolio,t − r_SPY,t)²     ← tracking error
subject to  Σ wᵢ = 1                                ← fully invested
            0 ≤ wᵢ ≤ 0.15                           ← max 15% per stock
```

In plain English: *"Find the weights that make the portfolio's daily returns as close to SPY's as possible, while staying diversified."*

### Rolling Rebalancing

The optimizer uses **walk-forward monthly rebalancing**:
- Every month, it re-solves the QP using the **last 5 years** (1,260 trading days) of data
- This means the portfolio adapts as correlations shift over time
- Tickers that IPO midway (SHOP in 2015, QSR in 2014) are naturally included once they have enough history

### Files Created / Modified

| File | What |
|------|------|
| `src/qp_solver.py` | Core optimizer (`optimize_tracking`) + rolling rebalancer (`rolling_optimization`) |
| `src/backtester.py` | Performance metrics: Sharpe, Sortino, MaxDD, Calmar, tracking error, turnover |
| `src/eda.py` | 3 plots: portfolio vs SPY, rolling TE, weight evolution |
| `main.py` | Updated to run Phase 1 + Phase 2 sequentially |

---

## How to Run

```bash
python main.py
```

This runs both Phase 1 and Phase 2. Phase 2 will:
1. Run ~130 monthly QP optimizations (2015–2026, after 5-year lookback warmup)
2. Print a strategy comparison table (Portfolio vs SPY vs Equal-Weight)
3. Save plots + CSV to `results/`

**Expected outputs in `results/`:**
- `clone_vs_spy.png` — cumulative returns overlay with over/underperformance shading
- `tracking_error.png` — rolling 63-day annualized tracking error
- `weight_evolution.png` — stacked area chart showing how weights shift over time
- `strategy_comparison.csv` — metrics table

**Expected runtime:** ~30–60 seconds.

---

## What to Expect

- **Portfolio Sharpe** should be roughly in range 0.4–0.8 (SPY is 0.73)
- **Tracking error** should be < 10% annualized — lower is better
- The weight evolution chart will show the optimizer favoring banks (RY, TD, BMO) heavily, since they correlate most with SPY
- **COVID period (March 2020)** will show tracking error spiking — this is exactly the problem Phase 3's AI Supervisor is designed to fix

---

## What's Next: Phase 3 (AI Supervisor)

Phase 3 adds the Supervisor layer — an XGBoost meta-labeling system that monitors the macroeconomic environment and outputs a continuous confidence score `P ∈ [0,1]`. This score blends the portfolio between aggressive (full tracking) and defensive (capital preservation) allocations.
