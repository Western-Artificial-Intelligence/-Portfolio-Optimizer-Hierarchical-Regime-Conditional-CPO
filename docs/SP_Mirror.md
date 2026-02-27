# SP_Mirror: 3-Layer Hierarchical S&P 500 Tracking System

## Overview

SP_Mirror is a derivative-overlay extension that builds a synthetic S&P 500 tracker from Canadian-listed equities. It operates as a **3-layer hierarchical architecture** where each layer addresses a distinct source of tracking error between a Canadian stock portfolio and the S&P 500 Total Return Index (SPY_Total_Returns, Bloomberg ticker SPXT).

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 3 — AI Supervisor (optional)                         │
│  Regime-aware allocation dial / cash blend                  │
│  Adjusts overlay aggressiveness based on macro conditions   │
├─────────────────────────────────────────────────────────────┤
│  Layer 2 — Derivative Overlay + FX Hedge                    │
│  ES1 futures overlay to close beta/vol gap vs benchmark     │
│  Spot USDCAD hedge to neutralize currency exposure          │
├─────────────────────────────────────────────────────────────┤
│  Layer 1 — Canadian Clone (QP Worker)                       │
│  Monthly rolling QP: min tracking error vs SPY              │
│  Universe: ~32 Canadian equities across all GICS sectors    │
└─────────────────────────────────────────────────────────────┘
```

---

## Layer 1 — Canadian Clone (QP Worker)

**Goal:** Build a long-only portfolio of Canadian stocks that minimizes tracking error vs the S&P 500.

**Method:** Quadratic programming solved monthly on a rolling 5-year lookback window.

```
minimize   (1/T) || R_canadian @ w  -  r_SPY ||²
s.t.       sum(w) = 1,   0 <= w_i <= 0.15
```

- **Universe:** ~32 liquid Canadian equities spanning Financials, Energy, Industrials, Tech, Telecom, Utilities, Materials, Consumer.
- **Rebalance:** Monthly (last trading day of month).
- **Output:** Daily return series `clone_return[t]` and optional weight history.

**What it cannot do alone:** The clone is constrained to Canadian stocks — it cannot perfectly match the S&P's sector mix, individual constituents, or currency denomination. Typical annualized tracking error is ~10%.

---

## Layer 2 — Derivative Overlay + FX Hedge

**Goal:** Reduce tracking error between the clone and the benchmark by adding a calibrated S&P 500 futures position and hedging the resulting USD exposure.

### Alpha estimation

On each rebalance date, a scaling factor `alpha` is estimated over a rolling window (L = 252 days by default):

| Mode     | Formula                                                                  |
|----------|--------------------------------------------------------------------------|
| `beta`   | `alpha = 1 / beta`, where `beta = cov(clone, bench) / var(bench)`       |
| `vol`    | `alpha = sigma_bench / sigma_clone`                                      |
| `hybrid` | `alpha = lambda * (1/beta) + (1 - lambda) * (sigma_bench / sigma_clone)` |

Alpha is clipped to `[alpha_min, alpha_max]` (default `[0.6, 1.4]`) and held constant (forward-filled) until the next rebalance.

### Daily return decomposition

```
overlay_return[t]    = (alpha[t] - 1) * r_ES[t]
fx_hedge_return[t]   = -max(0, alpha[t] - 1) * r_USDCAD[t]
cost[t]              = cost_bps * |alpha[t] - alpha[t-1]|
SP_mirror_return[t]  = clone_return[t] + overlay_return[t] + fx_hedge_return[t] - cost[t]
```

- **Overlay:** `(alpha - 1)` is the net futures notional as a fraction of NAV. When `alpha > 1`, we are *adding* S&P exposure; when `alpha < 1`, we are *reducing* it.
- **FX hedge:** Only the USD exposure introduced by the overlay (i.e., `max(0, alpha - 1)`) is hedged via spot USDCAD returns. No carry is modeled (no FX forward rates available).
- **Costs:** A simplified proxy — `cost_bps` (default 0, easy to set e.g. 1 bp = 0.0001) times the absolute change in alpha. This captures the notional change at each rebalance.

### Data requirements

| Series              | Source CSV              | Role                                    |
|---------------------|-------------------------|-----------------------------------------|
| `ES1__PX_SETTLE`    | E-mini S&P 500 futures  | Overlay return stream (daily pct_change) |
| `USDCAD__PX_LAST`   | Spot USD/CAD            | FX hedge return stream                  |
| `SPXT__PX_LAST`     | SPY Total Returns       | Primary benchmark (total return)        |
| `SPX__PX_LAST`      | S&P 500 Price Index     | SPY benchmark (price return only)       |

### Known limitations

- **No futures margin/funding rates.** ES1 daily returns are used directly — the implicit funding cost embedded in the futures basis is not separated.
- **No explicit roll yield.** Bloomberg's generic continuous ES1 series is used as-is; no contract stitching.
- **FX hedge is spot-only.** Without FX forward rates, the carry component of a proper FX hedge is missing.
- **Cost model is a simplified proxy.** Real transaction costs depend on bid-ask, market impact, and exchange fees, not just notional change.

---

## Layer 3 — AI Supervisor (optional)

**Goal:** Dynamically modulate the overlay aggressiveness based on macro/regime conditions.

The supervisor is an XGBoost meta-labeling classifier trained on a "super-state" of uncertainty and macro features (VIX, yield curve, credit spreads, DXY, MOVE index). It outputs a probability `P ∈ [0, 1]` representing confidence that the portfolio will avoid a significant drawdown.

Three supervisor modes:

| Mode   | Formula                                                | Description                                      |
|--------|--------------------------------------------------------|--------------------------------------------------|
| `none` | No modification                                        | Pure Layer 1 + Layer 2                           |
| `dial` | `alpha_final = 1 + P * (alpha - 1)`                   | Scales overlay toward/away from 1 based on P     |
| `cash` | `final_return = P * sp_mirror_return + (1-P) * 0`     | Blends SP_mirror with cash (for comparison only) |

When `P = 1` (high confidence), the full overlay is applied. When `P → 0` (danger), the overlay shrinks toward zero (`alpha → 1`, i.e., clone-only).

---

## Configuration

All toggles are controlled via the `SPMirrorConfig` dataclass:

```python
from src.sp_mirror_SPY_Total_Returns import SPMirrorConfig

cfg = SPMirrorConfig(
    overlay_mode="beta",       # "beta" | "vol" | "hybrid"
    overlay_lookback=252,      # Rolling window for alpha estimation
    hybrid_lambda=0.5,         # Weight for hybrid mode
    alpha_min=0.6,             # Overlay bounds
    alpha_max=1.4,
    rebal_freq="M",            # "M" (monthly) | "W" (weekly) | "D" (daily)
    fx_hedge=True,             # Hedge overlay USD exposure
    cost_bps=0.0,              # Transaction cost in decimal (0.0001 = 1 bp)
    supervisor_mode="none",    # "none" | "dial" | "cash"
    prob_series=None,          # pd.Series of P values for dial/cash
)
```

---

## Evaluation Metrics

All metrics are computed on the **evaluation period only** (2016-01-04 onward; training = dataset start through end-2015). Benchmarked against SPY_Total_Returns and SPY:

| Metric                    | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| Ann Return (%)            | `mean(daily_return) * 252`                                                  |
| Ann Vol (%)               | `std(daily_return) * sqrt(252)`                                             |
| Sharpe                    | `mean(excess_return) / std(return) * sqrt(252)` (rf = 2%)                   |
| Sortino                   | `(ann_return - rf) / downside_std`                                          |
| Max DD (%)                | Largest peak-to-trough decline                                              |
| Calmar                    | `ann_return / abs(max_dd)`                                                  |
| Skewness                  | Skewness of daily returns                                                   |
| Kurtosis                  | Excess kurtosis of daily returns                                            |
| Tracking Error (%)        | `std(active_return) * sqrt(252)`                                            |
| Corr w/ benchmark         | Pearson correlation of daily returns                                        |
| Beta vs benchmark         | OLS regression slope of strategy returns on benchmark returns               |
| Beta rolling mean         | Mean of 252-day rolling beta                                                |
| Mean Active Return (%)    | `mean(strategy - bench) * 252`                                              |
| Information Ratio         | `mean_active_return / tracking_error`                                       |
| Avg Cumulative Gap (pts)  | Mean difference between cumulative indices (both starting at 100)           |
| FX beta proxy             | Correlation with USDCAD returns (should be near 0 for hedged overlay)       |

---

## How to Run

### Prerequisites

- Python 3.10+
- `pandas`, `numpy`, `matplotlib`, `cvxpy` (for Layer 1 QP), `xgboost` (for Layer 3 supervisor)

### Data setup

Place Bloomberg BDH-exported CSV files in `data/derivatives/`:

```
data/derivatives/
├── ES1__PX_SETTLE.csv          # Required
├── USDCAD__PX_LAST.csv         # Required
├── SPXT__PX_LAST.csv           # Required (SPY Total Returns)
├── SPX__PX_LAST.csv            # Required (SPY price return)
├── VIX__PX_LAST.csv            # Optional
├── VIX3M__PX_LAST.csv          # Optional
├── MOVE__PX_LAST.csv           # Optional
├── DXY__PX_LAST.csv            # Optional
├── USGG2YR__PX_LAST.csv        # Optional
├── USGG10YR__PX_LAST.csv       # Optional
├── S5INFT__PX_LAST.csv         # Optional (sector indices)
├── ...                         # Other sector CSVs
└── README.md
```

Each CSV has two columns: `Date` and a value column (e.g. `PX_LAST`, `Settle`).

### Clone return

The script will attempt to load clone returns in this priority:

1. `data/clone_return.csv` (columns: `date`, `return`) if it exists
2. Auto-generate from Layer 1 QP pipeline (requires `stock_history.csv` and `cvxpy`)

### Run

```bash
cd -Portfolio-Optimizer-Hierarchical-Regime-Conditional-CPO
python run_sp_mirror.py
```

A single run evaluates all overlay variants against both SPY_Total_Returns and SPY.

### Outputs

All SP_mirror results are saved to `results/Derivatives_cloning/`:

```
results/Derivatives_cloning/
├── sp_mirror_SPY_Total_Returns_comparison.png    # SPY_Total_Returns vs Clone vs SP_mirror variants
├── sp_mirror_SPY_Total_Returns_metrics.csv       # Metrics table vs SPY_Total_Returns
├── sp_mirror_SPY_comparison.png                  # SPY vs Clone vs SP_mirror variants
└── sp_mirror_SPY_metrics.csv                     # Metrics table vs SPY
```

### Programmatic usage

```python
from src.sp_mirror_SPY_Total_Returns import (
    SPMirrorConfig, load_derivatives_csv_folder,
    run_sp_mirror_single, metrics_table, OPTIONAL_SERIES,
)

derivatives_df = load_derivatives_csv_folder("data/derivatives/",
    required={"ES1__PX_SETTLE", "USDCAD__PX_LAST", "SPXT__PX_LAST"},
    optional=OPTIONAL_SERIES)

cfg = SPMirrorConfig(overlay_mode="hybrid", cost_bps=0.0001)

# SPY_Total_Returns benchmark
sp_ret, overlay, fx_hedge, alpha, r_bench = run_sp_mirror_single(
    clone_return, derivatives_df, cfg, bench_col="SPXT__PX_LAST")

# SPY benchmark
sp_ret, overlay, fx_hedge, alpha, r_spy = run_sp_mirror_single(
    clone_return, derivatives_df, cfg, bench_col="SPX__PX_LAST")
```

---

## Project Structure

```
├── src/
│   ├── sp_mirror_SPY_Total_Returns.py   # Layer 2+3: overlay engine, metrics (configurable benchmark)
│   ├── sp_mirror_SPY.py                 # Convenience wrapper for SPY benchmark
│   ├── qp_solver.py                     # Layer 1: QP tracking portfolio
│   ├── supervisor.py                     # Layer 3: XGBoost meta-labeling supervisor
│   ├── data_loader.py                    # Bloomberg data parsing
│   ├── config.py                         # Paths, universe, constraints
│   ├── features.py                       # Feature engineering
│   ├── backtester.py                     # General portfolio metrics
│   └── ...
├── run_sp_mirror.py         # Single entrypoint: runs both SPY_Total_Returns + SPY
├── main.py                  # Full pipeline (Phases 1-6)
└── data/
    ├── derivatives/         # Bloomberg BDH CSVs for Layer 2
    ├── stock_history.csv    # Canadian + SPY price history for Layer 1
    └── ...
```
