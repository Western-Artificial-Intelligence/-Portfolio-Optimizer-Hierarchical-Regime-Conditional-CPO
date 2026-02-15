# System Architecture

> **Hierarchical Regime-Conditional Portfolio Optimization (CPO)**

## High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                                │
│  Bloomberg CSVs → data_loader.py → features.py              │
│  (stock prices, macro indicators, yield curve, profiles)     │
└──────────────────────┬──────────────────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │   Layer 1: The Worker   │
          │    (qp_solver.py)       │
          │                         │
          │  Rolling QP Solver      │
          │  min ||R_spy - Rw||²    │
          │  Monthly rebalance      │
          │  5-year lookback        │
          │                         │
          │  Output: W_aggressive   │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │  Layer 2: The Supervisor│
          │   (supervisor.py)       │
          │                         │
          │  ┌───────────────────┐  │
          │  │ Feature Eng.      │  │
          │  │ (forecaster.py)   │  │
          │  │ Vol, Vol-of-Vol,  │  │
          │  │ Dispersion, EWMA  │  │
          │  └───────┬───────────┘  │
          │          │              │
          │  ┌───────▼───────────┐  │
          │  │ XGBoost Classifier│  │
          │  │ Meta-Labeling     │  │
          │  │ TimeSeriesSplit CV│  │
          │  └───────┬───────────┘  │
          │          │              │
          │  Output: P ∈ [0, 1]    │
          │  (confidence score)     │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │   Execution Logic       │
          │                         │
          │  W_final = P × W_agg   │
          │    + (1-P) × W_def     │
          │                         │
          │  Continuous blending    │
          │  between aggressive     │
          │  and defensive alloc.   │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │   Validation Layer      │
          │                         │
          │  Backtester             │
          │  SHAP Analysis          │
          │  Synthetic Validation   │
          │  (1,000 bootstrapped    │
          │   market histories)     │
          └─────────────────────────┘
```

## Module Responsibilities

| Module | Role |
|--------|------|
| `config.py` | Central config: paths, tickers, date splits, constraints |
| `data_loader.py` | Parses Bloomberg CSVs (multi-level headers, macro data) |
| `features.py` | Returns computation, macro merge, rolling stats, sparse ticker filter |
| `forecaster.py` | Uncertainty feature engineering (rolling vol proxies) |
| `qp_solver.py` | Layer 1: QP tracking error minimization with CVXPY/OSQP |
| `supervisor.py` | Layer 2: meta-labels, super-state features, XGBoost, execution |
| `backtester.py` | Performance metrics (Sharpe, Sortino, MaxDD, Calmar, TE) |
| `eda.py` | All visualization (heatmaps, cumulative returns, regime plots) |
| `main.py` | Pipeline runner: Phase 1 → Phase 2 → Phase 3 |

## Key Design Decisions

1. **Separation of concerns**: Worker knows nothing about regimes; Supervisor knows nothing about stock selection. Each layer can be independently validated.

2. **Continuous vs. binary output**: The Supervisor outputs a smooth probability P ∈ [0,1] rather than a binary regime switch, reducing turnover and enabling proportional hedging.

3. **Rolling volatility over DeepAR**: We use computationally simple rolling estimators (vol, vol-of-vol, EWMA variance) that capture the same uncertainty signal as probabilistic forecasting, with better interpretability.

4. **Synthetic validation**: Strategy robustness is tested on 1,000 bootstrapped market histories (Chan 2018), not just a single historical backtest.

## Data Flow

```
Bloomberg Terminal
    ↓
CSV files (data/)
    ↓
data_loader.py → prices, profiles, econ, yield_curve
    ↓
features.py → returns, merged macro features
    ↓
qp_solver.py → weight_history, clone_returns (Layer 1)
    ↓
forecaster.py → uncertainty features
supervisor.py → confidence scores, regime, supervised_returns (Layer 2)
    ↓
backtester.py → metrics comparison
eda.py → plots saved to results/
```
