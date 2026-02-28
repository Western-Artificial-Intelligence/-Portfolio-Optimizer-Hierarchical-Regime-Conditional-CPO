# System Architecture

> **Hierarchical Regime-Conditional Portfolio Optimization (CPO)**

## High-Level Design

```text
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
          │  Output: W_clone        │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │  Layer 2: The Supervisor│
          │   (gnn_supervisor.py)   │
          │                         │
          │  ┌───────────────────┐  │
          │  │ LSTM Temporal     │  │
          │  │ Encoder (20 days) │  │
          │  └───────┬───────────┘  │
          │          │              │
          │  ┌───────▼───────────┐  │
          │  │ Graph Attn (GAT)  │  │
          │  │ Spatial Predictor │  │
          │  │ (33 nodes, fully  │  │
          │  │  connected graph) │  │
          │  └───────┬───────────┘  │
          │          │              │
          │  Output: α_t ∈ [0, 1]  │
          │  (blending coefficient) │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │   Execution Logic       │
          │                         │
          │  W_final = α_t * W_clone│
          │    + (1 - α_t) * Cash  │
          │                         │
          │  End-to-end training    │
          │  to directly maximize   │
          │  the Sharpe ratio.      │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │   Validation Layer      │
          │                         │
          │  Walk-Forward CV        │
          │  (4 Expanding Folds)    │
          │                         │
          │  Synthetic Validation   │
          │  (1,000 bootstrapped    │
          │   market histories)     │
          └─────────────────────────┘
```

## Module Responsibilities

| Module | Role |
|--------|------|
| `config.py` | Central config: paths, tickers, hyperparameters, constraints |
| `data_loader.py` | Parses Bloomberg CSVs (multi-level headers, macro data) |
| `features.py` | Returns computation, macro merge, normalization |
| `qp_solver.py` | Layer 1: QP tracking error minimization with OSQP |
| `gnn_model.py` | Layer 2: PyTorch models (LSTM + GAT architecture) |
| `gnn_train.py` | End-to-end training loop with Sharpe ratio loss function |
| `gnn_supervisor.py`| Inference and execution logic generating daily $\alpha_t$ |
| `synthetic_validation.py` | Robustness testing using stationary block bootstrap |
| `backtester.py` | Performance metrics (Sharpe, MaxDD, Calmar, Turnover) |
| `main.py` | Pipeline runner combining Worker and Supervisor outputs |

## Key Design Decisions

1. **Separation of Concerns**: The Worker knows nothing about regimes; it solely minimizes tracking error in normal conditions. The Supervisor knows nothing about individual stock selection; it solely modulates overall portfolio risk based on macro conditions.

2. **Continuous Blending ($\alpha_t$)**: The Supervisor outputs a smooth confidence signal $\alpha_t \in [0,1]$ rather than a binary regime switch, reducing turnover and enabling proportional hedging into cash during correlation spikes.

3. **Label-Free Training**: Unlike prior architectures that required noisy binary meta-labels (e.g., XGBoost), the GNN is trained end-to-end to directly maximize the differentiable annualized Sharpe ratio: $\mathcal{L} = -\text{Sharpe} \times \sqrt{252} + \lambda \cdot \overline{|\Delta\alpha|}$.

4. **Emergent Regime Detection**: Regime detection is not pre-defined by thresholds. The GAT attention mechanism learns which cross-asset relationships matter dynamically. The model autonomously drops $\alpha_t$ during crises (like COVID-19) as a consequence of Sharpe maximization.

5. **Robust Validation**: Strategy robustness is tested on 1,000 bootstrapped synthetic market histories (Chan 2018) combined with four-fold expanding walk-forward cross-validation.

## Data Flow

```text
Bloomberg Terminal Data (CSVs)
    ↓
data_loader.py & features.py → Cleaned Prices, Returns, Macro Features
    ↓
qp_solver.py → w_clone (Base Portfolio Weights)
    ↓
[Training Phase]
gnn_train.py → Learns LSTM + GAT weights to maximize Sharpe
    ↓
[Inference Phase]
gnn_supervisor.py → Daily $\alpha_t$ blending values
    ↓
synthetic_validation.py → 1,000 stress-tested market simulations
    ↓
backtester.py → Final portfolio performance metrics
```
