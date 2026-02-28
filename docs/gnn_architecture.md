# Dynamic GNN Supervisor — Architecture Documentation

> **Status:** Proposed  
> **Branch:** `feature/gnn-supervisor` (to be created)  
> **Reference:** CRISP — *Crisis-Resilient Investment through Spatio-temporal Patterns* (2024)  
> **Hardware:** GPU recommended (RTX 3090); CPU fallback supported

---

## 1. Motivation

The existing Phase 3 supervisor uses an **XGBoost classifier** trained on a binary meta-label: *"will the clone portfolio drawdown > 2% in the next 5 days?"* This has two fundamental limitations:

1. **Noisy short-horizon target.** Markets can drop 2% in 5 days even in healthy regimes. The training signal is weak.
2. **No relational structure.** XGBoost sees each feature independently. It cannot model how a stress event in one sector propagates to others — the mechanism behind real regime transitions.

The Dynamic GNN Supervisor addresses both by:
- Replacing the binary label with a **direct Sharpe-based loss** (no label engineering needed)
- Modelling the market as a **dynamic asset graph** where contagion propagates through learned edge weights

---

## 2. Architecture Overview

```
              INPUT: 20-day window of daily stock features
              ┌─────────────────────────────────────────┐
              │  For each of 33 stocks (32 CA + SPY):   │
              │  [return_1d, vol_5/21/63d, mom_5/21/63d,│
              │   volume_zscore, sector_onehot, beta,    │
              │   T10Y2Y, DXY, MOVE, IG_spread, YC_spr] │
              │  Shape per stock: [20 timesteps × ~28f]  │
              └──────────────┬──────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   LSTM Encoder  │  (per stock, shared weights)
                    │  [20, 28] → 32  │  Captures: how vol/momentum
                    └────────┬────────┘  arrived at current state
                             │
              ┌──────────────▼──────────────────────┐
              │        Multi-Head GAT (4 heads)      │
              │   Fully connected: 33×33 edges       │
              │   No hard threshold — attention       │
              │   weights learned end-to-end          │
              │   Emergent sparsity (~90% → 0)        │
              └──────────────┬──────────────────────┘
                             │
              ┌──────────────▼──────────────────────┐
              │        Global Mean Pool              │
              │   All 33 node embeddings → 1 vector  │
              └──────────────┬──────────────────────┘
                             │
              ┌──────────────▼──────────────────────┐
              │   Linear → ReLU → Linear → Sigmoid  │
              │   Output: α ∈ [0, 1]                 │
              │   Interpretation: market confidence   │
              └──────────────┬──────────────────────┘
                             │
              ┌──────────────▼──────────────────────┐
              │         Portfolio Blending           │
              │  r_t = α_t × r_clone + (1−α_t) × 0  │
              └─────────────────────────────────────┘
```

**Loss function** (trained directly on portfolio returns — no labels):

```
L = −Sharpe(portfolio) × √252  +  λ × Turnover(α)
  = −mean(r) / std(r) × √252  +  λ × mean|Δα|
```

where `λ = 0.01` controls the penalty for excessive switching.

---

## 3. Why Fully Connected Edges?

Traditional graph-based finance models pre-bake edges using a hard correlation threshold (e.g., connect stocks only if `corr > 0.3`). This fails during novel crises because:

- During the 2020 COVID crash, correlations spiked to 0.95+ between stocks that normally showed 0.2 correlation
- The *type* of connection that matters changes by regime: sector links dominate in normal markets, but liquidity links dominate in crises

By initializing a **fully connected graph** and letting the GAT learn attention weights, the model discovers which connections matter in which conditions. The CRISP paper observed 92.5% emergent sparsity — the network naturally zeroed out irrelevant edges during training. With only 33 nodes, the fully connected graph has only 1,056 edges, making this computationally trivial.

---

## 4. Node Features (per stock, per day)

| Feature | Dim | Source | Notes |
|---|---|---|---|
| Daily return | 1 | `stock_history.csv` → `PX_LAST` | Raw 1-day return |
| Rolling volatility (5/21/63d) | 3 | Computed | Annualized `std × √252` |
| Momentum (5/21/63d) | 3 | Computed | Cumulative return over window |
| Volume z-score (21d) | 1 | `stock_history.csv` → `PX_VOLUME` | `(vol - μ) / σ` over trailing 21d |
| Beta (static) | 1 | `stock_profiles.csv` → `EQY_BETA` | Market sensitivity, static |
| GICS sector one-hot | 8 | `stock_profiles.csv` → `GICS_SECTOR_NAME` | 8 unique sectors in universe |
| **Subtotal per-node** | **17** | | |
| Macro (global, shared across all nodes) | | | |
| T10Y2Y (yield curve) | 1 | `economic_indicators.csv` | In basis points |
| DXY level (USD index) | 1 | `economic_indicators.csv` | Dollar strength |
| MOVE level (bond vol) | 1 | `economic_indicators.csv` | Bond market stress |
| IG spread | 1 | `economic_indicators.csv` | Investment grade credit stress |
| YC spread + inversion flag | 2 | `yield_curve_spread.csv` | Recession signal |
| **Total per node** | **23** | | Appended to each node |

> **No Bloomberg pulls required.** All features are derived from existing project data files.

---

## 5. Training Protocol

### Walk-Forward Splits (no data leakage)

| Fold | Train Period | Validation | Test |
|---|---|---|---|
| 1 | 2010–2015 | 2016 | 2017 |
| 2 | 2010–2016 | 2017 | 2018 |
| 3 | 2010–2017 | 2018 | 2019 |
| 4 | 2010–2019 | 2020 | 2021–2024 |

### Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| Sequence length | 20 days | ~1 trading month; captures trend trajectory |
| LSTM hidden size | 32 | Small due to 33-node graph; prevents overfit |
| GAT heads | 4 | Multi-view attention |
| GAT hidden | 32 | Kept small for 33-node regime |
| Dropout | 0.2 | Regularize on small graph |
| Optimizer | AdamW | lr=1e-3, weight_decay=1e-4 |
| LR Schedule | CosineAnnealing | 50-epoch period |
| Early stopping | Patience=10 | On validation Sharpe |
| λ turnover | 0.01 | Light penalty; regime shifts should be allowed |

---

## 6. New Files

| File | Purpose |
|---|---|
| `src/gnn_data.py` | Node feature engineering, sliding windows, fully-connected edge construction, `GNNDataset` |
| `src/gnn_model.py` | `DynamicGNNSupervisor` model class (LSTM + GAT), Sharpe loss function |
| `src/gnn_train.py` | Walk-forward training loop, checkpointing, evaluation |
| `src/gnn_supervisor.py` | Inference pipeline — loads checkpoint, produces daily α, integrates with Phase 3 |

### Modified Files

| File | Change |
|---|---|
| `main.py` | `USE_GNN_SUPERVISOR` flag in `phase3()` — defaults to `True` once trained; falls back to XGBoost if no checkpoint |

---

## 7. Integration with Existing Pipeline

```
main.py
  └── phase3()
        ├── [USE_GNN_SUPERVISOR=True]  → gnn_supervisor.run_gnn_supervisor_pipeline()
        │        └── loads gnn_checkpoint_fold4.pt
        │        └── generates α_t for each day in test period
        │        └── blends clone + cash using α
        │
        └── [USE_GNN_SUPERVISOR=False] → supervisor.run_supervisor_pipeline()  [unchanged]
```

The GNN supervisor returns the **same interface** as the existing supervisor:
```python
supervised_returns, alpha, model = run_gnn_supervisor_pipeline(...)
#    ↑ pd.Series         ↑ pd.Series    ↑ nn.Module
```

This means all downstream phases (SHAP, ablation, synthetic validation) remain unchanged.

---

## 8. Key Differences: XGBoost vs GNN Supervisor

| Dimension | XGBoost (current) | GNN (proposed) |
|---|---|---|
| **Training target** | Binary: drawdown > 2% in 5d | Portfolio Sharpe (direct) |
| **Input structure** | Tabular (flat feature vector) | Graph (per-node sequences + edges) |
| **Cross-asset info** | None — each feature is marginal | GAT message-passing across all pairs |
| **Temporal horizon** | 5 trading days | 20-day lookback → regime-level |
| **Interpretability** | SHAP feature importance | Attention weights per edge per day |
| **Regime adaptation** | Train once, static weights | Attention weights shift daily |
| **Training signal** | Noisy binary labels | Smooth portfolio return signal |

---

## 9. Evaluation Criteria

After training, the GNN earns its place in the pipeline if:

1. **GNN Sharpe ≥ XGBoost Sharpe** on the 2020–2024 test period (held-out fold 4)
2. **Regime alignment:** α drops sharply during known stress events:
   - March 2020 (COVID crash) → expect α < 0.3
   - Jan–Oct 2022 (inflation shock) → expect α < 0.5
3. **Turnover control:** mean daily |Δα| < 0.05 (no excessive switching)
4. **Attention sparsity:** after training, most edge weights should collapse near zero (validates CRISP finding)

---

## 10. Dependencies

```bash
# PyTorch (CPU or GPU)
pip install torch torchvision

# PyTorch Geometric
pip install torch-geometric

# For GPU (RTX 3090, CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

All other dependencies (`pandas`, `numpy`, `cvxpy`, `xgboost`) are already in the project environment.

---

*Document last updated: February 27, 2026*
