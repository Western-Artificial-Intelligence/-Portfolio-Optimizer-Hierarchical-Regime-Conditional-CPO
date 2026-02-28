# GNN Supervisor — Optimization & SP Mirror Integration Roadmap

**Current state:** GNN v2 trained, val Sharpe 3.024, live Sharpe 0.929, Max DD −12.65%.
This document tracks what to do next.

---

## Part 1 — GNN Model Optimizations

### 1.1 Add US Market Nodes to the Graph (High Impact)

**Problem:** The GNN currently runs on 32 Canadian stocks. The regime signals it learns are Canadian-market-specific. For an SP Mirror supervisor, we need US market structure.

**Fix:** Add SPY sector ETFs as additional nodes in the graph. The SP Mirror already loads these as optional data:
```
S5INFT (Tech), S5HLTH (Health), S5FINL (Financials), S5COND (Consumer Disc.),
S5CONS (Consumer Staples), S5INDU (Industrials), S5MATR (Materials),
S5ENER (Energy), S5UTL (Utilities), S5TELS (Telecom)
```
The GAT attention would then learn Canadian↔US cross-sector contagion paths (e.g., Canadian bank stress → US financial stress → regime shift). This is the single highest-value improvement.

**Implementation:** Modify `gnn_data.py → build_graph_dataset()` to accept an optional `us_sector_df` argument and append sector ETF nodes to the feature matrix. Change `n_nodes` from 32 → 42.

---

### 1.2 Dynamic (Learned) Graph Edges — Replace Fully-Connected

**Problem:** The current graph is fully connected (all 992 edges active always). This is computationally expensive and noisy for training.

**Fix (CRISP-inspired):** Learn an adjacency matrix dynamically using a Gumbel-Softmax relaxation. On each forward pass, the model samples a sparse edge set based on current node features. This is the core CRISP innovation.

```python
# In gnn_model.py — replace static edges with learned adjacency
adj_logits = self.edge_predictor(node_emb)   # [N, N]
adj_soft   = gumbel_softmax(adj_logits, tau=0.5, hard=False)
```

**Expected benefit:** Sparse attention → easier to interpret, faster training, more generalizable.

---

### 1.3 Extend Training Window: 20d → 30d

**Problem:** 20-day LSTM windows may miss slow-moving macro regime shifts (yield curve inversions typically unfold over months).

**Fix:** Change `WINDOW = 20` → `WINDOW = 30` in `gnn_train.py`. Requires retraining. Adds ~15% compute.

---

### 1.4 Regime-Weighted Loss

**Problem:** The Sharpe loss weights all days equally. But getting the regime call right during a crash (5% of days) matters far more than normal days.

**Fix:** Add a sample weight to the loss:
```python
# Weight crash days 5× higher
weights = 1 + 4 * (portfolio_returns < portfolio_returns.quantile(0.10)).float()
loss = (-sharpe + λ_turnover * turnover + λ_defensive * defensive_penalty) * weights.mean()
```

---

### 1.5 Train on Longer History (Include 2008 GFC)

**Problem:** The training data starts Jan 2010, missing the 2008 Global Financial Crisis — the worst regime shift in modern history.

**Fix:** Pull stock history back to Jan 2007 from Bloomberg. Gives the model 2 full crisis periods (GFC + COVID) to learn from instead of 1.

---

### 1.6 Multi-fold Ensemble Instead of Single Fold

**Problem:** Production uses only Fold 4's checkpoint. A single model can have lucky/unlucky initialization.

**Fix:** Load all 4 fold checkpoints and average their α outputs:
```python
alpha_ensemble = mean([fold1.alpha, fold2.alpha, fold3.alpha, fold4.alpha])
```
Takes 4× inference time (still sub-second) but gives more stable regime signals.

---

## Part 2 — SP Mirror Integration

### Architecture Overview

The dev's `apply_supervisor()` already has the integration hook:
```python
# In SPMirrorConfig (already coded by dev):
supervisor_mode: str = "none"   # ← change to "cash"
prob_series:     pd.Series = None  # ← plug in GNN α here
```

The `apply_supervisor()` function handles `mode="cash"` by scaling returns:
```python
# apply_supervisor() at mode="cash":
final_return = prob_series * sp_mirror_return
```

**The GNN α (0.30–1.0) maps directly to SP Mirror exposure.** No architecture change needed.

---

### 2.1 Plug-In Integration (No Retraining, Works Now)

Use the existing GNN α as a transfer signal on top of SP Mirror:

```python
# In run_sp_mirror.py (or wherever SP Mirror is called from main)
from src.gnn_supervisor import run_gnn_supervisor_pipeline

# Run GNN to get regime signal
_, alpha_series, _ = run_gnn_supervisor_pipeline(
    clone_returns, prices_clean, all_fields, profiles, econ, yield_curve,
    fold=4, verbose=False
)

# Feed into SP Mirror config
config = SPMirrorConfig(
    supervisor_mode="cash",
    prob_series=alpha_series,    # ← GNN α replaces VIX rule
)

# SP Mirror now uses GNN α to scale overlay exposure
sp_mirror_returns = compute_sp_mirror_returns(
    clone_return, r_ES, r_USDCAD, bench_return, config, rebal_dates
)
```

**This replaces the static VIX threshold rule** with the GNN's learned regime signal. No retraining of the GNN needed — the regime signals generalize because cross-asset correlation spikes are global.

---

### 2.2 Retrain GNN on SP Mirror Returns (Proper Integration)

For a fully calibrated supervisor, retrain the GNN with SP Mirror returns in the Sharpe loss instead of QP clone returns:

```python
# In gnn_train.py — change the blending target:
# Instead of: portfolio_ret = alpha * clone_returns_batch
# Use:        portfolio_ret = alpha * sp_mirror_returns_batch
```

This calibrates α specifically to ES1 futures volatility (higher vol → needs different α scaling than the quieter Canadian equity portfolio).

**Note:** Requires SP Mirror historical returns to be computed first and stored as a precomputed Series.

---

### 2.3 Add US Sector Nodes for SP Mirror (Combined with 1.1)

If we add S5INFT/FINL/etc. as graph nodes (roadmap item 1.1), the model will directly attend to US sector correlations when predicting regimes — making it far more appropriate for supervising SPY-tracking strategies.

```
Graph topology (after enhancement):
  32 Canadian stocks  +  10 US sector ETFs  →  42 nodes
  Feature: ES1 momentum  +  USDCAD vol  →  added to macro features
```

---

## Part 3 — Implementation Priority Order

| Priority | Item | Effort | Expected Impact |
|---|---|---|---|
| **P1** | SP Mirror plug-in (2.1) | 1–2h | Test GNN on SP Mirror immediately |
| **P1** | Multi-fold ensemble (1.6) | 30 min | Stabler α, no retraining |
| **P2** | US sector nodes (1.1) | 3–4h + retrain | Biggest quality improvement |
| **P2** | Retrain on SP Mirror returns (2.2) | 2h + retrain | Calibrated supervisor |
| **P3** | Dynamic edges (1.2) | 1–2 days | CRISP full implementation |
| **P3** | Regime-weighted loss (1.4) | 2h + retrain | Better crisis detection |
| **P3** | Extend to GFC data (1.5) | Data pull + retrain | More robust model |

---

## Retraining Now

The current GNN v2 already has the α floor + defensiveness penalty + 2019 val fold fixes.
To retrain with these improvements active:

```bash
python src/gnn_train.py
```

Output goes to `results5/`. After training, run:
```bash
python main.py
```

Results go to `results5/phase3_gnn_comparison.csv` and `results5/gnn_alpha_over_time.png` (two-panel: α + GNN vs SPY cumulative returns).
