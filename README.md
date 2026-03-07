# Portfolio Optimizer — Hierarchical Regime-Conditional CPO

> **CUCAI 2026** | Western University
> 
> *Dynamic Graph Attention for Regime-Conditional Convex Portfolio Allocation*

A hierarchical portfolio optimization framework that pairs a quadratic programming Worker with a learned GNN Supervisor. The Worker constructs a diversified Canadian equity clone of the S&P 500; the GNN Supervisor dynamically modulates portfolio aggressiveness based on inter-asset contagion patterns detected through graph attention.

## Key Results

| Metric | GNN Supervisor | Worker Only | SPY Buy & Hold |
|--------|---------------|-------------|----------------|
| **Ann. Return** | 13.77% | 14.48% | 16.74% |
| **Sharpe Ratio** | **1.052** | 0.691 | 0.745 |
| **Max Drawdown** | **−17.74%** | −37.80% | −34.10% |
| **Calmar Ratio** | **0.776** | 0.383 | 0.491 |

The GNN achieves the highest risk-adjusted return with **53% drawdown reduction** versus the unsupervised baseline, placing it within institutional risk mandates (≤20% max DD).

## Architecture

```
Layer 1 (Worker)     → Rolling QP solver minimizes tracking error to SPY
Layer 2 (Supervisor) → LSTM temporal encoder + Multi-Head GAT over 33-node asset graph
Execution            → r_t = α_t × r_clone + (1 − α_t) × r_cash
Validation           → Walk-forward CV (4 folds) + 1,000 synthetic market histories
```

## Setup

```bash
# Clone the repository
git clone https://github.com/Western-Artificial-Intelligence/-Portfolio-Optimizer-Hierarchical-Regime-Conditional-CPO.git
cd -Portfolio-Optimizer-Hierarchical-Regime-Conditional-CPO

# Install dependencies
pip install -r requirements.txt
```

### Data

All market data sourced from **Bloomberg Terminal** at Ivey Business School:

https://drive.google.com/drive/folders/1cgXNTCe2qkbtR56CPsRL1uga2PDuqypV

Download the CSVs and place them in the `data/` directory.

- **32 TSX equities** + SPY benchmark (2010–2026)
- **Macro indicators**: VIX, yield curve, credit spreads, MOVE, DXY
- Fields: PX_LAST, TOT_RETURN_INDEX_GROSS_DVDS, PX_VOLUME, CUR_MKT_CAP, PE_RATIO

## Run

```bash
python main.py
```

This runs the full pipeline:
1. **Phase 1** — Data loading & EDA
2. **Phase 2** — Rolling QP optimization (Worker)
3. **Phase 3** — GNN Supervisor (trains if no checkpoint found, then evaluates)
4. **Phase 4** — Ablation study
5. **Phase 5** — Synthetic validation (Chan 2018)

Results are saved to `results/`.

### Additional Scripts

| Script | Purpose |
|--------|---------|
| `evaluate_folds.py` | Evaluate GNN across all 4 walk-forward folds |
| `bootstrap_ci.py` | Compute bootstrap confidence intervals for Sharpe ratios |
| `generate_paper_plots.py` | Regenerate paper figures |
| `generate_poster_plots.py` | Generate CUCAI poster figures |
| `generate_attention_map.py` | Visualize GNN attention weights |

## Project Structure

```
├── data/                        # Bloomberg CSVs (not committed — see link above)
├── docs/
│   ├── paper.tex                # Conference paper (LaTeX)
│   ├── architecture.md          # System design overview
│   └── gnn_architecture.md      # GNN architecture details
├── src/
│   ├── config.py                # Central configuration
│   ├── data_loader.py           # Bloomberg data parsing
│   ├── features.py              # Feature engineering
│   ├── qp_solver.py             # Layer 1: QP tracking error minimization
│   ├── gnn_model.py             # Layer 2: LSTM + GAT model
│   ├── gnn_data.py              # Graph dataset construction
│   ├── gnn_train.py             # Walk-forward training loop
│   ├── gnn_supervisor.py        # GNN inference pipeline
│   ├── supervisor.py            # XGBoost baseline (tabular comparison)
│   ├── backtester.py            # Performance metrics
│   ├── benchmarks.py            # Benchmark strategies (VIX, 60/40, Equal-Weight)
│   └── synthetic_validation.py  # Chan 2018 bootstrap robustness testing
├── results/                     # GNN checkpoints & output metrics
├── experimental/                # Exploratory work (SPY mirroring, derivatives)
├── tests/                       # Unit tests
├── main.py                      # Pipeline entry point
├── requirements.txt
└── LICENSE
```

## Key References

- Chan, E. (2023). "How to Use Machine Learning for Optimization" (CPO)
- Chan, E. (2018). "Optimizing Trading Strategies without Overfitting" (Synthetic Validation)
- López de Prado, M. (2018). *Advances in Financial Machine Learning* (Meta-Labeling)
- Li, Z. & Fan, R. (2024). "Crisis-Resilient Investment through Spatio-temporal Patterns" (CRISP)

## Citation

```bibtex
@inproceedings{leite2026dynamicgraph,
  title     = {Dynamic Graph Attention for Regime-Conditional Convex Portfolio Allocation},
  author    = {Leite, Henrique and Huynh, Micky and Schultz, Nathan and Khamissa, Hamza and Kostesku, Noah and Li, Sheng Chang and Yousseff, Hadi},
  booktitle = {Canadian Undergraduate Conference on AI (CUCAI)},
  year      = {2026}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
