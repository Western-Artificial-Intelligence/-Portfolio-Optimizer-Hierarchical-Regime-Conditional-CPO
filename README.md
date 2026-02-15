# Portfolio Optimizer — Hierarchical Regime-Conditional CPO

> **CUCAI 2026** | Western University

A hierarchical portfolio optimization system that combines classical convex optimization with machine learning-based regime detection. The Worker (QP solver) constructs the portfolio; the Supervisor (XGBoost meta-labeling) adapts exposure based on macroeconomic conditions.

## Architecture

```
Layer 1 (Worker)     → Rolling QP solver minimizes tracking error
Layer 2 (Supervisor) → XGBoost outputs confidence P ∈ [0,1]
Execution            → W_final = P × W_aggressive + (1-P) × W_defensive
Validation           → Tested on 1,000 synthetic market histories
```

## Setup

```bash
# Install dependencies with uv (fast)
uv sync

# Or with pip (slower)
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

This runs the full pipeline:
1. **Phase 1** — Data loading & EDA
2. **Phase 2** — Rolling QP optimization (Worker)
3. **Phase 3** — AI Supervisor (meta-labeling + continuous blending)

Results are saved to `results/`.

## Project Structure

```
PortfolioOptimizer-pt-2/
├── data/                    # Bloomberg CSVs (not committed)
│   ├── stock_history.csv
│   ├── stock_profiles.csv
│   ├── economic_indicators.csv
│   └── yield_curve_spread.csv
├── docs/
│   ├── paper.tex            # Conference paper
│   ├── architecture.md      # System design
│   ├── goals.md             # Project roadmap & to-do
│   └── phase1-3.md          # Phase reports
├── src/
│   ├── config.py            # Central configuration
│   ├── data_loader.py       # Bloomberg data parsing
│   ├── features.py          # Feature engineering
│   ├── forecaster.py        # Uncertainty features (vol proxies)
│   ├── qp_solver.py         # Layer 1: QP tracking error minimization
│   ├── supervisor.py        # Layer 2: XGBoost meta-labeling
│   ├── backtester.py        # Performance metrics
│   └── eda.py               # Visualization
├── results/                 # Output plots & CSVs
├── main.py                  # Pipeline runner
├── requirements.txt
└── .gitignore
```

## Data

All market data sourced from **Bloomberg Terminal** at Ivey Business School:

https://drive.google.com/drive/folders/1cgXNTCe2qkbtR56CPsRL1uga2PDuqypV


- **30+ TSX 60 equities** + SPY benchmark (2010–2026)
- **Macro indicators**: VIX, yield curve, credit spreads, MOVE, DXY
- Fields: PX_LAST, TOT_RETURN_INDEX_GROSS_DVDS, PX_VOLUME, CUR_MKT_CAP, PE_RATIO

## Key References

- Chan, E. (2023). "How to Use Machine Learning for Optimization" (CPO)
- Chan, E. (2018). "Optimizing Trading Strategies without Overfitting" (Synthetic Validation)
- López de Prado, M. (2018). *Advances in Financial Machine Learning* (Meta-Labeling)
