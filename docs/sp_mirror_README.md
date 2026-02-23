# SP_mirror – Derivatives Overlay

A single entrypoint runs overlay variants against both benchmarks:

| Benchmark | Index tracked | Required CSV |
|-----------|--------------|--------------|
| **SPY_Total_Returns** | S&P 500 Total Return Index (dividends reinvested) | `SPXT__PX_LAST.csv` |
| **SPY** | S&P 500 Price Index | `SPX__PX_LAST.csv` |

The overlay engine lives in `src/sp_mirror_SPY_Total_Returns.py`. The SPY module (`src/sp_mirror_SPY.py`) is a thin wrapper that swaps the benchmark column.

All results are saved to `results/Derivatives_cloning/`.

## Run

```bash
python run_sp_mirror.py
```

## Source files

```
src/
├── sp_mirror_SPY_Total_Returns.py    # Full overlay engine (configurable benchmark)
└── sp_mirror_SPY.py                  # Convenience wrapper for SPY benchmark

run_sp_mirror.py                      # Single entrypoint – runs both benchmarks
```

## Output

```
results/Derivatives_cloning/
├── sp_mirror_SPY_Total_Returns_comparison.png    # SPY_Total_Returns vs Clone vs SP_mirror variants
├── sp_mirror_SPY_Total_Returns_metrics.csv       # Metrics table vs SPY_Total_Returns
├── sp_mirror_SPY_comparison.png                  # SPY vs Clone vs SP_mirror variants
└── sp_mirror_SPY_metrics.csv                     # Metrics table vs SPY
```

## Required Bloomberg CSVs

Place Bloomberg BDH-exported CSV files in `data/derivatives/` — one file per ticker/field. Filenames use double underscores (e.g. `ES1__PX_SETTLE.csv`).

| Filename | Required by | Description |
|----------|-------------|-------------|
| `ES1__PX_SETTLE.csv` | Both | E-mini S&P 500 futures settle |
| `USDCAD__PX_LAST.csv` | Both | Spot USD/CAD FX |
| `SPXT__PX_LAST.csv` | SPY_Total_Returns benchmark | S&P 500 Total Return Index |
| `SPX__PX_LAST.csv` | SPY benchmark | S&P 500 Price Index |

## Optional CSVs

- Macro: `VIX__PX_LAST`, `VIX3M__PX_LAST`, `MOVE__PX_LAST`, `DXY__PX_LAST`, `USGG2YR__PX_LAST`, `USGG10YR__PX_LAST`, `Yield_Spread__PX_LAST_USGG10YR-USGG2YR`
- S&P sector indices: `S5INFT__PX_LAST`, `S5HLTH__PX_LAST`, `S5FINL__PX_LAST`, `S5COND__PX_LAST`, `S5CONS__PX_LAST`, `S5INDU__PX_LAST`, `S5MATR__PX_LAST`, `S5ENER__PX_LAST`, `S5UTL__PX_LAST`, `S5TELS__PX_LAST`

## CSV format

Each CSV should have:

- A **date** column (header `date`, `Date`, or first column parsed as datetime)
- A **value** column (numeric); the loader uses the last numeric column or a column matching the filename stem

```text
date,ES1__PX_SETTLE
2010-01-04,1125.5
2010-01-05,1128.25
...
```

Date range: **2010-01-04 to 2026-02-04** (daily) is expected for full backtest.
