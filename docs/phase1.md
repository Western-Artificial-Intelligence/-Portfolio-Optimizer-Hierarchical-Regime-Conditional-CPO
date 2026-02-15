# Phase 1: Data Preparation & Exploration — Report

## Overview

Phase 1 built the data pipeline for the Hierarchical CPO system. It loads, cleans, and analyzes 16 years of Bloomberg data (2010–2026) covering **32 equities** (TSX 60 constituents + SPY benchmark) and **macroeconomic regime indicators**.

### Files Created

| File | Purpose |
|------|---------|
| `src/config.py` | Central config — paths, 32 tickers, date splits, constraints |
| `src/data_loader.py` | Parses all 4 Bloomberg CSVs (incl. multi-level stock header) |
| `src/features.py` | Returns, macro merge, rolling stats, sparse ticker filter |
| `src/eda.py` | Correlation heatmap, cumulative returns, vol, regime plots |
| `main.py` | Pipeline runner |

---

## Data Summary

| Dataset | Rows | Range | Notes |
|---------|------|-------|-------|
| Stock History | 4,122 trading days | 2010-01-04 → 2026-02-04 | 32 tickers × 8 fields |
| Stock Profiles | 32 tickers | Static | GICS sector, beta, analyst rating |
| Economic Indicators | 4,199 rows | 2010-01-01 → 2026-02-04 | IG/HY spreads start late (~440 NaN rows) |
| Yield Curve | 4,197 rows | 2010-01-01 → 2026-02-04 | **574 inversion days (13.7%)** |

All 32 tickers have ≥50% data coverage. Later-IPO stocks (SHOP 66.6%, QSR 69.3%, KXS 72.5%) have intentional gaps.

---

## Output Analysis

### 1. Correlation Heatmap

**Key observations:**
- **Banks cluster tightly** (RY, TD, BMO, BNS — correlations ~0.6–0.8) — expected for same-sector exposure
- **Gold miners (AEM, ABX) have low correlation** to the rest of the universe — useful diversifiers
- **SPY correlates moderately** with most Canadian stocks (~0.3–0.5) — this is the gap the QP solver needs to close
- **Tech stocks (SHOP, CSU, CLS)** show moderate inter-correlation but lower correlation to financials

### 2. Cumulative Returns

**Key observations:**
- **CSU (Constellation Software)** is the standout: ~140x return since 2010
- **DOL, CLS, ATD** all significantly outperformed SPY over the full period
- **SPY** had a solid ~5x return — this is the benchmark the QP Worker must track
- The divergence in terminal values confirms that **stock selection and weighting matter enormously**

### 3. Rolling Volatility

**Key observations:**
- **COVID crash (March 2020)** is the dominant volatility event — SPY vol spiked to ~60%, portfolio mean vol to ~80%
- Canadian stocks are **structurally more volatile** (~20–25% annualized) than SPY (~10–15%)
- **Cross-sectional dispersion increases during crises** — exactly when the AI Supervisor should intervene
- Volatility clustering is evident: calm periods (2013–2014, 2017) vs. turbulence (2011, 2015, 2020, 2022)

### 4. Regime Indicators

**Key observations:**
- **MOVE Index** spikes during: Euro crisis (2011), Fed taper tantrum (2013), COVID (2020), SVB crisis (2023)
- **DXY** shows a structural uptrend from ~77 to ~110 — important for CAD/USD effects on the portfolio
- **Yield curve inversion** (2022–2023) is clearly visible — lasted ~2 years, historically associated with recession risk
- **Credit spreads (IG)** spike during stress periods — key features for the Supervisor

### 5. Summary Statistics

| Ticker | Ann Ret (%) | Ann Vol (%) | Sharpe | Max DD (%) |
|--------|-------------|-------------|--------|------------|
| DOL | 26.8 | 22.7 | **1.18** | -46.6 |
| CSU | 29.1 | 25.6 | **1.14** | -55.8 |
| SHOP | 52.1 | 57.0 | 0.91 | -83.5 |
| SPY | 12.5 | 17.1 | **0.73** | -34.1 |
| BCE | 2.6 | 16.3 | 0.16 | -60.3 |

---

## Implications for Phase 2 (Worker / QP Solver)

1. The moderate correlations between Canadian stocks and SPY (~0.3–0.5) confirm that tracking is **feasible but non-trivial**
2. Sector concentration risk is real — banks dominate the TSX, so the **15% cap constraint** is critical
3. The data spans multiple regimes (bull, bear, COVID, rate-hike cycle) — good for robust optimization
4. Later-IPO stocks (SHOP, QSR, KXS) will need special handling during walk-forward windows where they have no data
