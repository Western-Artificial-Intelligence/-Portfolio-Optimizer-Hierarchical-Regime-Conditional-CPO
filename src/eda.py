"""
Exploratory Data Analysis – visualizations and summary statistics.

All plots are saved to the results/ directory.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from src.config import RESULTS_DIR, FIGURE_DPI, FIGURE_SIZE


def _save(fig, name):
    """Save figure and close."""
    path = RESULTS_DIR / f"{name}.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[eda] Saved: {path}")


def plot_correlation_heatmap(returns, benchmark_col="SPY US Equity"):
    """
    Heatmap of pairwise correlations between all stocks and SPY.
    """
    corr = returns.corr()

    fig, ax = plt.subplots(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr, mask=mask, cmap=cmap, center=0,
        vmin=-1, vmax=1,
        square=True, linewidths=0.5,
        cbar_kws={"shrink": 0.7, "label": "Correlation"},
        ax=ax,
        annot=False,
        xticklabels=[t.replace(" CN Equity", "").replace(" US Equity", " (SPY)")
                     for t in corr.columns],
        yticklabels=[t.replace(" CN Equity", "").replace(" US Equity", " (SPY)")
                     for t in corr.columns],
    )
    ax.set_title("Pairwise Return Correlations – TSX Universe + SPY",
                 fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)

    _save(fig, "correlation_heatmap")


def plot_cumulative_returns(prices, benchmark_col="SPY US Equity", top_n=10):
    """
    Cumulative returns for the benchmark and top-N Canadian stocks.
    """
    # Normalize to 100 at start
    normed = prices.div(prices.iloc[0]) * 100

    # Pick top-N by total return
    final_returns = normed.iloc[-1].sort_values(ascending=False)
    top = final_returns.head(top_n).index.tolist()

    # Always include SPY
    if benchmark_col not in top and benchmark_col in normed.columns:
        top.append(benchmark_col)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    for ticker in top:
        label = ticker.replace(" CN Equity", "").replace(" US Equity", " (SPY)")
        lw = 2.5 if "SPY" in ticker else 1.2
        alpha = 1.0 if "SPY" in ticker else 0.75
        ax.plot(normed.index, normed[ticker], label=label, linewidth=lw, alpha=alpha)

    ax.set_title("Cumulative Returns – Top Performers vs SPY",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Indexed (100 = start)")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)

    _save(fig, "cumulative_returns")


def plot_rolling_volatility(returns, benchmark_col="SPY US Equity", window=63):
    """
    Rolling 3-month annualized volatility for all stocks.
    """
    vol = returns.rolling(window).std() * np.sqrt(252)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Panel 1: SPY vol
    if benchmark_col in vol.columns:
        ax = axes[0]
        spy_vol = vol[benchmark_col]
        ax.plot(spy_vol.index, spy_vol.values, color="#e74c3c", linewidth=1.5)
        ax.fill_between(spy_vol.index, 0, spy_vol.values, alpha=0.15, color="#e74c3c")
        ax.set_title(f"SPY – Rolling {window}-day Annualized Volatility",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("Volatility")
        ax.grid(True, alpha=0.3)

    # Panel 2: Canadian stocks (mean ± 1 std band)
    canadian = vol.drop(columns=[benchmark_col], errors="ignore")
    ax = axes[1]
    mean_vol = canadian.mean(axis=1)
    std_vol = canadian.std(axis=1)

    ax.plot(mean_vol.index, mean_vol.values, color="#2980b9", linewidth=1.5,
            label="Mean Canadian vol")
    ax.fill_between(mean_vol.index,
                    (mean_vol - std_vol).values,
                    (mean_vol + std_vol).values,
                    alpha=0.2, color="#2980b9", label="± 1 std")
    ax.set_title(f"Canadian Universe – Mean Rolling {window}-day Volatility",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Volatility")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, "rolling_volatility")


def plot_regime_indicators(econ, yield_curve):
    """
    Multi-panel plot of macro regime indicators:
    VIX proxy (MOVE), DXY, yield curve spread, and inversion flag.
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)

    # Panel 1: MOVE Index (bond vol, proxy for market stress)
    ax = axes[0]
    col = [c for c in econ.columns if "MOVE" in c.upper()]
    if col:
        move = econ[col[0]].dropna()
        ax.plot(move.index, move.values, color="#8e44ad", linewidth=1)
        ax.fill_between(move.index, 0, move.values, alpha=0.15, color="#8e44ad")
        ax.set_title("MOVE Index (Bond Volatility)", fontsize=12, fontweight="bold")
        ax.set_ylabel("MOVE")
        ax.grid(True, alpha=0.3)

    # Panel 2: DXY
    ax = axes[1]
    col = [c for c in econ.columns if "DXY" in c.upper()]
    if col:
        dxy = econ[col[0]].dropna()
        ax.plot(dxy.index, dxy.values, color="#27ae60", linewidth=1)
        ax.set_title("US Dollar Index (DXY)", fontsize=12, fontweight="bold")
        ax.set_ylabel("DXY")
        ax.grid(True, alpha=0.3)

    # Panel 3: Yield Curve Spread
    ax = axes[2]
    if "YIELD_CURVE_SPREAD" in yield_curve.columns:
        spread = yield_curve["YIELD_CURVE_SPREAD"].dropna()
        ax.plot(spread.index, spread.values, color="#2c3e50", linewidth=1)
        ax.axhline(0, color="red", linestyle="--", alpha=0.6, label="Zero line")
        ax.fill_between(spread.index, 0, spread.values,
                        where=spread.values < 0,
                        alpha=0.3, color="#e74c3c", label="Inverted")
        ax.set_title("10Y−2Y Yield Curve Spread", fontsize=12, fontweight="bold")
        ax.set_ylabel("Spread (%)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Panel 4: Credit Spreads
    ax = axes[3]
    ig_col = [c for c in econ.columns if "IG" in c.upper()]
    hy_col = [c for c in econ.columns if "HY" in c.upper()]
    if ig_col:
        ig = econ[ig_col[0]].dropna()
        ax.plot(ig.index, ig.values, color="#3498db", linewidth=1, label="IG Spread")
    if hy_col:
        hy = econ[hy_col[0]].dropna()
        ax.plot(hy.index, hy.values, color="#e67e22", linewidth=1, label="HY Spread")
    ax.set_title("Credit Spreads", fontsize=12, fontweight="bold")
    ax.set_ylabel("Spread (bps)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, "regime_indicators")


def print_summary_stats(returns, benchmark_col="SPY US Equity"):
    """
    Print a formatted table of summary statistics per ticker.
    """
    stats = pd.DataFrame({
        "Ann Return (%)": returns.mean() * 252 * 100,
        "Ann Vol (%)": returns.std() * np.sqrt(252) * 100,
        "Sharpe": (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
        "Skewness": returns.skew(),
        "Kurtosis": returns.kurtosis(),
        "Max DD (%)": _max_drawdown(returns) * 100,
        "Coverage (%)": returns.notna().mean() * 100,
    })

    stats = stats.sort_values("Sharpe", ascending=False)

    # Clean ticker names for display
    stats.index = [
        t.replace(" CN Equity", "").replace(" US Equity", " (SPY)")
        for t in stats.index
    ]

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS (annualized)")
    print("=" * 80)
    print(stats.round(2).to_string())
    print("=" * 80)

    # Save to CSV
    path = RESULTS_DIR / "summary_stats.csv"
    stats.round(4).to_csv(path)
    print(f"\n[eda] Saved stats: {path}")

    return stats


# ── Phase 2 Plots ─────────────────────────────────────────


def plot_clone_vs_spy(clone_returns, spy_returns):
    """
    Cumulative returns of the Canadian Clone vs SPY.
    """
    common = clone_returns.index.intersection(spy_returns.index)
    clone = clone_returns.loc[common]
    spy = spy_returns.loc[common]

    cum_clone = (1 + clone).cumprod() * 100
    cum_spy = (1 + spy).cumprod() * 100

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.plot(cum_spy.index, cum_spy.values, color="#e74c3c", linewidth=2,
            label="SPY (Benchmark)", alpha=0.9)
    ax.plot(cum_clone.index, cum_clone.values, color="#2980b9", linewidth=2,
            label="Canadian Clone", alpha=0.9)

    ax.fill_between(cum_clone.index, cum_clone.values, cum_spy.values,
                    where=cum_clone.values >= cum_spy.values,
                    alpha=0.1, color="#2980b9", label="Clone outperforms")
    ax.fill_between(cum_clone.index, cum_clone.values, cum_spy.values,
                    where=cum_clone.values < cum_spy.values,
                    alpha=0.1, color="#e74c3c", label="SPY outperforms")

    ax.set_title("Canadian Clone vs SPY – Cumulative Returns",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Indexed (100 = start)")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)

    _save(fig, "clone_vs_spy")


def plot_clone_spy_equalweight(clone_returns, spy_returns, ew_returns):
    """
    Cumulative returns: Canadian Clone (QP), SPY Buy & Hold, and Equal-Weight TSX.
    """
    common = clone_returns.index.intersection(spy_returns.index).intersection(ew_returns.index)
    clone = clone_returns.loc[common]
    spy = spy_returns.loc[common]
    ew = ew_returns.loc[common]

    cum_clone = (1 + clone).cumprod() * 100
    cum_spy = (1 + spy).cumprod() * 100
    cum_ew = (1 + ew).cumprod() * 100

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.plot(cum_spy.index, cum_spy.values, color="#e74c3c", linewidth=2,
            label="SPY Buy & Hold", alpha=0.9)
    ax.plot(cum_clone.index, cum_clone.values, color="#2980b9", linewidth=2,
            label="Canadian Clone (QP)", alpha=0.9)
    ax.plot(cum_ew.index, cum_ew.values, color="#27ae60", linewidth=2,
            label="Equal-Weight TSX", alpha=0.9)

    ax.set_title("Canadian Clone (QP), SPY Buy & Hold, and Equal-Weight TSX – Cumulative Returns",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Indexed (100 = start)")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)

    _save(fig, "clone_spy_equalweight_tsx")


def plot_tracking_error(clone_returns, spy_returns, window=63):
    """
    Rolling tracking error between Clone and SPY.
    """
    common = clone_returns.index.intersection(spy_returns.index)
    diff = clone_returns.loc[common] - spy_returns.loc[common]
    rolling_te = diff.rolling(window).std() * np.sqrt(252)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.plot(rolling_te.index, rolling_te.values, color="#8e44ad", linewidth=1.2)
    ax.fill_between(rolling_te.index, 0, rolling_te.values,
                    alpha=0.15, color="#8e44ad")

    # Highlight high-TE periods
    mean_te = rolling_te.mean()
    ax.axhline(mean_te, color="#e74c3c", linestyle="--", alpha=0.6,
               label=f"Mean TE: {mean_te:.1%}")

    ax.set_title(f"Rolling {window}-day Annualized Tracking Error",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Tracking Error")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)

    _save(fig, "tracking_error")


def plot_weight_evolution(weights_history, top_n=10):
    """
    Stacked area chart showing how portfolio weights evolve over time.
    """
    # Pick the top-N stocks by average weight
    avg_weight = weights_history.mean().sort_values(ascending=False)
    top = avg_weight.head(top_n).index.tolist()

    # Group the rest as "Other"
    plot_data = weights_history[top].copy()
    plot_data["Other"] = weights_history.drop(columns=top, errors="ignore").sum(axis=1)

    # Clean ticker names
    plot_data.columns = [
        c.replace(" CN Equity", "").replace(" US Equity", " (SPY)")
        if c != "Other" else c
        for c in plot_data.columns
    ]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    plot_data.plot.area(ax=ax, stacked=True, alpha=0.75, linewidth=0.5)

    ax.set_title("Portfolio Weight Evolution – Canadian Clone",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)

    _save(fig, "weight_evolution")


# ── Phase 3 Plots ─────────────────────────────────────────


def plot_supervisor_decisions(confidence, clone_returns, regime):
    """
    Supervisor confidence over time with bull/bear regime shading.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1]})

    # Panel 1: Confidence score with regime shading
    ax = axes[0]
    ax.plot(confidence.index, confidence.values, color="#2c3e50",
            linewidth=0.8, alpha=0.8)
    ax.axhline(0.7, color="#27ae60", linestyle="--", alpha=0.5, label="Aggressive threshold")
    ax.axhline(0.3, color="#e74c3c", linestyle="--", alpha=0.5, label="Defensive threshold")

    # Shade regimes
    if regime is not None:
        common = confidence.index.intersection(regime.index)
        r = regime.loc[common]
        agg_mask = r == "aggressive"
        def_mask = r == "defensive"

        for idx in common[agg_mask]:
            ax.axvspan(idx, idx + pd.Timedelta(days=1), alpha=0.05, color="green")
        for idx in common[def_mask]:
            ax.axvspan(idx, idx + pd.Timedelta(days=1), alpha=0.05, color="red")

    ax.set_title("AI Supervisor — Confidence Score & Regime Detection",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Confidence P")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Clone returns colored by regime
    ax = axes[1]
    common = clone_returns.index.intersection(confidence.index)
    clone_plot = clone_returns.loc[common]
    cum = (1 + clone_plot).cumprod() * 100

    ax.plot(cum.index, cum.values, color="#2980b9", linewidth=1)
    ax.set_title("Clone Cumulative Returns (test period)", fontsize=12)
    ax.set_ylabel("Indexed (100)")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)

    plt.tight_layout()
    _save(fig, "supervisor_decisions")


def plot_supervised_vs_unsupervised(supervised_returns, clone_returns, spy_returns):
    """
    Triple overlay: Supervised Clone vs Unsupervised Clone vs SPY.
    """
    # Align all to the supervised period
    start = supervised_returns.index[0]
    end = supervised_returns.index[-1]

    sup = supervised_returns
    clone = clone_returns.loc[start:end]
    spy = spy_returns.loc[start:end]

    cum_sup = (1 + sup).cumprod() * 100
    cum_clone = (1 + clone).cumprod() * 100
    cum_spy = (1 + spy).cumprod() * 100

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    ax.plot(cum_spy.index, cum_spy.values, color="#95a5a6", linewidth=2,
            label="SPY", alpha=0.8, linestyle="--")
    ax.plot(cum_clone.index, cum_clone.values, color="#e74c3c", linewidth=2,
            label="Clone (unsupervised)", alpha=0.7)
    ax.plot(cum_sup.index, cum_sup.values, color="#2980b9", linewidth=2.5,
            label="Clone + AI Supervisor", alpha=0.9)

    ax.set_title("AI Supervisor Impact — Test Period",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Indexed (100 = start)")
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)

    _save(fig, "supervised_vs_unsupervised")


def _max_drawdown(returns):
    """Compute maximum drawdown from returns."""
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min()
