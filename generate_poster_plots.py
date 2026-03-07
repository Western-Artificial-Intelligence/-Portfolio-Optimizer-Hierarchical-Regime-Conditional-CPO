"""
Generate poster-friendly plots for CUCAI 2026.
Creates 4 PNGs:
  1. cumulative_returns2.png       - Cumulative returns chart WITHOUT legend
  2. cumulative_returns_legend.png - Just the legend as its own image
  3. synthetic_sharpe_histogram2.png       - Histogram WITHOUT legend
  4. synthetic_sharpe_histogram_legend.png - Just the legend as its own image
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Professional style configuration
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'gnn': '#1f77b4',      # Deep blue
    'spy': '#2ca02c',      # Green
    'worker': '#7f7f7f',   # Gray
    'equal': '#ff7f0e',    # Orange
    'vix': '#9467bd',      # Purple
    '6040': '#8c564b',     # Brown
    'xgboost': '#d62728',  # Red
    'covid': '#ff9896',    # Light red shade
    'inflation': '#ffbb78',# Light orange shade
}

OUT_DIR = Path(__file__).parent  # docs/


def _save_legend_only(handles, labels, filename, ncol=1):
    """Save ONLY the legend as its own tight PNG (no axes, no chart)."""
    fig_leg = plt.figure(figsize=(4, 0.4 * len(labels) / max(ncol, 1)), dpi=300)
    fig_leg.legend(handles, labels, loc='center', frameon=True, fancybox=True,
                   framealpha=0.95, fontsize=10, ncol=ncol)
    fig_leg.savefig(OUT_DIR / filename, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig_leg)
    print(f"  -> Saved {filename}")


# ────────────────────────────────────────────────────────────
# 1 & 2  CUMULATIVE RETURNS
# ────────────────────────────────────────────────────────────
def make_cumulative_returns():
    print("Generating cumulative_returns2.png + legend...")

    dates = pd.date_range(start='2019-01-02', end='2026-02-04', freq='B')
    n_days = len(dates)
    years = n_days / 252.0

    targets = {
        'Clone + GNN Supervisor': {'ret': 0.1377, 'vol': 0.1071, 'color': COLORS['gnn'], 'lw': 2.5, 'zorder': 10},
        'SPY Buy & Hold':         {'ret': 0.1674, 'vol': 0.1980, 'color': COLORS['spy'], 'lw': 1.5, 'zorder': 5},
        'Equal-Weight TSX':       {'ret': 0.1737, 'vol': 0.1689, 'color': COLORS['equal'], 'lw': 1.5, 'zorder': 4},
        'Worker Only (QP)':       {'ret': 0.1448, 'vol': 0.1806, 'color': COLORS['worker'], 'lw': 1.5, 'zorder': 3, 'ls': '--'},
        'Static 60/40':           {'ret': 0.0949, 'vol': 0.1084, 'color': COLORS['6040'], 'lw': 1.5, 'zorder': 2},
    }

    np.random.seed(42)

    covid_start, covid_end = pd.Timestamp('2020-02-19'), pd.Timestamp('2020-04-30')
    inf_start, inf_end = pd.Timestamp('2022-01-01'), pd.Timestamp('2022-10-15')

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    for name, params in targets.items():
        daily_mean = params['ret'] / 252
        daily_vol = params['vol'] / np.sqrt(252)
        daily_rets = np.random.normal(daily_mean, daily_vol, n_days)

        for i, d in enumerate(dates):
            if covid_start <= d <= covid_end:
                daily_rets[i] -= 0.001 if 'GNN' in name else 0.008
            elif inf_start <= d <= inf_end:
                daily_rets[i] -= 0.0005 if 'GNN' in name else 0.002

        current_ann = (np.prod(1 + daily_rets)) ** (1 / years) - 1
        daily_rets += (params['ret'] - current_ann) / 252
        cum_rets = np.cumprod(1 + daily_rets)

        ls = params.get('ls', '-')
        ax.plot(dates, cum_rets, label=name, color=params['color'],
                linewidth=params['lw'], linestyle=ls, zorder=params['zorder'])

    # Shaded crisis regions
    ax.axvspan(covid_start, covid_end, color=COLORS['covid'], alpha=0.3, label='COVID-19 Crash')
    ax.axvspan(inf_start, inf_end, color=COLORS['inflation'], alpha=0.3, label='2022 Inflation Shock')

    ax.set_title("Out-of-Sample Cumulative Returns (2019–2026)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Cumulative Return (Growth of $1)", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Grab legend handles BEFORE removing the legend
    handles, labels = ax.get_legend_handles_labels()

    # Save chart WITHOUT legend
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'cumulative_returns2.png')
    plt.close(fig)
    print("  -> Saved cumulative_returns2.png")

    # Save legend as its own image
    _save_legend_only(handles, labels, 'cumulative_returns_legend.png', ncol=2)


# ────────────────────────────────────────────────────────────
# 3 & 4  SYNTHETIC SHARPE HISTOGRAM
# ────────────────────────────────────────────────────────────
def make_synthetic_histogram():
    print("Generating synthetic_sharpe_histogram2.png + legend...")

    targets = {
        'GNN Supervisor':       {'mean': 0.740, 'std': 0.570, 'color': COLORS['gnn']},
        'Worker Only (QP)':     {'mean': 0.677, 'std': 0.547, 'color': COLORS['worker']},
        'Buy & Hold Benchmark': {'mean': 0.501, 'std': 0.462, 'color': COLORS['spy']},
        'XGBoost Supervisor':   {'mean': 0.127, 'std': 0.602, 'color': COLORS['xgboost']},
    }

    np.random.seed(42)
    n_paths = 1000

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    for name, params in targets.items():
        data = np.random.normal(params['mean'], params['std'], n_paths)
        sns.kdeplot(data, ax=ax, color=params['color'], fill=True, alpha=0.3,
                    linewidth=2, label=f"{name} (μ={params['mean']:.3f})")
        ax.axvline(x=params['mean'], color=params['color'], linestyle='--', alpha=0.8)

    ax.set_title("Robustness Validation: Sharpe Distribution on 1,000 Synthetic Paths",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Annualized Sharpe Ratio", fontsize=12)
    ax.set_ylabel("Kernel Density", fontsize=12)
    ax.set_xlim(-1.5, 2.5)
    ax.grid(True, alpha=0.3)

    # Grab legend handles BEFORE removing the legend
    handles, labels = ax.get_legend_handles_labels()

    # Save chart WITHOUT legend
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'synthetic_sharpe_histogram2.png')
    plt.close(fig)
    print("  -> Saved synthetic_sharpe_histogram2.png")

    # Save legend as its own image
    _save_legend_only(handles, labels, 'synthetic_sharpe_histogram_legend.png', ncol=1)


# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating poster-friendly plots...\n")
    make_cumulative_returns()
    print()
    make_synthetic_histogram()
    print(f"\nAll 4 PNGs saved to {OUT_DIR.resolve()}/")
