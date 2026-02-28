"""
Bootstrap Confidence Intervals for Sharpe Ratio.

Computes 95% CI on annualized Sharpe ratio for each strategy
using stationary block bootstrap (10,000 resamples, block=21 days).

Usage:
    python bootstrap_ci.py

Output:
    Prints table ready to paste into paper2.tex Table IV.
    Also saves results6/bootstrap_ci.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Load results ─────────────────────────────────────────────────────────────
RESULTS_DIR = Path("results6")
RESULTS2_DIR = Path("results2")

# Load Phase 3 GNN comparison CSV for strategy returns
# We reconstruct daily returns from the comparison CSV metrics OR
# load directly from the pipeline's saved data.
# Since we have phase3_gnn_comparison.csv (metrics only, not daily returns),
# we need to recompute. Import pipeline components instead.

import sys
sys.path.insert(0, ".")

from src.config import GNN_RESULTS_DIR, RESULTS_DIR as R2
from src.data_loader import load_all
from src.features import filter_sparse_tickers
from src.qp_solver import rolling_optimization
from src.gnn_supervisor import run_gnn_supervisor_pipeline
from src.config import BENCHMARK

def block_bootstrap_sharpe(returns: pd.Series,
                            n_boot: int = 10_000,
                            block_size: int = 21,
                            annualize: int = 252) -> tuple[float, float]:
    """
    Stationary block bootstrap CI for annualized Sharpe ratio.

    Returns (lower_95, upper_95).
    """
    r = returns.dropna().values
    n = len(r)
    sharpes = []

    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        # Draw random starting indices for blocks
        n_blocks = int(np.ceil(n / block_size))
        starts   = rng.integers(0, n - block_size + 1, size=n_blocks)
        sample   = np.concatenate([r[s:s + block_size] for s in starts])[:n]
        mu  = sample.mean()
        std = sample.std(ddof=1)
        if std > 1e-10:
            sharpes.append(mu / std * np.sqrt(annualize))

    sharpes = np.array(sharpes)
    lo = np.percentile(sharpes, 2.5)
    hi = np.percentile(sharpes, 97.5)
    return lo, hi


def main():
    print("Loading data...")
    prices, all_fields, profiles, econ, yield_curve = load_all()
    prices_clean, _ = filter_sparse_tickers(prices, min_coverage=0.5)

    print("Running QP solver...")
    _, clone_returns = rolling_optimization(prices_clean, benchmark_col=BENCHMARK, lookback=252*5, rebal_freq="ME")

    spy_returns = prices_clean[BENCHMARK].pct_change(fill_method=None).iloc[1:]

    print("Running GNN supervisor...")
    supervised_returns, alpha, _ = run_gnn_supervisor_pipeline(
        clone_returns, prices_clean, all_fields, profiles, econ, yield_curve,
        fold=4, verbose=False
    )

    # Align all to GNN test period
    test_start = supervised_returns.index[0]
    clone_test = clone_returns.loc[test_start:]
    spy_test   = spy_returns.loc[test_start:]
    gnn_test   = supervised_returns.loc[test_start:]

    strategies = {
        "Clone + GNN Supervisor": gnn_test,
        "Clone (Worker only)":    clone_test,
        "SPY Buy & Hold":         spy_test,
    }

    print("\n" + "=" * 65)
    print("Bootstrap 95% CI — Annualized Sharpe Ratio")
    print("(10,000 resamples, block=21 days, seed=42)")
    print("=" * 65)
    print(f"{'Strategy':<30} {'Sharpe':>8} {'95% CI':>20}")
    print("-" * 65)

    rows = []
    for name, rets in strategies.items():
        point = rets.mean() / (rets.std(ddof=1) + 1e-10) * np.sqrt(252)
        lo, hi = block_bootstrap_sharpe(rets)
        print(f"{name:<30} {point:>8.3f}   [{lo:+.3f}, {hi:+.3f}]")
        rows.append({"Strategy": name, "Sharpe": round(point, 3),
                     "CI_lower": round(lo, 3), "CI_upper": round(hi, 3)})

    print("=" * 65)

    df = pd.DataFrame(rows)
    out_path = GNN_RESULTS_DIR / "bootstrap_ci.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    print("\n── LaTeX snippet for Table IV in paper2.tex ──")
    for _, row in df.iterrows():
        lo, hi = row["CI_lower"], row["CI_upper"]
        print(f"{row['Strategy']} & {row['Sharpe']:.3f} & "
              f"[{lo:+.3f}, {hi:+.3f}] \\\\")


if __name__ == "__main__":
    main()
