"""
Synthetic Validation — Chan (2018) Framework.

Generates synthetic market histories via stationary block bootstrap
and runs the full Worker + Supervisor pipeline on each to validate
that strategy performance is not an artifact of overfitting to a
single historical path.

Reference: Chan, E. (2018), "Optimizing Trading Strategies without Overfitting"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.qp_solver import rolling_optimization
from src.supervisor import (
    generate_meta_labels,
    build_super_state,
    train_classifier,
    apply_supervisor,
)
from src.backtester import compute_metrics
from src.config import BENCHMARK, RESULTS_DIR


# ──────────────────────────────────────────────
# 1. Stationary Block Bootstrap
# ──────────────────────────────────────────────

def stationary_block_bootstrap(returns, n_paths=1000, avg_block_len=21,
                                seed=42):
    """
    Generate synthetic return histories via stationary block bootstrap.

    Randomly samples blocks of consecutive returns (geometric block lengths)
    and concatenates them. Preserves cross-asset correlations, volatility
    clustering, and fat tails within each block.

    Parameters
    ----------
    returns : pd.DataFrame
        Historical daily returns (assets as columns, dates as rows).
    n_paths : int
        Number of synthetic histories to generate.
    avg_block_len : int
        Average block length (geometrically distributed).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    synthetic_paths : list[pd.DataFrame]
        Each element is a synthetic return history with same shape as input.
    """
    rng = np.random.RandomState(seed)
    T, N = returns.shape
    prob = 1.0 / avg_block_len  # Geometric distribution parameter

    synthetic_paths = []

    for path_idx in range(n_paths):
        # Build synthetic history by sampling blocks
        synthetic_rows = []
        total_rows = 0

        while total_rows < T:
            # Random start point
            start = rng.randint(0, T)
            # Random block length (geometric distribution)
            block_len = rng.geometric(prob)
            block_len = min(block_len, T - total_rows)  # Don't exceed target

            # Extract block (wrap around if needed)
            indices = [(start + j) % T for j in range(block_len)]
            block = returns.iloc[indices].values
            synthetic_rows.append(block)
            total_rows += block_len

        # Stack and trim to exact length
        synthetic_array = np.vstack(synthetic_rows)[:T]
        synthetic_df = pd.DataFrame(
            synthetic_array,
            columns=returns.columns,
            index=returns.index,
        )
        synthetic_paths.append(synthetic_df)

    return synthetic_paths


# ──────────────────────────────────────────────
# 2. Run Pipeline on a Single Synthetic Path
# ──────────────────────────────────────────────

def _run_pipeline_on_path(synthetic_returns, econ, yield_curve,
                           benchmark_col, train_end="2019-12-31"):
    """
    Run the full Worker + Supervisor pipeline on one synthetic history.

    Returns metrics dict or None if pipeline fails.
    """
    try:
        # Reconstruct prices from returns
        synthetic_prices = (1 + synthetic_returns).cumprod()
        # Scale to start at 100 for numerical stability
        synthetic_prices = synthetic_prices * 100

        # Worker: Rolling QP
        weights_history, worker_returns = rolling_optimization(
            synthetic_prices,
            benchmark_col=benchmark_col,
            lookback=252 * 5,
            rebal_freq="M",
        )

        if len(worker_returns) < 252:  # Need at least 1 year
            return None

        # Supervisor: Meta-labeling
        labels = generate_meta_labels(worker_returns, threshold=0.02, horizon=5)
        X = build_super_state(worker_returns, synthetic_returns, econ, yield_curve)

        common = X.index.intersection(labels.index)
        X = X.loc[common]
        y = labels.loc[common]

        train_mask = X.index <= train_end
        test_mask = X.index > train_end

        X_train, y_train = X.loc[train_mask], y.loc[train_mask]
        X_test = X.loc[test_mask]

        if len(X_train) < 100 or len(X_test) < 50:
            return None

        model, _ = train_classifier(X_train, y_train, n_splits=3)

        confidence_test = pd.Series(
            model.predict_proba(X_test)[:, 1],
            index=X_test.index,
        )

        test_worker = worker_returns.loc[confidence_test.index]
        supervised_returns, _, _ = apply_supervisor(test_worker, confidence_test)

        # Benchmark returns
        spy_returns = synthetic_returns[benchmark_col].loc[confidence_test.index]

        # Compute metrics for each strategy
        cpo_metrics = compute_metrics(supervised_returns)
        worker_metrics = compute_metrics(test_worker)
        benchmark_metrics = compute_metrics(spy_returns)

        return {
            "cpo_sharpe": cpo_metrics.get("Sharpe", np.nan),
            "cpo_maxdd": cpo_metrics.get("Max Drawdown", np.nan),
            "worker_sharpe": worker_metrics.get("Sharpe", np.nan),
            "worker_maxdd": worker_metrics.get("Max Drawdown", np.nan),
            "benchmark_sharpe": benchmark_metrics.get("Sharpe", np.nan),
            "benchmark_maxdd": benchmark_metrics.get("Max Drawdown", np.nan),
        }

    except Exception as e:
        return None


# ──────────────────────────────────────────────
# 3. Full Synthetic Validation
# ──────────────────────────────────────────────

def run_synthetic_validation(returns, econ, yield_curve,
                              n_paths=100, avg_block_len=21,
                              benchmark_col="SPY US Equity",
                              train_end="2019-12-31",
                              seed=42, verbose=True):
    """
    Run the full synthetic validation framework.

    Parameters
    ----------
    returns : pd.DataFrame
        Historical daily returns.
    econ : pd.DataFrame
        Economic indicators.
    yield_curve : pd.DataFrame
        Yield curve data.
    n_paths : int
        Number of synthetic histories (default 100 for speed, use 1000 for paper).
    avg_block_len : int
        Average bootstrap block length in days.
    benchmark_col : str
        Column name for benchmark.
    train_end : str
        Training period cutoff.
    seed : int
        Random seed.
    verbose : bool
        Print progress.

    Returns
    -------
    results_df : pd.DataFrame
        Sharpe ratios and max drawdowns for each simulation.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("SYNTHETIC VALIDATION (Chan 2018)")
        print("=" * 60)
        print(f"Generating {n_paths} synthetic market histories...")

    # Generate synthetic paths
    synthetic_paths = stationary_block_bootstrap(
        returns, n_paths=n_paths, avg_block_len=avg_block_len, seed=seed
    )

    if verbose:
        print(f"Running full pipeline on each path...")

    results = []
    successes = 0

    for i, syn_returns in enumerate(synthetic_paths):
        # Suppress print output during batch runs
        import io, sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        result = _run_pipeline_on_path(
            syn_returns, econ, yield_curve,
            benchmark_col=benchmark_col,
            train_end=train_end,
        )

        sys.stdout = old_stdout

        if result is not None:
            results.append(result)
            successes += 1

        if verbose and (i + 1) % 10 == 0:
            print(f"  Completed {i+1}/{n_paths} "
                  f"({successes} successful)")

    results_df = pd.DataFrame(results)

    if verbose and len(results_df) > 0:
        print(f"\n[synthetic] Completed: {len(results_df)}/{n_paths} successful")
        print(f"\n[synthetic] Sharpe Ratio Summary:")
        print(f"  CPO Model:    {results_df['cpo_sharpe'].mean():.3f} "
              f"± {results_df['cpo_sharpe'].std():.3f}")
        print(f"  Worker Only:  {results_df['worker_sharpe'].mean():.3f} "
              f"± {results_df['worker_sharpe'].std():.3f}")
        print(f"  Benchmark:    {results_df['benchmark_sharpe'].mean():.3f} "
              f"± {results_df['benchmark_sharpe'].std():.3f}")

    return results_df


# ──────────────────────────────────────────────
# 4. Visualization
# ──────────────────────────────────────────────

def plot_synthetic_validation(results_df, save_dir=None):
    """
    Plot histogram comparing Sharpe ratio distributions.
    """
    if save_dir is None:
        save_dir = RESULTS_DIR
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Sharpe Ratio Histogram ──
    ax = axes[0]
    bins = np.linspace(
        min(results_df[["cpo_sharpe", "worker_sharpe", "benchmark_sharpe"]].min()),
        max(results_df[["cpo_sharpe", "worker_sharpe", "benchmark_sharpe"]].max()),
        30
    )

    ax.hist(results_df["benchmark_sharpe"], bins=bins, alpha=0.5,
            label=f"Benchmark (μ={results_df['benchmark_sharpe'].mean():.2f})",
            color="#e74c3c", edgecolor="white")
    ax.hist(results_df["worker_sharpe"], bins=bins, alpha=0.5,
            label=f"Worker Only (μ={results_df['worker_sharpe'].mean():.2f})",
            color="#3498db", edgecolor="white")
    ax.hist(results_df["cpo_sharpe"], bins=bins, alpha=0.7,
            label=f"CPO Model (μ={results_df['cpo_sharpe'].mean():.2f})",
            color="#2ecc71", edgecolor="white")

    ax.set_xlabel("Sharpe Ratio", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Sharpe Ratio Distribution\n(Synthetic Validation)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ── Max Drawdown Histogram ──
    ax = axes[1]
    for col, label, color in [
        ("benchmark_maxdd", "Benchmark", "#e74c3c"),
        ("worker_maxdd", "Worker Only", "#3498db"),
        ("cpo_maxdd", "CPO Model", "#2ecc71"),
    ]:
        vals = results_df[col].dropna() * 100  # Convert to percentage
        ax.hist(vals, bins=25, alpha=0.5, label=label, color=color,
                edgecolor="white")

    ax.set_xlabel("Maximum Drawdown (%)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Max Drawdown Distribution\n(Synthetic Validation)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = save_dir / "synthetic_validation.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[synthetic] Saved → {path}")

    # Save raw results
    csv_path = save_dir / "synthetic_validation_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"[synthetic] Saved → {csv_path}")

    return path
