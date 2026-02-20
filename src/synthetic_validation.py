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
                           benchmark_col, train_frac=0.6,
                           min_train=15, min_test=3, verbose=False):
    """
    Run the full Worker + Supervisor pipeline on one synthetic history.

    Uses relative time split (train_frac) instead of fixed date, since
    synthetic paths share the same calendar but econ/yc may have limited range.

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
            rebal_freq="ME",
        )

        if len(worker_returns) < 252:  # Need at least 1 year
            return None

        # Supervisor: Meta-labeling (verbose=False for batch)
        labels = generate_meta_labels(worker_returns, threshold=0.02, horizon=5, verbose=verbose)
        X = build_super_state(worker_returns, synthetic_returns, econ, yield_curve, verbose=verbose)

        common = X.index.intersection(labels.index)
        X = X.loc[common]
        y = labels.loc[common]

        # Relative split: first train_frac for train, rest for test
        n = len(X)
        n_train = int(n * train_frac)
        n_test = n - n_train

        X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
        X_test = X.iloc[n_train:]

        if len(X_train) < min_train or len(X_test) < min_test:
            return None

        model, _ = train_classifier(X_train, y_train, n_splits=3, verbose=verbose)

        proba = model.predict_proba(X_test)
        # Handle single-class DummyClassifier: proba has shape (n, 1) not (n, 2)
        conf_col = min(1, proba.shape[1] - 1)
        confidence_test = pd.Series(
            proba[:, conf_col],
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


def _run_pipeline_on_path_debug(synthetic_returns, econ, yield_curve,
                                 benchmark_col, train_frac=0.6,
                                 min_train=15, min_test=3):
    """Debug version: raises on first failure to surface the actual error."""
    # Reconstruct prices from returns
    synthetic_prices = (1 + synthetic_returns).cumprod()
    synthetic_prices = synthetic_prices * 100

    weights_history, worker_returns = rolling_optimization(
        synthetic_prices,
        benchmark_col=benchmark_col,
        lookback=252 * 5,
        rebal_freq="ME",
    )

    if len(worker_returns) < 252:
        raise ValueError(f"worker_returns too short: {len(worker_returns)} < 252")

    labels = generate_meta_labels(worker_returns, threshold=0.02, horizon=5)
    X = build_super_state(worker_returns, synthetic_returns, econ, yield_curve)

    common = X.index.intersection(labels.index)
    X = X.loc[common]
    y = labels.loc[common]

    n = len(X)
    n_train = int(n * train_frac)
    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_test = X.iloc[n_train:]

    if len(X_train) < min_train or len(X_test) < min_test:
        raise ValueError(
            f"Insufficient train/test: X_train={len(X_train)}, X_test={len(X_test)} "
            f"(need >={min_train} train, >={min_test} test). "
            f"Total common samples: {n}. Check econ/yield_curve date range vs returns."
        )

    model, _ = train_classifier(X_train, y_train, n_splits=3)
    proba = model.predict_proba(X_test)
    conf_col = min(1, proba.shape[1] - 1)
    confidence_test = pd.Series(
        proba[:, conf_col],
        index=X_test.index,
    )
    test_worker = worker_returns.loc[confidence_test.index]
    supervised_returns, _, _ = apply_supervisor(test_worker, confidence_test)
    spy_returns = synthetic_returns[benchmark_col].loc[confidence_test.index]

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


# ──────────────────────────────────────────────
# 3. Full Synthetic Validation
# ──────────────────────────────────────────────

def run_synthetic_validation(returns, econ, yield_curve,
                              n_paths=100, avg_block_len=21,
                              benchmark_col="SPY US Equity",
                              train_frac=0.6, min_train=15, min_test=3,
                              seed=42, verbose=True, min_successful=None):
    """
    Run the full synthetic validation framework.

    Uses relative time split (train_frac) since econ/yc may have limited
    date range vs returns; fixed-date split can yield 0 test samples.

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
    train_frac : float
        Fraction of common samples for training (default 0.8).
    min_train, min_test : int
        Minimum samples required for train and test.
    seed : int
        Random seed.
    verbose : bool
        Print progress.
    min_successful : int, optional
        If set, keep generating paths until at least this many succeed.

    Returns
    -------
    results_df : pd.DataFrame
        Sharpe ratios and max drawdowns for each simulation.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("SYNTHETIC VALIDATION (Chan 2018)")
        print("=" * 60)

    results = []
    total_run = 0
    current_seed = seed

    while True:
        batch_size = n_paths if min_successful is None else max(n_paths, min_successful * 2)
        if verbose:
            print(f"Generating {batch_size} synthetic market histories (seed={current_seed})...")

        synthetic_paths = stationary_block_bootstrap(
            returns, n_paths=batch_size, avg_block_len=avg_block_len, seed=current_seed
        )

        if verbose:
            print(f"Running full pipeline on each path...")

        for i, syn_returns in enumerate(synthetic_paths):
            # Suppress print output during batch runs
            import io, sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

            result = _run_pipeline_on_path(
                syn_returns, econ, yield_curve,
                benchmark_col=benchmark_col,
                train_frac=train_frac,
                min_train=min_train,
                min_test=min_test,
                verbose=False,
            )

            sys.stdout = old_stdout

            if result is not None:
                results.append(result)

            total_run += 1
            if verbose and total_run % 10 == 0:
                print(f"  Completed {total_run} paths ({len(results)} successful)")

        if min_successful is None or len(results) >= min_successful:
            break
        current_seed += 1
        if verbose:
            print(f"  Need {min_successful - len(results)} more. Trying new seed...")

    results_df = pd.DataFrame(results)
    if min_successful is not None and len(results_df) > min_successful:
        results_df = results_df.head(min_successful)

    # If all paths failed, run first path in debug mode to surface the actual error
    if len(results_df) == 0 and verbose and len(synthetic_paths) > 0:
        print("\n[synthetic] All paths failed. Running first path in debug mode...")
        import io, sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            _run_pipeline_on_path_debug(
                synthetic_paths[0], econ, yield_curve,
                benchmark_col=benchmark_col,
                train_frac=train_frac,
                min_train=min_train,
                min_test=min_test,
            )
        except Exception as e:
            sys.stdout = old_stdout
            print(f"\n[synthetic] First path failed with: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        else:
            sys.stdout = old_stdout

    if verbose and len(results_df) > 0:
        print(f"\n[synthetic] Completed: {len(results_df)} successful (ran {total_run} paths)")
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
