"""
Portfolio Optimizer – Pipeline Runner

Phase 1: Load → Clean → Feature Engineer → EDA
Phase 2: QP Solver → Backtest → Evaluate
Phase 3: AI Supervisor → Meta-Labeling → Regime-Aware Allocation

Run with:  python main.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_all
from src.features import compute_returns, merge_macro, filter_sparse_tickers
from src.eda import (
    plot_correlation_heatmap,
    plot_cumulative_returns,
    plot_rolling_volatility,
    plot_regime_indicators,
    print_summary_stats,
    plot_clone_vs_spy,
    plot_tracking_error,
    plot_weight_evolution,
    plot_supervisor_decisions,
    plot_supervised_vs_unsupervised,
)
from src.qp_solver import rolling_optimization
from src.backtester import (
    compute_metrics, compare_benchmarks,
    compute_turnover, compute_tracking_error,
)
from src.supervisor import run_supervisor_pipeline
from src.config import BENCHMARK, RESULTS_DIR


def phase1(prices, all_fields, profiles, econ, yield_curve):
    """Phase 1: Data prep & EDA."""
    print("\n" + "=" * 60)
    print("PHASE 1: Data Preparation & EDA")
    print("=" * 60)

    prices_clean, removed = filter_sparse_tickers(prices, min_coverage=0.5)
    returns = compute_returns(prices_clean, method="simple")
    merged = merge_macro(returns, econ, yield_curve)

    print("\nGenerating Phase 1 plots...")
    plot_correlation_heatmap(returns)
    plot_cumulative_returns(prices_clean)
    plot_rolling_volatility(returns)
    plot_regime_indicators(econ, yield_curve)
    stats = print_summary_stats(returns)

    return prices_clean, returns


def phase2(prices_clean):
    """Phase 2: QP Solver — Canadian Clone."""
    print("\n" + "=" * 60)
    print("PHASE 2: QP Solver — Canadian Clone")
    print("=" * 60)

    # Run rolling optimization
    weights_history, clone_returns = rolling_optimization(
        prices_clean,
        benchmark_col=BENCHMARK,
        lookback=252 * 5,
        rebal_freq="M",
    )

    # Get SPY returns for comparison
    spy_returns = prices_clean[BENCHMARK].pct_change().iloc[1:]

    # Compute turnover
    avg_turnover = compute_turnover(weights_history)
    print(f"\n[phase2] Average monthly turnover: {avg_turnover:.2%}")

    # Compare strategies
    te = compute_tracking_error(clone_returns, spy_returns)
    print(f"[phase2] Annualized tracking error: {te:.2%}")

    # Equal-weight benchmark
    canadian_prices = prices_clean.drop(columns=[BENCHMARK], errors="ignore")
    ew_returns = canadian_prices.pct_change().iloc[1:].mean(axis=1)

    results = {
        "Canadian Clone (QP)": clone_returns,
        "SPY Buy & Hold": spy_returns,
        "Equal-Weight TSX": ew_returns,
    }
    comparison = compare_benchmarks(results, spy_returns)

    comparison.to_csv(RESULTS_DIR / "strategy_comparison.csv")
    print(f"\n[phase2] Saved → {RESULTS_DIR / 'strategy_comparison.csv'}")

    # Phase 2 plots
    print("\nGenerating Phase 2 plots...")
    plot_clone_vs_spy(clone_returns, spy_returns)
    plot_tracking_error(clone_returns, spy_returns)
    plot_weight_evolution(weights_history)

    return weights_history, clone_returns


def phase3(clone_returns, prices_clean, econ, yield_curve):
    """Phase 3: AI Supervisor — Meta-Labeling."""

    # Get all returns for uncertainty computation
    returns_all = prices_clean.pct_change().iloc[1:]

    # Run supervisor pipeline (train up to 2019, test 2020+)
    supervised_returns, regime, model, confidence, importances, allocation = (
        run_supervisor_pipeline(
            clone_returns, returns_all, econ, yield_curve,
            train_end="2019-12-31",
        )
    )

    # Get SPY returns for comparison
    spy_returns = prices_clean[BENCHMARK].pct_change().iloc[1:]

    # Compare all strategies on the test period
    test_start = supervised_returns.index[0]
    test_clone = clone_returns.loc[test_start:]
    test_spy = spy_returns.loc[test_start:]

    results = {
        "Clone + AI Supervisor": supervised_returns,
        "Clone (unsupervised)": test_clone,
        "SPY Buy & Hold": test_spy,
    }
    comparison = compare_benchmarks(results, test_spy)

    comparison.to_csv(RESULTS_DIR / "phase3_comparison.csv")
    print(f"\n[phase3] Saved → {RESULTS_DIR / 'phase3_comparison.csv'}")

    # Phase 3 plots
    print("\nGenerating Phase 3 plots...")
    plot_supervisor_decisions(confidence, clone_returns, regime)
    plot_supervised_vs_unsupervised(supervised_returns, clone_returns, spy_returns)

    # Save feature importances
    importances.to_csv(RESULTS_DIR / "feature_importances.csv")
    print(f"[phase3] Saved → {RESULTS_DIR / 'feature_importances.csv'}")

    return supervised_returns, regime, model


def main():
    print("🚀 Portfolio Optimizer – Full Pipeline")
    print("=" * 60)

    # Load all data
    prices, all_fields, profiles, econ, yield_curve = load_all()

    # Phase 1
    prices_clean, returns = phase1(prices, all_fields, profiles, econ, yield_curve)

    # Phase 2
    weights_history, clone_returns = phase2(prices_clean)

    # Phase 3
    supervised_returns, regime, model = phase3(
        clone_returns, prices_clean, econ, yield_curve
    )

    print("\n✅ All phases complete! Check results/ for outputs.")


if __name__ == "__main__":
    main()
