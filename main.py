"""
Portfolio Optimizer - Pipeline Runner

Phase 1: Load -> Clean -> Feature Engineer -> EDA
Phase 2: QP Solver -> Backtest -> Evaluate
Phase 3: AI Supervisor -> Meta-Labeling -> Regime-Aware Allocation
Phase 4: SHAP Analysis -> Feature Importance
Phase 5: Ablation Study -> Feature Group Validation
Phase 6: Synthetic Validation -> Robustness Testing (Optional, slow)

Run with:  python main.py
"""

import sys
import os
import io

# Fix Unicode output on Windows (cp1252 can't encode arrows, emojis, etc.)
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

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
    plot_clone_spy_equalweight,
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
from src.supervisor import (
    run_supervisor_pipeline,
    generate_meta_labels,
    build_super_state,
)
from src.shap_analysis import run_shap_analysis
from src.ablation import run_ablation_study
from src.benchmarks import run_all_benchmarks
from src.synthetic_validation import (
    run_synthetic_validation,
    plot_synthetic_validation,
)
from src.config import BENCHMARK, RESULTS_DIR, GNN_RESULTS_DIR

# ─────────────────────────────────────────────────────────────────────────────
# GNN Supervisor toggle
# Set USE_GNN_SUPERVISOR = True  to use the CRISP-inspired GNN supervisor.
# Set USE_GNN_SUPERVISOR = False to fall back to the original XGBoost supervisor.
# The GNN must be trained first (phase5b). On first run, leave = False.
# ─────────────────────────────────────────────────────────────────────────────
USE_GNN_SUPERVISOR = False   # ← flip to True after running phase5b once


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
        rebal_freq="ME",
    )

    # Get SPY returns for comparison
    spy_returns = prices_clean[BENCHMARK].pct_change(fill_method=None).iloc[1:]

    # Compute turnover
    avg_turnover = compute_turnover(weights_history)
    print(f"\n[phase2] Average monthly turnover: {avg_turnover:.2%}")

    # Compare strategies
    te = compute_tracking_error(clone_returns, spy_returns)
    print(f"[phase2] Annualized tracking error: {te:.2%}")

    # Equal-weight benchmark
    canadian_prices = prices_clean.drop(columns=[BENCHMARK], errors="ignore")
    ew_returns = canadian_prices.pct_change(fill_method=None).iloc[1:].mean(axis=1)

    results = {
        "Canadian Clone (QP)": clone_returns,
        "SPY Buy & Hold": spy_returns,
        "Equal-Weight TSX": ew_returns,
    }
    comparison = compare_benchmarks(results, spy_returns)

    comparison.to_csv(RESULTS_DIR / "strategy_comparison.csv")
    print(f"\n[phase2] Saved: {RESULTS_DIR / 'strategy_comparison.csv'}")

    # Phase 2 plots
    print("\nGenerating Phase 2 plots...")
    plot_clone_vs_spy(clone_returns, spy_returns)
    plot_clone_spy_equalweight(clone_returns, spy_returns, ew_returns)
    plot_tracking_error(clone_returns, spy_returns)
    plot_weight_evolution(weights_history)

    return weights_history, clone_returns


def phase3(clone_returns, prices_clean, all_fields, profiles, econ, yield_curve):
    """
    Phase 3: AI Supervisor.

    Branches between two implementations:
      USE_GNN_SUPERVISOR = False → XGBoost meta-labeling (original)
      USE_GNN_SUPERVISOR = True  → Dynamic GNN supervisor (CRISP-inspired)
    """
    if USE_GNN_SUPERVISOR:
        return _phase3_gnn(clone_returns, prices_clean, all_fields, profiles, econ, yield_curve)
    else:
        return _phase3_xgboost(clone_returns, prices_clean, econ, yield_curve)


def _phase3_gnn(clone_returns, prices_clean, all_fields, profiles, econ, yield_curve):
    """Phase 3 via GNN Supervisor."""
    print("\n" + "=" * 60)
    print("PHASE 3: GNN Supervisor (CRISP-inspired)")
    print("=" * 60)

    from src.gnn_supervisor import run_gnn_supervisor_pipeline, plot_alpha_over_time

    supervised_returns, alpha, model = run_gnn_supervisor_pipeline(
        clone_returns, prices_clean, all_fields, profiles, econ, yield_curve,
        fold=4, window=20, verbose=True,
    )

    plot_alpha_over_time(alpha, save_dir=RESULTS_DIR)

    spy_returns = prices_clean[BENCHMARK].pct_change(fill_method=None).iloc[1:]
    test_start  = supervised_returns.index[0]
    test_clone  = clone_returns.loc[test_start:]
    test_spy    = spy_returns.loc[test_start:]

    canadian_prices = prices_clean.drop(columns=[BENCHMARK], errors="ignore")
    benchmarks = run_all_benchmarks(
        clone_returns, spy_returns, canadian_prices,
        test_start=str(test_start.date()),
    )

    results = {
        "Clone + GNN Supervisor": supervised_returns,
        "Clone (unsupervised)": test_clone,
        "SPY Buy & Hold": test_spy,
    }
    for name, rets in benchmarks.items():
        results[name] = rets.loc[rets.index >= test_start]

    from src.backtester import compare_benchmarks
    comparison = compare_benchmarks(results, test_spy)
    comparison.to_csv(GNN_RESULTS_DIR / "phase3_gnn_comparison.csv")
    print(f"\n[phase3-gnn] Saved: {GNN_RESULTS_DIR / 'phase3_gnn_comparison.csv'}")

    plot_supervised_vs_unsupervised(supervised_returns, clone_returns, spy_returns)

    # Return dummy model/X_test so downstream SHAP phases still run (on XGBoost fallback)
    returns_all = prices_clean.pct_change().iloc[1:]
    X_full = build_super_state(clone_returns, returns_all, econ, yield_curve)
    X_test = X_full.loc[X_full.index > "2019-12-31"]

    return supervised_returns, alpha, model, X_test


def _phase3_xgboost(clone_returns, prices_clean, econ, yield_curve):
    """Phase 3: original XGBoost meta-labeling supervisor."""

    # Get all returns for uncertainty computation
    returns_all = prices_clean.pct_change(fill_method=None).iloc[1:]

    # Run supervisor pipeline (train up to 2019, test 2020+)
    supervised_returns, regime, model, confidence, importances, allocation = (
        run_supervisor_pipeline(
            clone_returns, returns_all, econ, yield_curve,
            train_end="2019-12-31",
        )
    )

    # Get SPY returns for comparison
    spy_returns = prices_clean[BENCHMARK].pct_change(fill_method=None).iloc[1:]

    # Compare all strategies on the test period
    test_start = supervised_returns.index[0]
    test_clone = clone_returns.loc[test_start:]
    test_spy = spy_returns.loc[test_start:]

    # Run all benchmark strategies
    canadian_prices = prices_clean.drop(columns=[BENCHMARK], errors="ignore")
    benchmarks = run_all_benchmarks(
        clone_returns, spy_returns, canadian_prices,
        test_start=str(test_start.date())
    )

    # Combine all results for comparison
    results = {
        "Clone + AI Supervisor": supervised_returns,
        "Clone (unsupervised)": test_clone,
        "SPY Buy & Hold": test_spy,
    }
    # Add benchmark strategies
    for name, returns in benchmarks.items():
        # Align to test period
        aligned = returns.loc[returns.index >= test_start]
        results[name] = aligned

    comparison = compare_benchmarks(results, test_spy)

    comparison.to_csv(RESULTS_DIR / "phase3_comparison.csv")
    print(f"\n[phase3] Saved: {RESULTS_DIR / 'phase3_comparison.csv'}")

    # Phase 3 plots
    print("\nGenerating Phase 3 plots...")
    plot_supervisor_decisions(confidence, clone_returns, regime)
    plot_supervised_vs_unsupervised(supervised_returns, clone_returns, spy_returns)

    # Save feature importances
    importances.to_csv(RESULTS_DIR / "feature_importances.csv")
    print(f"[phase3] Saved: {RESULTS_DIR / 'feature_importances.csv'}")

    # Build X_test for SHAP analysis
    returns_all = prices_clean.pct_change().iloc[1:]
    X_full = build_super_state(clone_returns, returns_all, econ, yield_curve)
    X_test = X_full.loc[X_full.index > "2019-12-31"]

    # Ensure X_test has exactly the same columns the model was trained on
    # (build_super_state may produce different columns on full vs train data)
    try:
        model_features = model.get_booster().feature_names
    except AttributeError:
        model_features = list(X_test.columns)
    X_test = X_test.reindex(columns=model_features)

    return supervised_returns, regime, model, X_test


def phase4_shap(model, X_test):
    """Phase 4: SHAP Analysis (XGBoost supervisor only)."""
    if USE_GNN_SUPERVISOR:
        print("\n[phase4] SHAP skipped — GNN supervisor active (not XGBoost).")
        return None
    shap_values = run_shap_analysis(model, X_test, save_dir=RESULTS_DIR)
    return shap_values


def phase5b_train_gnn(prices_clean, all_fields, profiles, econ, yield_curve):
    """
    Phase 5b: Train the GNN Supervisor (walk-forward, 4 folds).

    Only needed once. After training, flip USE_GNN_SUPERVISOR = True
    and re-run main.py to use the GNN in phase3.

    Training time: ~5–15 min on GPU (RTX 3090), ~30–90 min on CPU.
    """
    print("\n" + "=" * 60)
    print("PHASE 5b: GNN Supervisor Training (walk-forward)")
    print("=" * 60)
    from src.gnn_train import train_gnn
    fold_results, *_ = train_gnn(
        prices_clean, all_fields, profiles, econ, yield_curve,
        window=20, epochs=50, patience=10, verbose=True,
    )
    return fold_results


def phase5_ablation(clone_returns, prices_clean, econ, yield_curve):
    """Phase 5: Ablation Study."""
    returns_all = prices_clean.pct_change().iloc[1:]
    results = run_ablation_study(
        clone_returns, returns_all, econ, yield_curve,
        train_end="2019-12-31", save_dir=RESULTS_DIR,
    )
    return results


def phase6_synthetic(prices_clean, econ, yield_curve, n_paths=100):
    """Phase 6: Synthetic Validation (slow — run with small n_paths first)."""
    returns = prices_clean.pct_change().iloc[1:]
    results = run_synthetic_validation(
        returns, econ, yield_curve,
        n_paths=n_paths,
        benchmark_col=BENCHMARK,
        train_frac=0.6,
        min_train=15,
        min_test=3,
        min_successful=50,
    )
    if len(results) > 0:
        plot_synthetic_validation(results, save_dir=RESULTS_DIR)
    return results


def main():
    print("Portfolio Optimizer - Full Pipeline")
    print("=" * 60)
    print(f"Supervisor mode: {'GNN (CRISP)' if USE_GNN_SUPERVISOR else 'XGBoost'}")

    # Load all data
    prices, all_fields, profiles, econ, yield_curve = load_all()

    # Phase 1: Data Prep & EDA
    prices_clean, returns = phase1(prices, all_fields, profiles, econ, yield_curve)

    # Phase 2: Worker (QP Solver)
    weights_history, clone_returns = phase2(prices_clean)

    # Phase 5b: Train GNN if needed (only when USE_GNN_SUPERVISOR=True)
    if USE_GNN_SUPERVISOR:
        ckpt = GNN_RESULTS_DIR / "gnn_checkpoint_fold4.pt"
        if not ckpt.exists():
            print("\n[main] No GNN checkpoint found — training now (phase 5b)...")
            phase5b_train_gnn(prices_clean, all_fields, profiles, econ, yield_curve)
        else:
            print(f"\n[main] GNN checkpoint found: {ckpt.name} — skipping training.")

    # Phase 3: Supervisor (XGBoost or GNN depending on USE_GNN_SUPERVISOR)
    supervised_returns, regime_or_alpha, model, X_test = phase3(
        clone_returns, prices_clean, all_fields, profiles, econ, yield_curve
    )

    # Phase 4: SHAP Analysis (XGBoost only — auto-skipped for GNN)
    shap_values = phase4_shap(model, X_test)

    # Phase 5: Ablation Study
    ablation_results = phase5_ablation(
        clone_returns, prices_clean, econ, yield_curve
    )

    # Phase 6: Synthetic Validation (optional — slow)
    # n_paths=50 ~5 min; n_paths=1000 ~30-60 min
    synthetic_results = phase6_synthetic(prices_clean, econ, yield_curve, n_paths=1000)

    print("\nAll phases complete! Check results/ for outputs.")


if __name__ == "__main__":
    main()
