"""
Ablation Study — Feature Group Importance.

Systematically removes feature groups from the Supervisor's input
and measures the impact on out-of-sample Sharpe ratio. Validates
that each feature category contributes meaningfully to performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import io, sys

from src.supervisor import (
    generate_meta_labels,
    build_super_state,
    train_classifier,
    apply_supervisor,
)
from src.backtester import compute_metrics
from src.config import RESULTS_DIR


# ──────────────────────────────────────────────
# Feature Group Definitions
# ──────────────────────────────────────────────

FEATURE_GROUPS = {
    "Volatility": lambda cols: [c for c in cols if any(
        k in c for k in ["vol_", "ewma_", "dispersion"]
    )],
    "Macro (MOVE/DXY)": lambda cols: [c for c in cols if any(
        k in c for k in ["macro_MOVE", "macro_DXY", "macro_VIX"]
    )],
    "Yield Curve": lambda cols: [c for c in cols if any(
        k in c for k in ["yc_", "macro_T10Y2Y", "macro_YIELD"]
    )],
    "Credit Spreads": lambda cols: [c for c in cols if any(
        k in c.upper() for k in ["CREDIT", "IG_", "HY_", "SPREAD"]
    ) and "yc" not in c],
    "Downside Risk": lambda cols: [c for c in cols if any(
        k in c for k in ["downside", "max_dd", "drawdown"]
    )],
    "Momentum": lambda cols: [c for c in cols if any(
        k in c for k in ["clone_ret_", "momentum"]
    )],
}


def _train_and_evaluate(X_train, y_train, X_test, clone_returns_test):
    """Train XGBoost and compute Sharpe on test set."""
    # Suppress output
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        model, _ = train_classifier(X_train, y_train, n_splits=3)

        confidence_test = pd.Series(
            model.predict_proba(X_test)[:, 1],
            index=X_test.index,
        )

        test_clone = clone_returns_test.loc[confidence_test.index]
        supervised_returns, _, _ = apply_supervisor(test_clone, confidence_test)
        metrics = compute_metrics(supervised_returns)

        return {
            "sharpe": metrics.get("Sharpe", np.nan),
            "max_drawdown": metrics.get("Max DD (%)", np.nan),
            "ann_return": metrics.get("Ann Return (%)", np.nan),
        }
    except Exception:
        return {"sharpe": np.nan, "max_drawdown": np.nan, "ann_return": np.nan}
    finally:
        sys.stdout = old_stdout


def run_ablation_study(clone_returns, returns_all, econ, yield_curve,
                        train_end="2019-12-31", save_dir=None):
    """
    Run ablation study by removing feature groups one at a time.

    Parameters
    ----------
    clone_returns : pd.Series
        Worker portfolio returns.
    returns_all : pd.DataFrame
        All asset returns.
    econ : pd.DataFrame
        Economic indicators.
    yield_curve : pd.DataFrame
        Yield curve data.
    train_end : str
        Training period cutoff.
    save_dir : Path, optional
        Where to save results.

    Returns
    -------
    results_df : pd.DataFrame
        Ablation results with Sharpe for each experiment.
    """
    if save_dir is None:
        save_dir = RESULTS_DIR
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("ABLATION STUDY — Feature Group Importance")
    print("=" * 60)

    # Build features and labels
    labels = generate_meta_labels(clone_returns, threshold=0.02, horizon=5)
    X = build_super_state(clone_returns, returns_all, econ, yield_curve)

    common = X.index.intersection(labels.index)
    X = X.loc[common]
    y = labels.loc[common]

    train_mask = X.index <= train_end
    test_mask = X.index > train_end

    X_train_full = X.loc[train_mask]
    y_train = y.loc[train_mask]
    X_test_full = X.loc[test_mask]
    clone_test = clone_returns.loc[X_test_full.index]

    all_cols = list(X.columns)
    results = []

    # ── Baseline: All features ──
    print(f"\n[ablation] Running baseline (all {len(all_cols)} features)...")
    baseline = _train_and_evaluate(X_train_full, y_train, X_test_full, clone_test)
    results.append({
        "experiment": "Full Model (Baseline)",
        "features_removed": 0,
        "features_remaining": len(all_cols),
        **baseline,
    })
    print(f"  Baseline Sharpe: {baseline['sharpe']:.4f}")

    # ── Remove each feature group ──
    for group_name, selector_fn in FEATURE_GROUPS.items():
        removed_cols = selector_fn(all_cols)
        if not removed_cols:
            print(f"\n[ablation] {group_name}: no matching features, skipping")
            continue

        remaining_cols = [c for c in all_cols if c not in removed_cols]
        if not remaining_cols:
            print(f"\n[ablation] {group_name}: would remove ALL features, skipping")
            continue

        print(f"\n[ablation] Removing {group_name} "
              f"({len(removed_cols)} features)...")

        X_train_ablated = X_train_full[remaining_cols]
        X_test_ablated = X_test_full[remaining_cols]

        result = _train_and_evaluate(
            X_train_ablated, y_train, X_test_ablated, clone_test
        )

        sharpe_drop = baseline["sharpe"] - result["sharpe"]
        pct_drop = (sharpe_drop / abs(baseline["sharpe"])) * 100 if baseline["sharpe"] != 0 else 0

        results.append({
            "experiment": f"Remove {group_name}",
            "features_removed": len(removed_cols),
            "features_remaining": len(remaining_cols),
            **result,
        })
        print(f"  Sharpe: {result['sharpe']:.4f} "
              f"(Delta = {-sharpe_drop:+.4f}, {-pct_drop:+.1f}%)")

    results_df = pd.DataFrame(results)

    # ── Summary ──
    print(f"\n{'='*60}")
    print("ABLATION RESULTS")
    print(f"{'='*60}")
    print(f"{'Experiment':<30s} {'Sharpe':>8s} {'D.Sharpe':>10s} {'MaxDD':>8s}")
    print("-" * 60)
    for _, row in results_df.iterrows():
        delta = row["sharpe"] - baseline["sharpe"]
        print(f"{row['experiment']:<30s} {row['sharpe']:>8.4f} "
              f"{delta:>+10.4f} {row['max_drawdown']:>8.2%}")

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(10, 6))

    experiments = results_df["experiment"].values
    sharpes = results_df["sharpe"].values
    colors = ["#2ecc71" if s >= baseline["sharpe"] * 0.95 else
              "#f39c12" if s >= baseline["sharpe"] * 0.80 else
              "#e74c3c" for s in sharpes]

    bars = ax.barh(range(len(experiments)), sharpes, color=colors,
                    edgecolor="white", height=0.6)

    # Baseline reference line
    ax.axvline(baseline["sharpe"], color="black", linestyle="--",
               linewidth=1, alpha=0.7, label=f"Baseline ({baseline['sharpe']:.3f})")

    ax.set_yticks(range(len(experiments)))
    ax.set_yticklabels(experiments, fontsize=10)
    ax.set_xlabel("Sharpe Ratio", fontsize=12)
    ax.set_title("Ablation Study — Feature Group Importance", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    plot_path = save_dir / "ablation_study.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[ablation] Saved: {plot_path}")

    csv_path = save_dir / "ablation_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"[ablation] Saved: {csv_path}")

    return results_df
