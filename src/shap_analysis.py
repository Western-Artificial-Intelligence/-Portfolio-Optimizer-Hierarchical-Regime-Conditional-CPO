"""
SHAP Analysis — Explainability for the XGBoost Supervisor.

Generates SHAP beeswarm and dependence plots to demonstrate that
the Supervisor learns economically meaningful regime signals.

Reference: Lundberg & Lee (2017), "A Unified Approach to Interpreting
Model Predictions"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path

from src.config import RESULTS_DIR


def run_shap_analysis(model, X_test, save_dir=None, top_n=15):
    """
    Generate SHAP analysis plots for the XGBoost Supervisor.

    Parameters
    ----------
    model : XGBClassifier
        Trained XGBoost model from the Supervisor.
    X_test : pd.DataFrame
        Test set feature matrix.
    save_dir : Path, optional
        Directory to save plots.
    top_n : int
        Number of top features to display.

    Returns
    -------
    shap_values : np.ndarray
        SHAP values for each prediction.
    """
    if save_dir is None:
        save_dir = RESULTS_DIR
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("SHAP ANALYSIS — Feature Importance")
    print("=" * 60)

    # Compute SHAP values
    # Fix for XGBoost ≥ 2.1 + SHAP compatibility: base_score format changed
    # from float to bracketed string like '[8.788133E-1]'.
    # Patch Python's built-in float to handle the bracketed format.
    
    import builtins
    _original_float = builtins.float
    
    def _patched_float(x):
        if isinstance(x, str) and x.startswith("[") and x.endswith("]"):
            x = x.strip("[]")
        return _original_float(x)
    
    builtins.float = _patched_float
    print("[shap] Applied base_score compatibility patch")
    
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    finally:
        # Restore original float
        builtins.float = _original_float

    # ── 1. Beeswarm Plot (Global Feature Importance) ──
    print("[shap] Generating beeswarm plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_test,
        max_display=top_n,
        show=False,
        plot_size=None,
    )
    plt.title("SHAP Feature Importance — XGBoost Supervisor", fontsize=13)
    plt.tight_layout()
    beeswarm_path = save_dir / "shap_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[shap] Saved: {beeswarm_path}")

    # ── 2. Bar Plot (Mean |SHAP|) ──
    print("[shap] Generating bar plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X_test,
        plot_type="bar",
        max_display=top_n,
        show=False,
        plot_size=None,
    )
    plt.title("Mean |SHAP| — Feature Importance Ranking", fontsize=13)
    plt.tight_layout()
    bar_path = save_dir / "shap_bar.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[shap] Saved: {bar_path}")

    # ── 3. Top Dependence Plots ──
    # Find top features by mean |SHAP|
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_features = pd.Series(
        mean_abs_shap, index=X_test.columns
    ).sort_values(ascending=False).head(4)

    print(f"[shap] Generating dependence plots for top 4 features...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, (feat_name, _) in enumerate(top_features.items()):
        ax = axes[idx // 2][idx % 2]
        feat_idx = list(X_test.columns).index(feat_name)

        shap.dependence_plot(
            feat_idx, shap_values, X_test,
            ax=ax, show=False,
        )
        ax.set_title(f"SHAP Dependence: {feat_name}", fontsize=11)

    plt.suptitle("SHAP Dependence Plots — Top 4 Features", fontsize=13, y=1.02)
    plt.tight_layout()
    dep_path = save_dir / "shap_dependence.png"
    plt.savefig(dep_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[shap] Saved: {dep_path}")

    # ── 4. Summary Table ──
    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    csv_path = save_dir / "shap_importance.csv"
    importance_df.to_csv(csv_path, index=False)
    print(f"[shap] Saved: {csv_path}")

    print(f"\n[shap] Top {min(10, len(importance_df))} features by mean |SHAP|:")
    for _, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']:>35s}: {row['mean_abs_shap']:.4f}")

    return shap_values
