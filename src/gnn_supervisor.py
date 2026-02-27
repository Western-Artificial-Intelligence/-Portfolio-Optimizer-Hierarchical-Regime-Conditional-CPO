"""
GNN Supervisor — Inference and integration with the existing pipeline.

Drop-in replacement for supervisor.py's run_supervisor_pipeline().
Loads the trained GNN checkpoint and generates daily blending coefficients α.

Usage:
    from src.gnn_supervisor import run_gnn_supervisor_pipeline
    supervised_returns, alpha, model = run_gnn_supervisor_pipeline(
        clone_returns, prices_clean, all_fields, profiles, econ, yield_curve
    )
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from src.gnn_data import build_graph_dataset
from src.gnn_model import DynamicGNNSupervisor
from src.backtester import compute_metrics
from src.config import GNN_RESULTS_DIR, BENCHMARK

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────
# 1. Checkpoint Loading
# ──────────────────────────────────────────────

def load_checkpoint(ckpt_path):
    """
    Load GNN checkpoint and reconstruct model.

    Returns
    -------
    model     : DynamicGNNSupervisor  (eval mode, on DEVICE)
    ckpt      : dict                  (all saved metadata)
    """
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    model = DynamicGNNSupervisor(
        n_features=ckpt["n_features"],
        n_nodes=ckpt["n_nodes"],
        seq_len=ckpt["window"],
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, ckpt


# ──────────────────────────────────────────────
# 2. Main Inference Pipeline
# ──────────────────────────────────────────────

def run_gnn_supervisor_pipeline(clone_returns, prices_clean, all_fields,
                                profiles, econ, yield_curve,
                                fold=4, window=20, verbose=True):
    """
    Run the trained GNN supervisor on the test period.

    Uses the production fold checkpoint (fold 4: trained on 2010–2019,
    tested on 2020+). Returns the same interface as run_supervisor_pipeline()
    so all downstream phases work unchanged.

    Parameters
    ----------
    clone_returns : pd.Series       — QP solver daily returns
    prices_clean  : pd.DataFrame    — clean price history
    all_fields    : dict            — from data_loader.load_stock_history()
    profiles      : pd.DataFrame    — from data_loader.load_stock_profiles()
    econ          : pd.DataFrame    — economic indicators
    yield_curve   : pd.DataFrame    — yield curve data
    fold          : int             — checkpoint fold to use (default 4 = production)
    window        : int             — must match training window (default 20)
    verbose       : bool

    Returns
    -------
    supervised_returns : pd.Series  — GNN-blended portfolio returns
    alpha              : pd.Series  — daily blending coefficient ∈ [0, 1]
    model              : nn.Module  — trained GNN (for interpretability)
    """
    ckpt_path = GNN_RESULTS_DIR / f"gnn_checkpoint_fold{fold}.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"\n[gnn_supervisor] Checkpoint not found: {ckpt_path}\n"
            f"  → Train first:  python src/gnn_train.py\n"
            f"  → Or in main:   train_gnn(prices_clean, all_fields, ...)"
        )

    if verbose:
        print(f"\n{'='*60}")
        print("GNN SUPERVISOR — Inference")
        print(f"{'='*60}")
        print(f"[gnn_supervisor] Loading: {ckpt_path.name}")

    model, ckpt = load_checkpoint(ckpt_path)
    feat_mean  = ckpt["feat_mean"]     # np.ndarray [1, N, F]
    feat_std   = ckpt["feat_std"]      # np.ndarray [1, N, F]
    train_end  = ckpt["train_end"]
    val_sharpe_at_train = ckpt.get("val_sharpe", None)

    if verbose:
        print(f"[gnn_supervisor] Trained on:   2010 → {train_end}")
        print(f"[gnn_supervisor] Val Sharpe:   {val_sharpe_at_train:.3f}" if val_sharpe_at_train else "")
        print(f"[gnn_supervisor] Architecture: {ckpt['n_nodes']} nodes × "
              f"{ckpt['n_features']} features, window={ckpt['window']}d")
        print(f"[gnn_supervisor] Device:       {DEVICE}")

    # ── Build graph dataset (full history for feature computation) ───────────
    returns = prices_clean.pct_change(fill_method=None).iloc[1:]
    windows, _, _, dates, _, _ = build_graph_dataset(
        returns, all_fields, profiles, econ, yield_curve,
        worker_returns=None, window=window,
    )

    # Only predict on the test period (after training cutoff)
    test_mask     = dates > pd.Timestamp(train_end)
    windows_test  = windows[test_mask]
    test_dates    = dates[test_mask]

    if len(windows_test) == 0:
        raise ValueError(
            f"[gnn_supervisor] No test data found after {train_end}. "
            f"Data ends at {dates[-1].date()}."
        )

    # Normalize using training statistics (same as at training time)
    T_w, wl, N, F = windows_test.shape
    w_flat  = windows_test.reshape(T_w * wl, N, F)
    w_norm  = (w_flat - feat_mean) / feat_std
    windows_norm = w_norm.reshape(T_w, wl, N, F)

    if verbose:
        print(f"\n[gnn_supervisor] Predicting α for {len(windows_test)} test days "
              f"({test_dates[0].date()} → {test_dates[-1].date()})...")

    # ── Daily inference ──────────────────────────────────────────────────────
    alphas      = []
    attn_list   = []

    model.eval()
    with torch.no_grad():
        for i in range(len(windows_norm)):
            x = torch.tensor(windows_norm[i], dtype=torch.float32).to(DEVICE)
            x = x.permute(1, 0, 2)           # [N, window, F]
            alpha, attn = model(x)
            alphas.append(alpha.item())
            attn_list.append(attn.cpu().numpy())

    alpha_series = pd.Series(alphas, index=test_dates, name="gnn_alpha")

    # ── Blend clone with cash ────────────────────────────────────────────────
    clone_test         = clone_returns.reindex(test_dates).fillna(0.0)
    supervised_returns = (alpha_series * clone_test).rename("gnn_supervised")

    # ── Logging & regime check ───────────────────────────────────────────────
    if verbose:
        gnn_sharpe   = supervised_returns.mean() / (supervised_returns.std() + 1e-8) * np.sqrt(252)
        clone_sharpe = clone_test.mean() / (clone_test.std() + 1e-8) * np.sqrt(252)

        print(f"\n[gnn_supervisor] ── Performance Summary ──")
        print(f"  GNN Supervisor Sharpe : {gnn_sharpe:.3f}")
        print(f"  Clone (no supervisor) : {clone_sharpe:.3f}")
        print(f"  α — mean: {alpha_series.mean():.3f}  │  std: {alpha_series.std():.3f}  │  "
              f"min: {alpha_series.min():.3f}  │  max: {alpha_series.max():.3f}")

        # Regime alignment check — model should go defensive in known crises
        print(f"\n[gnn_supervisor] ── Regime Alignment Check ──")
        _check_regime(alpha_series, "2020-02-20", "2020-04-01", "COVID crash")
        _check_regime(alpha_series, "2022-01-01", "2022-10-01", "Inflation shock")
        _check_regime(alpha_series, "2020-04-01", "2021-12-31", "Post-COVID recovery")

    return supervised_returns, alpha_series, model


def _check_regime(alpha_series, start, end, label):
    """Print α mean for a given known regime period."""
    try:
        segment = alpha_series.loc[start:end]
        if len(segment) == 0:
            return
        mean_alpha = segment.mean()
        expected_defensive = "COVID" in label or "shock" in label.lower()
        status = ""
        if expected_defensive:
            status = "✓ defensive" if mean_alpha < 0.55 else "✗ not defensive (check training)"
        else:
            status = "✓ invested" if mean_alpha > 0.5 else "✗ under-invested"
        print(f"  {label} ({start[:7]} → {end[:7]}): α = {mean_alpha:.3f}  {status}")
    except Exception:
        pass


# ──────────────────────────────────────────────
# 3. Attention Visualisation (optional)
# ──────────────────────────────────────────────

def plot_alpha_over_time(alpha_series, save_dir=None):
    """
    Plot daily GNN blending coefficient α over time.
    Highlights known crisis periods for visual validation.
    """
    if save_dir is None:
        save_dir = GNN_RESULTS_DIR
    save_dir = Path(save_dir)

    fig, ax = plt.subplots(figsize=(14, 4))

    ax.fill_between(alpha_series.index, alpha_series.values,
                    alpha=0.4, color="#2ecc71", label="α (GNN confidence)")
    ax.plot(alpha_series.index, alpha_series.values,
            color="#27ae60", linewidth=0.8)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8,
               label="Neutral threshold (α = 0.5)")

    # Mark known crisis periods
    crises = [
        ("2020-02-20", "2020-04-15", "#e74c3c", "COVID"),
        ("2022-01-01", "2022-10-15", "#e67e22", "Inflation"),
    ]
    for start, end, color, name in crises:
        try:
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                       alpha=0.15, color=color, label=f"{name} crisis")
        except Exception:
            pass

    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("α (1 = fully invested, 0 = cash)", fontsize=11)
    ax.set_title("GNN Supervisor — Daily Blending Coefficient α", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    path = save_dir / "gnn_alpha_over_time.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[gnn_supervisor] Saved: {path}")
    return path
