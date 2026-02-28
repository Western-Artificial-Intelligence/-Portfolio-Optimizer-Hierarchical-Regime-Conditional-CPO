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

# ── Institutional parameters ─────────────────────────────────────────────────
DRAWDOWN_STOP    = 0.10   # α clamped to α-floor if cumulative DD exceeds 10%
ALPHA_FLOOR_HARD = 0.30   # model's minimum α (matches the trained floor)


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

    # ── Institutional Feature 1: Hard Drawdown Stop ──────────────────────────
    # If cumulative portfolio drawdown from its running peak exceeds DRAWDOWN_STOP,
    # override α → ALPHA_FLOOR_HARD regardless of model output.
    # This is a compliance circuit breaker, not a model decision.
    clone_test_raw = clone_returns.reindex(test_dates).fillna(0.0)
    cum_ret   = (1 + clone_test_raw * alpha_series).cumprod()
    running_peak = cum_ret.cummax()
    drawdown     = (cum_ret - running_peak) / running_peak  # always ≤ 0
    hard_stop_active = drawdown < -DRAWDOWN_STOP
    alpha_series_adj = alpha_series.copy()
    alpha_series_adj[hard_stop_active] = ALPHA_FLOOR_HARD
    n_stopped = hard_stop_active.sum()
    if verbose and n_stopped > 0:
        print(f"[gnn_supervisor] Hard stop activated on {n_stopped} days "
              f"(>{DRAWDOWN_STOP*100:.0f}% drawdown) — α forced to {ALPHA_FLOOR_HARD}")

    alpha_series = alpha_series_adj

    # ── Blend clone with cash ────────────────────────────────────────────────
    clone_test         = clone_test_raw
    supervised_returns = (alpha_series * clone_test).rename("gnn_supervised")

    # ── Institutional Feature 2: Attention Logging (Compliance Report) ───────
    # For each test day, record the top-5 highest-attention stock pairs.
    # This gives compliance teams a human-readable log of *why* α changed.
    tickers = ckpt.get("tickers", [f"Asset_{i}" for i in range(ckpt["n_nodes"])])
    _save_attention_log(attn_list, test_dates, tickers)

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

def _save_attention_log(attn_list, dates, tickers, top_k=5):
    """
    Institutional Feature 2: Attention Logging.

    For each test day, extract the top-k highest-attention stock pairs
    from the GAT attention matrix and save as a CSV compliance report.

    This answers the question 'why did the model reduce α on day X?'
    by showing which cross-asset relationships the model was attending to.

    Output: results5/gnn_attention_log.csv
    Columns: date, rank, source_ticker, target_ticker, attention_weight
    """
    rows = []
    for i, (date, attn) in enumerate(zip(dates, attn_list)):
        # attn shape: [N, N, 1] — squeeze to [N, N]
        attn_sq = attn.squeeze(-1) if attn.ndim == 3 else attn
        N = attn_sq.shape[0]
        # Flatten upper triangle (directed pairs)
        pairs = []
        for src in range(N):
            for tgt in range(N):
                if src != tgt:
                    pairs.append((src, tgt, float(attn_sq[src, tgt])))
        # Sort by attention weight descending, take top_k
        pairs.sort(key=lambda x: -x[2])
        for rank, (src, tgt, w) in enumerate(pairs[:top_k], start=1):
            src_name = tickers[src] if src < len(tickers) else f"Asset_{src}"
            tgt_name = tickers[tgt] if tgt < len(tickers) else f"Asset_{tgt}"
            rows.append({
                "date": date.date(),
                "rank": rank,
                "source": src_name,
                "target": tgt_name,
                "attention": round(w, 6),
            })

    if rows:
        df = pd.DataFrame(rows)
        path = GNN_RESULTS_DIR / "gnn_attention_log.csv"
        df.to_csv(path, index=False)
        print(f"[gnn_supervisor] Attention log saved: {path}")
        print(f"  → {len(dates)} days × top-{top_k} pairs = {len(rows)} rows")


def plot_alpha_over_time(alpha_series, clone_returns=None, spy_returns=None,
                         save_dir=None):
    """
    Institutional Feature 3: GNN vs SPY cumulative return plot.

    Plot 1: Daily α with crisis shading.
    Plot 2 (if returns provided): Cumulative returns — GNN, clone, SPY.
    """
    if save_dir is None:
        save_dir = GNN_RESULTS_DIR
    save_dir = Path(save_dir)

    n_panels = 2 if (clone_returns is not None and spy_returns is not None) else 1
    fig, axes = plt.subplots(n_panels, 1,
                             figsize=(14, 4 * n_panels),
                             sharex=(n_panels > 1))
    if n_panels == 1:
        axes = [axes]

    # ── Panel 1: α over time ─────────────────────────────────────────────────
    ax = axes[0]
    ax.fill_between(alpha_series.index, alpha_series.values,
                    alpha=0.4, color="#2ecc71", label="α (GNN confidence)")
    ax.plot(alpha_series.index, alpha_series.values,
            color="#27ae60", linewidth=0.8)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8,
               label="Neutral threshold (α = 0.5)")
    ax.axhline(DRAWDOWN_STOP + ALPHA_FLOOR_HARD, color="#e74c3c",
               linestyle=":", linewidth=1.2, label="Hard stop floor")

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
    ax.set_ylabel("α (1=fully invested, 0=cash)", fontsize=11)
    ax.set_title("GNN Supervisor — Daily Blending Coefficient α", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Cumulative returns ──────────────────────────────────────────
    if n_panels == 2:
        ax2 = axes[1]
        common_idx = alpha_series.index

        gnn_ret   = (alpha_series * clone_returns.reindex(common_idx).fillna(0))
        spy_ret   = spy_returns.reindex(common_idx).fillna(0)
        clone_ret = clone_returns.reindex(common_idx).fillna(0)

        cum_gnn   = (1 + gnn_ret).cumprod()
        cum_spy   = (1 + spy_ret).cumprod()
        cum_clone = (1 + clone_ret).cumprod()

        ax2.plot(cum_gnn.index,   cum_gnn.values,   label="Clone + GNN Supervisor",
                 color="#2ecc71", linewidth=2.0)
        ax2.plot(cum_spy.index,   cum_spy.values,   label="SPY Buy & Hold",
                 color="#3498db", linewidth=1.5, linestyle="--")
        ax2.plot(cum_clone.index, cum_clone.values, label="Clone (no supervisor)",
                 color="#95a5a6", linewidth=1.2, linestyle=":")

        for start, end, color, name in crises:
            try:
                ax2.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                            alpha=0.15, color=color)
            except Exception:
                pass

        ax2.set_ylabel("Cumulative Return (1 = breakeven)", fontsize=11)
        ax2.set_title("Cumulative Performance: GNN Supervisor vs SPY", fontsize=13)
        ax2.legend(fontsize=9, loc="upper left")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = save_dir / "gnn_alpha_over_time.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[gnn_supervisor] Saved: {path}")
    return path
