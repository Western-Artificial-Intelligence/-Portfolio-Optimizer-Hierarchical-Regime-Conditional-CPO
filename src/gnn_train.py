"""
GNN Training Script — Walk-Forward Cross-Validation.

Trains the DynamicGNNSupervisor on historical portfolio data using
expanding walk-forward splits (no data leakage). Saves one checkpoint
per fold to results2/gnn_checkpoint_fold{k}.pt.

Run standalone:
    python src/gnn_train.py

Imports for use in main.py:
    from src.gnn_train import train_gnn

Training time estimate (RTX 3090):   ~5–15 minutes for all 4 folds
Training time estimate (CPU only):   ~30–90 minutes for all 4 folds
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

from src.data_loader import load_all
from src.features import compute_returns, filter_sparse_tickers
from src.qp_solver import rolling_optimization
from src.gnn_data import build_graph_dataset, GNNDataset
from src.gnn_model import DynamicGNNSupervisor, sharpe_loss
from src.config import GNN_RESULTS_DIR, BENCHMARK

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Walk-forward folds: (train_end, val_end, test_end)
# Each fold trains on all data up to train_end, validates on the next year.
# Fold 4 is the "production" fold — model used for actual portfolio decisions.
FOLDS = [
    ("2015-12-31", "2016-12-31", "2017-12-31"),
    ("2016-12-31", "2017-12-31", "2018-12-31"),
    ("2017-12-31", "2018-12-31", "2019-12-31"),
    ("2019-12-31", "2020-12-31", "2024-12-31"),   # production fold
]


# ──────────────────────────────────────────────
# 1. Metrics
# ──────────────────────────────────────────────

def compute_sharpe(returns_array):
    """Annualized Sharpe from daily returns numpy array."""
    if len(returns_array) < 5 or returns_array.std() < 1e-10:
        return 0.0
    return float(returns_array.mean() / returns_array.std() * np.sqrt(252))


# ──────────────────────────────────────────────
# 2. Normalization
# ──────────────────────────────────────────────

def compute_norm_stats(windows):
    """
    Compute mean/std for z-score normalization.
    windows: [T, window, N, F]
    Returns: feat_mean [1, N, F], feat_std [1, N, F]  (numpy)
    """
    T_w, wl, N, F = windows.shape
    flat = windows.reshape(T_w * wl, N, F)              # [T*window, N, F]
    feat_mean = flat.mean(axis=0, keepdims=True)         # [1, N, F]
    feat_std  = flat.std(axis=0,  keepdims=True) + 1e-8  # [1, N, F]
    return feat_mean, feat_std


def normalize_windows(windows, feat_mean, feat_std):
    """Apply z-score normalization. windows: [T, window, N, F]"""
    T_w, wl, N, F = windows.shape
    flat = windows.reshape(T_w * wl, N, F)
    normed = (flat - feat_mean) / feat_std
    return normed.reshape(T_w, wl, N, F)


# ──────────────────────────────────────────────
# 3. Single-Fold Training
# ──────────────────────────────────────────────

def train_one_fold(fold_idx, train_end, val_end,
                   windows, worker_ret, dates,
                   n_features, n_nodes,
                   epochs=50, patience=10, lr=1e-3,
                   batch_days=32, lambda_turnover=0.01,
                   verbose=True):
    """
    Train GNN on one walk-forward fold.

    Returns
    -------
    model         : DynamicGNNSupervisor (best checkpoint)
    val_sharpe    : float
    feat_mean     : np.ndarray [1, N, F]   (for inference normalization)
    feat_std      : np.ndarray [1, N, F]
    """
    train_mask = dates <= pd.Timestamp(train_end)
    val_mask   = (dates > pd.Timestamp(train_end)) & (dates <= pd.Timestamp(val_end))

    n_train = train_mask.sum()
    n_val   = val_mask.sum()

    if n_train < 200:
        print(f"[gnn_train] Fold {fold_idx}: only {n_train} train days — skipping.")
        return None, None, None, None

    if verbose:
        print(f"  Train days: {n_train} | Val days: {n_val}")

    # Normalize using training statistics only (prevents data leakage)
    feat_mean, feat_std = compute_norm_stats(windows[train_mask])
    windows_train = normalize_windows(windows[train_mask],   feat_mean, feat_std)
    windows_val   = normalize_windows(windows[val_mask],     feat_mean, feat_std)
    worker_train  = worker_ret[train_mask]
    worker_val    = worker_ret[val_mask]

    # Build model
    model = DynamicGNNSupervisor(
        n_features=n_features,
        lstm_hidden=32,
        gat_hidden=32,
        gat_heads=4,
        n_nodes=n_nodes,
        seq_len=windows.shape[1],   # window length
        dropout=0.2,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_sharpe = -np.inf
    best_state      = None
    no_improve      = 0

    for epoch in range(1, epochs + 1):
        # ── Training ──────────────────────────────────────────────────────────
        model.train()
        indices     = np.random.permutation(n_train)
        batch_losses = []
        epoch_alphas = []
        epoch_rets   = []

        for start in range(0, n_train, batch_days):
            batch_idx = indices[start: start + batch_days]

            # Forward pass: one day at a time (each is a graph snapshot)
            batch_alphas = []
            for i in batch_idx:
                # x: [window, N, F] → permute to [N, window, F] for LSTM
                x = torch.tensor(windows_train[i], dtype=torch.float32).to(DEVICE)
                x = x.permute(1, 0, 2)          # [N, seq_len, features]
                alpha, _ = model(x)
                batch_alphas.append(alpha)

            alphas_t = torch.stack(batch_alphas)                            # [B]
            rets_t   = torch.tensor(worker_train[batch_idx],
                                    dtype=torch.float32).to(DEVICE)        # [B]

            # Portfolio return = α × worker return
            port_ret = alphas_t * rets_t

            loss = sharpe_loss(port_ret, alphas_t, lambda_turnover=lambda_turnover)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_losses.append(loss.item())
            epoch_alphas.extend(alphas_t.detach().cpu().numpy())
            epoch_rets.extend(rets_t.cpu().numpy())

        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_alphas = []
        with torch.no_grad():
            for i in range(n_val):
                x = torch.tensor(windows_val[i], dtype=torch.float32).to(DEVICE)
                x = x.permute(1, 0, 2)
                alpha, _ = model(x)
                val_alphas.append(alpha.item())

        val_alphas   = np.array(val_alphas)
        val_port_ret = val_alphas * worker_val
        val_sharpe   = compute_sharpe(val_port_ret)

        # Train Sharpe for logging
        ep_alphas_np = np.array(epoch_alphas)
        ep_rets_np   = np.array(epoch_rets)
        train_sharpe = compute_sharpe(ep_alphas_np * ep_rets_np)

        if verbose and epoch % 5 == 0:
            print(f"  Fold {fold_idx} | Epoch {epoch:3d} | "
                  f"Loss: {np.mean(batch_losses):+.4f} | "
                  f"Train Sharpe: {train_sharpe:.3f} | "
                  f"Val Sharpe: {val_sharpe:.3f} | "
                  f"α mean: {val_alphas.mean():.3f}")

        # Early stopping
        if val_sharpe > best_val_sharpe:
            best_val_sharpe = val_sharpe
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve  = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  Fold {fold_idx}: early stop @ epoch {epoch} "
                          f"(best val Sharpe: {best_val_sharpe:.3f})")
                break

    if best_state:
        model.load_state_dict(best_state)

    return model, best_val_sharpe, feat_mean, feat_std


# ──────────────────────────────────────────────
# 4. Full Training Pipeline
# ──────────────────────────────────────────────

def train_gnn(prices_clean, all_fields, profiles, econ, yield_curve,
              window=20, epochs=50, patience=10, verbose=True):
    """
    Full walk-forward GNN training.

    Runs 4 expanding-window folds. Saves a checkpoint for each.
    The last fold (fold 4) is the production model used by gnn_supervisor.

    Parameters
    ----------
    prices_clean : pd.DataFrame  — from phase1()
    all_fields   : dict          — from data_loader
    profiles     : pd.DataFrame
    econ         : pd.DataFrame
    yield_curve  : pd.DataFrame
    window       : int           — lookback window (default 20 days = ~1 month)
    epochs       : int           — max training epochs per fold
    patience     : int           — early stopping patience

    Returns
    -------
    fold_results : dict {fold_idx: (model, val_sharpe, feat_mean, feat_std)}
    """
    print(f"\n{'='*60}")
    print("GNN SUPERVISOR — Walk-Forward Training")
    print(f"Device: {DEVICE}  |  Window: {window}d  |  Max epochs: {epochs}")
    print(f"{'='*60}")

    # Step 1: Compute worker returns via existing QP solver
    print("\n[gnn_train] Computing worker (QP solver) returns...")
    returns = prices_clean.pct_change(fill_method=None).iloc[1:]
    weights_history, worker_returns = rolling_optimization(
        prices_clean, benchmark_col=BENCHMARK,
        lookback=252 * 5, rebal_freq="ME",
    )
    print(f"[gnn_train] Worker returns: {len(worker_returns)} days, "
          f"{worker_returns.index[0].date()} → {worker_returns.index[-1].date()}")

    # Step 2: Build graph dataset
    print("\n[gnn_train] Building graph dataset...")
    windows, worker_ret_aligned, edge_index, dates, n_features, n_nodes = (
        build_graph_dataset(
            returns, all_fields, profiles, econ, yield_curve,
            worker_returns=worker_returns, window=window,
        )
    )

    # Step 3: Walk-forward training
    fold_results = {}

    for fold_idx, (train_end, val_end, test_end) in enumerate(FOLDS):
        fold_num = fold_idx + 1
        print(f"\n[gnn_train] ══ Fold {fold_num}/{len(FOLDS)} ══")
        print(f"  Train: 2010→{train_end}  │  Val: →{val_end}  │  Test: →{test_end}")

        model, val_sharpe, feat_mean, feat_std = train_one_fold(
            fold_idx=fold_num,
            train_end=train_end,
            val_end=val_end,
            windows=windows,
            worker_ret=worker_ret_aligned,
            dates=dates,
            n_features=n_features,
            n_nodes=n_nodes,
            epochs=epochs,
            patience=patience,
            verbose=verbose,
        )

        if model is None:
            continue

        fold_results[fold_idx] = (model, val_sharpe, feat_mean, feat_std)

        # Save checkpoint
        ckpt_path = GNN_RESULTS_DIR / f"gnn_checkpoint_fold{fold_num}.pt"
        torch.save({
            "model_state": model.state_dict(),
            "feat_mean":   feat_mean,
            "feat_std":    feat_std,
            "n_features":  n_features,
            "n_nodes":     n_nodes,
            "window":      window,
            "val_sharpe":  val_sharpe,
            "train_end":   train_end,
            "val_end":     val_end,
            "test_end":    test_end,
        }, ckpt_path)

        print(f"[gnn_train] ✓ Saved: {ckpt_path.name}  (val Sharpe: {val_sharpe:.3f})")

    # Summary
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    if fold_results:
        for fi, (_, vs, _, _) in fold_results.items():
            marker = " ← production" if fi == len(FOLDS) - 1 else ""
            print(f"  Fold {fi+1}: val Sharpe = {vs:.3f}{marker}")

    return fold_results, windows, worker_ret_aligned, dates, n_features, n_nodes


# ──────────────────────────────────────────────
# Standalone entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    from src.features import filter_sparse_tickers

    print(f"[gnn_train] Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"[gnn_train] GPU: {torch.cuda.get_device_name(0)}")

    prices, all_fields, profiles, econ, yield_curve = load_all()
    prices_clean, _ = filter_sparse_tickers(prices, min_coverage=0.5)

    fold_results, *_ = train_gnn(
        prices_clean, all_fields, profiles, econ, yield_curve,
        window=20, epochs=50, patience=10, verbose=True,
    )
