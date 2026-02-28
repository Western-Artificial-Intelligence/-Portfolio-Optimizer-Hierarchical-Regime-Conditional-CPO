import torch
import numpy as np
import pandas as pd
from pathlib import Path

from src.data_loader import load_all
from src.features import compute_returns, filter_sparse_tickers
from src.qp_solver import rolling_optimization
from src.gnn_data import build_graph_dataset
from src.gnn_model import DynamicGNNSupervisor
from src.config import GNN_RESULTS_DIR, BENCHMARK

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_sharpe(returns_array):
    if len(returns_array) < 5 or returns_array.std() < 1e-10:
        return 0.0
    return float(returns_array.mean() / returns_array.std() * np.sqrt(252))

def evaluate_folds():
    prices, all_fields, profiles, econ, yield_curve = load_all()
    prices_clean, _ = filter_sparse_tickers(prices, min_coverage=0.5)
    
    returns = prices_clean.pct_change(fill_method=None).iloc[1:]
    weights_history, worker_returns = rolling_optimization(
        prices_clean, benchmark_col=BENCHMARK,
        lookback=252 * 5, rebal_freq="ME",
    )
    
    windows, worker_ret_aligned, edge_index, dates, n_features, n_nodes = build_graph_dataset(
        returns, all_fields, profiles, econ, yield_curve,
        worker_returns=worker_returns, window=20,
    )

    for fold in range(1, 5):
        ckpt_path = GNN_RESULTS_DIR / f"gnn_checkpoint_fold{fold}.pt"
        if not ckpt_path.exists():
            print(f"Fold {fold}: Missing checkpoint")
            continue
            
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        val_sharpe = ckpt.get("val_sharpe", 0.0)
        train_end = ckpt["train_end"]
        val_end = ckpt["val_end"]
        test_end = ckpt["test_end"]
        
        # Load model
        model = DynamicGNNSupervisor(
            n_features=n_features,
            lstm_hidden=32,
            gat_hidden=32,
            gat_heads=4,
            n_nodes=n_nodes,
            seq_len=20,
        ).to(DEVICE)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        
        # Get test features
        test_mask = (dates > pd.Timestamp(val_end)) & (dates <= pd.Timestamp(test_end))
        n_test = test_mask.sum()
        
        if n_test == 0:
            print(f"Fold {fold}: val={val_sharpe:.3f}, test=No test days")
            continue
            
        windows_test = windows[test_mask]
        worker_test = worker_ret_aligned[test_mask]
        
        feat_mean = ckpt["feat_mean"]
        feat_std = ckpt["feat_std"]
        
        # Normalize
        T_w, wl, N, F = windows_test.shape
        flat = windows_test.reshape(T_w * wl, N, F)
        normed = (flat - feat_mean) / feat_std
        windows_test = normed.reshape(T_w, wl, N, F)
        
        test_alphas = []
        with torch.no_grad():
            for i in range(n_test):
                x = torch.tensor(windows_test[i], dtype=torch.float32).to(DEVICE)
                x = x.permute(1, 0, 2)
                alpha, _ = model(x)
                test_alphas.append(alpha.item())
                
        test_alphas = np.array(test_alphas)
        test_port_ret = test_alphas * worker_test
        test_sharpe = compute_sharpe(test_port_ret)
        
        print(f"Fold {fold} | Val Sharpe: {val_sharpe:.3f} | Test Sharpe: {test_sharpe:.3f}")

if __name__ == "__main__":
    evaluate_folds()
