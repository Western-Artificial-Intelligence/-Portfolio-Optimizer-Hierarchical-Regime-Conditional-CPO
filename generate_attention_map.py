import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from src.data_loader import load_all
from src.features import filter_sparse_tickers
from src.qp_solver import rolling_optimization
from src.gnn_data import build_graph_dataset
from src.gnn_model import DynamicGNNSupervisor
from src.config import GNN_RESULTS_DIR, BENCHMARK

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
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
    
    tickers = prices_clean.columns.drop(BENCHMARK).tolist() + [BENCHMARK]
    
    fold = 4
    ckpt_path = GNN_RESULTS_DIR / f"gnn_checkpoint_fold{fold}.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    
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
    
    # COVID crash mask: 2020-02-20 to 2020-03-31
    crash_mask = (dates >= pd.Timestamp("2020-02-20")) & (dates <= pd.Timestamp("2020-03-31"))
    
    windows_crash = windows[crash_mask]
    feat_mean = ckpt["feat_mean"]
    feat_std = ckpt["feat_std"]
    
    # Normalize
    T_w, wl, N, F = windows_crash.shape
    flat = windows_crash.reshape(T_w * wl, N, F)
    normed = (flat - feat_mean) / feat_std
    windows_crash = normed.reshape(T_w, wl, N, F)
    
    all_attns = []
    with torch.no_grad():
        for i in range(T_w):
            x = torch.tensor(windows_crash[i], dtype=torch.float32).to(DEVICE)
            x = x.permute(1, 0, 2)  # [N, seq, features]
            alpha, attn = model(x)
            all_attns.append(attn.cpu().numpy())
            
    # Average attention across the crash days
    mean_attn = np.mean(all_attns, axis=0).squeeze() # [N, N]
    
    # Optional: Log transform or something similar if it's too skewed to visualize well.
    # We will just plot the raw attention first.
    
    # Plotting
    plt.figure(figsize=(10, 8))
    sns.heatmap(mean_attn, cmap='Blues', xticklabels=tickers, yticklabels=tickers, cbar_kws={'label': 'Attention Weight'})
    plt.title('GNN Average Attention Weights (COVID Crash: Feb-Mar 2020)')
    plt.xlabel('Source Node')
    plt.ylabel('Target Node')
    plt.tight_layout()
    plt.savefig('docs/gnn_attention_map.png', dpi=300)
    print("Saved docs/gnn_attention_map.png")

if __name__ == "__main__":
    main()
