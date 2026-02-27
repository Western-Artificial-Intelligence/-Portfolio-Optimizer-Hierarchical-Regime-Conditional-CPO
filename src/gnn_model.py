"""
Dynamic GNN Supervisor — CRISP-inspired Model Architecture.

Pure PyTorch (no PyTorch Geometric required).

Architecture:
    LSTM (shared, per node) → temporal node embeddings
    Multi-head GAT (fully connected) → relational market state
    Global mean pool → single market vector
    MLP → α ∈ [0, 1]   (blending: α × clone + (1−α) × cash)

Loss:
    −Sharpe × √252  +  λ × mean|Δα|
    Trained directly on portfolio returns — no binary labels needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ──────────────────────────────────────────────
# 1. Multi-Head Graph Attention (pure PyTorch)
# ──────────────────────────────────────────────

class MultiHeadGraphAttention(nn.Module):
    """
    Multi-head Graph Attention layer — pure PyTorch, no external graph lib.

    For N nodes (fully connected):
      1. Transform each node: Wh[i] = W × h[i]                   [N, heads, out_dim]
      2. Compute attention scores: e[i,j,k] = LeakyReLU(aᵀ [Wh[i,k] ‖ Wh[j,k]])
      3. Normalize: attn[i,j,k] = softmax over j (source nodes)
      4. Aggregate: out[i,k] = Σⱼ attn[i,j,k] × Wh[j,k]

    Memory: O(N²) — perfectly fine for N=33.
    """

    def __init__(self, in_dim, out_dim, heads=4, dropout=0.2, concat=True):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.concat = concat
        self.dropout_p = dropout

        # Shared linear transform (all heads at once)
        self.W = nn.Linear(in_dim, out_dim * heads, bias=False)

        # Attention parameter: [heads, 2*out_dim]
        self.a = nn.Parameter(torch.empty(heads, 2 * out_dim))
        nn.init.xavier_uniform_(self.a.unsqueeze(0))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        """
        h : [N, in_dim]
        Returns:
            out  : [N, heads*out_dim] if concat else [N, out_dim]
            attn : [N, N, heads]  attention weights for interpretability
        """
        N = h.size(0)

        # Transform all nodes: [N, heads, out_dim]
        Wh = self.W(h).view(N, self.heads, self.out_dim)

        # Broadcast to all (i, j) pairs
        Wh_i = Wh.unsqueeze(1).expand(N, N, self.heads, self.out_dim)  # target
        Wh_j = Wh.unsqueeze(0).expand(N, N, self.heads, self.out_dim)  # source

        # Concatenate and score: [N, N, heads]
        pair = torch.cat([Wh_i, Wh_j], dim=-1)              # [N, N, heads, 2*out_dim]
        e = self.leaky_relu((pair * self.a).sum(dim=-1))     # [N, N, heads]

        # Softmax over source nodes (dim=1 = j dimension)
        attn = F.softmax(e, dim=1)                           # [N, N, heads]
        attn = self.dropout(attn)

        # Aggregate messages: out[i, k, :] = Σⱼ attn[i,j,k] * Wh[j,k,:]
        # einsum: (N_target, N_source, heads) × (N_source, heads, out_dim) → (N_target, heads, out_dim)
        out = torch.einsum("ijk,jkl->ikl", attn, Wh)        # [N, heads, out_dim]

        if self.concat:
            return out.reshape(N, self.heads * self.out_dim), attn  # [N, heads*out_dim]
        else:
            return out.mean(dim=1), attn                             # [N, out_dim]


# ──────────────────────────────────────────────
# 2. Full GNN Supervisor Model
# ──────────────────────────────────────────────

class DynamicGNNSupervisor(nn.Module):
    """
    CRISP-inspired Dynamic GNN Supervisor.

    Reads a 20-day window of per-stock features and outputs a single
    blending coefficient α ∈ [0, 1] representing the model's confidence
    that the market is in a Risk-On regime.

    Parameters
    ----------
    n_features  : int   — feature dimension per node per day
    lstm_hidden : int   — LSTM output size per node (default 32)
    gat_hidden  : int   — GAT hidden dimension (default 32)
    gat_heads   : int   — number of attention heads (default 4)
    n_nodes     : int   — number of stocks (default 33)
    seq_len     : int   — lookback window (default 20 trading days)
    dropout     : float — regularization (default 0.2)
    """

    def __init__(self, n_features=23, lstm_hidden=32, gat_hidden=32,
                 gat_heads=4, n_nodes=33, seq_len=20, dropout=0.2):
        super().__init__()
        self.n_nodes = n_nodes
        self.seq_len = seq_len

        # ── Temporal encoder: shared LSTM applied per node ──────────────────
        # Processes each stock's 20-day feature sequence independently.
        # "Shared" = same weights for all 33 stocks (weight tying).
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,    # input: [N, seq, features]
        )

        # ── Spatial encoder: 2-layer GAT (fully connected graph) ────────────
        # Layer 1: concat=True → [N, gat_heads * gat_hidden]
        self.gat1 = MultiHeadGraphAttention(
            in_dim=lstm_hidden,
            out_dim=gat_hidden,
            heads=gat_heads,
            dropout=dropout,
            concat=True,
        )
        # Layer 2: concat=False → [N, gat_hidden]
        self.gat2 = MultiHeadGraphAttention(
            in_dim=gat_hidden * gat_heads,
            out_dim=gat_hidden,
            heads=1,
            dropout=dropout,
            concat=False,
        )

        # Batch norm after GAT layer 1
        self.bn1 = nn.BatchNorm1d(gat_hidden * gat_heads)

        # ── Output head: market state → blending coefficient ─────────────────
        # Global mean pool (all nodes → 1 vector) then MLP → sigmoid
        self.output_head = nn.Sequential(
            nn.Linear(gat_hidden, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            # No Sigmoid here — applied manually in forward() with α floor
        )

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for one daily graph snapshot.

        Parameters
        ----------
        x : Tensor [N, seq_len, n_features]
            20-day feature window for all N stocks.

        Returns
        -------
        alpha : scalar Tensor ∈ [0, 1]
            Blending coefficient (1 = fully invested, 0 = cash).
        attn  : Tensor [N, N, 1]  (from GAT layer 2)
            Attention weights — interpretable: which pairs matter today.
        """
        N, T, n_feat = x.shape   # renamed: was 'F' which shadowed torch.nn.functional as F

        # ── Step 1: LSTM temporal encoding ───────────────────────────────────
        # Each stock gets its own 20-day sequence encoded.
        # Shared LSTM weights = same temporal dynamics across all stocks.
        _, (h_n, _) = self.lstm(x)          # h_n: [1, N, lstm_hidden]
        node_emb = h_n.squeeze(0)           # [N, lstm_hidden]

        # ── Step 2: Graph Attention (relational encoding) ────────────────────
        # GAT discovers which stock-pair relationships matter today.
        # e.g., during a crisis it might upweight bank→energy propagation.
        node_emb1, attn1 = self.gat1(node_emb)              # [N, heads*gat_hidden]
        node_emb1 = self.bn1(node_emb1)
        node_emb1 = F.elu(node_emb1)        # F = torch.nn.functional ✓
        node_emb1 = self.dropout_layer(node_emb1)

        node_emb2, attn2 = self.gat2(node_emb1)             # [N, gat_hidden]
        node_emb2 = F.elu(node_emb2)

        # ── Step 3: Global pool → market-level state ─────────────────────────
        # Mean of all 33 node embeddings gives an aggregate market health signal.
        market_emb = node_emb2.mean(dim=0)                   # [gat_hidden]

        # ── Step 4: Regress to blending coefficient ───────────────────────────
        raw = self.output_head(market_emb).squeeze()   # unbounded logit

        # α floor: 0.3 + 0.7 × sigmoid(raw) → α ∈ [0.30, 1.0]
        # Prevents the degenerate all-cash (α≈0) local minimum that the
        # Sharpe loss inadvertently rewards (std→0 ⟹ Sharpe→∞ at α=0).
        alpha = 0.3 + 0.7 * torch.sigmoid(raw)

        return alpha, attn2


# ──────────────────────────────────────────────
# 3. Loss Function
# ──────────────────────────────────────────────

def sharpe_loss(portfolio_returns, alpha_series, lambda_turnover=0.01,
                lambda_defensive=0.2):
    """
    Differentiable training loss — directly maximizes risk-adjusted returns.

        Loss = −Sharpe × √252  +  λ × mean|Δα|  +  λ₂ × max(0, 0.5 − mean(α))²

    The third term penalises the model for having a mean α below 0.5,
    preventing the degenerate 'always cash' local minimum.

    Parameters
    ----------
    portfolio_returns  : Tensor [B]  — daily portfolio returns in mini-batch
    alpha_series       : Tensor [B]  — corresponding α values
    lambda_turnover    : float       — penalty for excessive day-to-day switching
    lambda_defensive   : float       — penalty for α drifting below 0.5
    """
    mean_ret = portfolio_returns.mean()
    std_ret  = portfolio_returns.std() + 1e-6
    annualized_sharpe = mean_ret / std_ret * (252 ** 0.5)

    # Turnover: penalise large swings in α between consecutive days
    if alpha_series.shape[0] > 1:
        turnover = (alpha_series[1:] - alpha_series[:-1]).abs().mean()
    else:
        turnover = torch.tensor(0.0)

    # Defensiveness penalty: discourage mean α from drifting below 0.5
    mean_alpha = alpha_series.mean()
    defensive_penalty = torch.clamp(0.5 - mean_alpha, min=0.0) ** 2

    return -annualized_sharpe + lambda_turnover * turnover + lambda_defensive * defensive_penalty
