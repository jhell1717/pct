from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .utils import knn_indices, index_points

class RelPosEncoding(nn.Module):
    def __init__(self, d_model: int, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden),  # dx, dy, dz, r
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, rel: torch.Tensor) -> torch.Tensor:
        """rel: (B, N, k, 4) → (B, N, k, d_model)"""
        return self.mlp(rel)
    
# -------------------------------
# Transformer Block for Points
# -------------------------------

class PointTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, k: int = 16, ff_mult: int = 4, drop: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.k = k
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.norm_q = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.rel_enc = RelPosEncoding(d_model)
        self.attn_out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(drop)

        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(d_model * ff_mult, d_model),
        )

    def forward(self, xyz: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        """
        xyz:   (B, N, 3)
        feats: (B, N, d_model)
        returns updated feats: (B, N, d_model)
        """
        B, N, _ = xyz.shape
        k = min(self.k, N - 1)
        idx = knn_indices(xyz, k=k)  # (B, N, k)

        x = self.norm_q(feats)
        q = self.q_proj(x)  # (B, N, d)
        k_lin = self.k_proj(x)  # (B, N, d)
        v_lin = self.v_proj(x)  # (B, N, d)

        # Gather neighbor keys/values
        k_nb = index_points(k_lin, idx)  # (B, N, k, d)
        v_nb = index_points(v_lin, idx)  # (B, N, k, d)

        # Relative encoding
        nb_xyz = index_points(xyz, idx)              # (B, N, k, 3)
        rel = torch.cat([nb_xyz - xyz.unsqueeze(2),  # (B, N, k, 3)
                          torch.norm(nb_xyz - xyz.unsqueeze(2), dim=-1, keepdim=True)], dim=-1)  # (B, N, k, 4)
        rel_e = self.rel_enc(rel)  # (B, N, k, d)

        k_nb = k_nb + rel_e
        v_nb = v_nb + rel_e

        # Reshape for multi-head
        def split_heads(t):
            # (B, N, k, d) → (B, nH, N, k, dH)
            return t.view(B, N, -1, self.n_heads, self.d_head).permute(0, 3, 1, 2, 4)

        qh = q.view(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)           # (B, nH, N, dH)
        kh = split_heads(k_nb)  # (B, nH, N, k, dH)
        vh = split_heads(v_nb)  # (B, nH, N, k, dH)

        # Attention: (q · k)
        attn = torch.einsum('bhnc,bhnkc->bhnk', qh, kh) / math.sqrt(self.d_head)  # (B, nH, N, k)
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('bhnk,bhnkc->bhnc', attn, vh)  # (B, nH, N, dH)

        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, -1)  # (B, N, d)
        out = self.attn_out(out)
        out = self.drop(out)

        # Residual + FFN
        feats = feats + out
        feats = feats + self.ff(feats)
        return feats
    
# -------------------------------
# Encoder + Readout
# -------------------------------

class PointTransformerEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, d_model: int = 128, depth: int = 4, n_heads: int = 4, k: int = 16, drop: float = 0.0):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.blocks = nn.ModuleList([
            PointTransformerBlock(d_model=d_model, n_heads=n_heads, k=k, drop=drop)
            for _ in range(depth)
        ])
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, xyz: torch.Tensor, feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return per-point features (B, N, d_model). If feats is None, uses xyz as input features.
        in_channels matches feats.size(-1) if provided, else 3.
        """
        if feats is None:
            feats = xyz
        x = self.input_proj(feats)
        for blk in self.blocks:
            x = blk(xyz, x)
        return self.out_norm(x)


class GlobalReadout(nn.Module):
    def __init__(self, d_model: int, latent_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, latent_dim),
        )

    def forward(self, per_point: torch.Tensor) -> torch.Tensor:
        # per_point: (B, N, d)
        mean_pool = per_point.mean(dim=1)
        max_pool = per_point.max(dim=1).values
        global_feat = torch.cat([mean_pool, max_pool], dim=-1)
        return self.proj(global_feat)  # (B, latent_dim)

class PointCloudTransformer(nn.Module):
    def __init__(self, in_channels: int = 3, d_model: int = 128, depth: int = 4, n_heads: int = 4, k: int = 16,
                 latent_dim: int = 256, out_dim: Optional[int] = None, drop: float = 0.0):
        super().__init__()
        self.encoder = PointTransformerEncoder(in_channels, d_model, depth, n_heads, k, drop)
        self.readout = GlobalReadout(d_model, latent_dim)
        self.reg_head = None
        if out_dim is not None:
            self.reg_head = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, latent_dim // 2),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(latent_dim // 2, out_dim),
            )

    @torch.no_grad()
    def embed(self, xyz: torch.Tensor, feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        per_point = self.encoder(xyz, feats)
        return self.readout(per_point)

    def forward(self, xyz: torch.Tensor, feats: Optional[torch.Tensor] = None, return_latent: bool = False):
        per_point = self.encoder(xyz, feats)
        latent = self.readout(per_point)
        if self.reg_head is None:
            return latent if return_latent else per_point
        pred = self.reg_head(latent)
        if return_latent:
            return pred, latent
        return pred