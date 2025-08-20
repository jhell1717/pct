"""
Simple Point Cloud Transformer for Latent Feature Learning & Regression

This single-file PyTorch implementation:
- Learns point-wise features with a lightweight point-cloud Transformer (kNN graph per layer).
- Produces a compact latent vector for each cloud (usable for downstream regression or retrieval).
- Includes a tiny training loop for a regression task and utilities to extract latents.

Core ideas
- Input: (B, N, 3) xyz points (optionally with extra per-point features; see notes below).
- Local neighborhoods via kNN (computed on-the-fly each block) to keep complexity manageable.
- Relative positional encoding with a small MLP on (x_j - x_i, ||x_j - x_i||).
- Multi-head attention + feed-forward with residuals.
- Global readout by mean + max pooling → latent → regression head (optional).

Usage (quick start)
1) Replace DummyPointCloudDataset with your own dataset that returns (points: [N,3], target: [target_dim]).
2) Run: python this_file.py to execute a small demo training loop (synthetic data by default).
3) Use extract_latents(dataloader, model) to get latent vectors for each cloud.

Tips
- If you have extra per-point features (normals, intensity), you can concat to xyz and set in_channels accordingly.
- For large N, reduce k and/or blocks to fit memory.
- Mixed precision is enabled by default when CUDA is available.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -------------------------------
# Demo main
# -------------------------------

def main_demo():
    # Synthetic data
    train_ds = DummyPointCloudDataset(n_samples=800, n_points=512)
    val_ds = DummyPointCloudDataset(n_samples=200, n_points=512)
    train_loader = DataLoader(train_ds, batch_size=TrainConfig.batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=TrainConfig.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)

    model = PointCloudTransformer(
        in_channels=3,
        d_model=TrainConfig.d_model,
        depth=TrainConfig.depth,
        n_heads=TrainConfig.n_heads,
        k=TrainConfig.k,
        latent_dim=TrainConfig.latent_dim,
        out_dim=TrainConfig.out_dim,
        drop=TrainConfig.drop,
    )

    train_regression(model, train_loader, val_loader, epochs=TrainConfig.epochs, lr=TrainConfig.lr)

    # Extract some latents
    latents = extract_latents(val_loader, model)
    print("Latents shape:", latents.shape)


if __name__ == "__main__":
    main_demo()
