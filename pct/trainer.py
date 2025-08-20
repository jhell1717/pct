from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .model import PointCloudTransformer


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 16
    lr: float = 3e-4
    d_model: int = 128
    depth: int = 4
    n_heads: int = 4
    k: int = 16
    latent_dim: int = 256
    out_dim: int = 1
    drop: float = 0.0


def train_regression(model: PointCloudTransformer, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
                     epochs: int = 10, lr: float = 3e-4, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.startswith('cuda')))
    best_val = float('inf')

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xyz, y in train_loader:
            xyz = xyz.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.startswith('cuda'))):
                pred = model(xyz)
                loss = F.mse_loss(pred, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * xyz.size(0)
        train_loss = running / len(train_loader.dataset)

        if val_loader is not None:
            model.eval()
            vloss = 0.0
            with torch.no_grad():
                for xyz, y in val_loader:
                    xyz = xyz.to(device)
                    y = y.to(device)
                    with torch.cuda.amp.autocast(enabled=(device.startswith('cuda'))):
                        pred = model(xyz)
                        loss = F.mse_loss(pred, y)
                    vloss += loss.item() * xyz.size(0)
            vloss /= len(val_loader.dataset)
            best_val = min(best_val, vloss)
            print(f"Epoch {ep:03d} | train {train_loss:.4f} | val {vloss:.4f}")
        else:
            print(f"Epoch {ep:03d} | train {train_loss:.4f}")


@torch.no_grad()
def extract_latents(dataloader: DataLoader, model: PointCloudTransformer,
                    device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> torch.Tensor:
    model.eval().to(device)
    latents = []
    for xyz, _ in dataloader:
        xyz = xyz.to(device)
        _, latent = model(xyz, return_latent=True)
        latents.append(latent.cpu())
    return torch.cat(latents, dim=0)  # (num_samples, latent_dim)