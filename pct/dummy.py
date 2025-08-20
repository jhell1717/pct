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
# Data: Dummy dataset & simple augmentations
# -------------------------------

class DummyPointCloudDataset(Dataset):
    """Synthetic dataset for quick sanity checks.
    Each sample: N points on a noisy sphere; target is the sphere radius.
    """
    def __init__(self, n_samples=1000, n_points=512, radius_range=(0.5, 1.5)):
        self.n_samples = n_samples
        self.n_points = n_points
        self.radius_range = radius_range

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        N = self.n_points
        r = random.uniform(*self.radius_range)
        # Sample random directions
        phi = torch.rand(N) * 2 * math.pi
        costheta = torch.rand(N) * 2 - 1
        theta = torch.acos(costheta)
        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(theta)
        pts = torch.stack([x, y, z], dim=-1) * r
        # Add small noise
        pts += 0.02 * torch.randn_like(pts)
        target = torch.tensor([r], dtype=torch.float32)
        return pts.float(), target


def collate_batch(batch):
    pts, y = zip(*batch)
    pts = torch.stack(pts, dim=0)  # (B, N, 3)
    y = torch.stack(y, dim=0)      # (B, 1)
    return pts, y