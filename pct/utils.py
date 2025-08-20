from __future__ import annotations

import plotly.graph_objects as go

import torch


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points/features by index.
    points: (B, N, C)
    idx:    (B, M, K) or (B, M)
    returns: (B, M, K, C) or (B, M, C)
    """
    B = points.size(0)
    batch_indices = torch.arange(B, device=points.device).view(B, 1, 1)
    if idx.dim() == 3:
        B, M, K = idx.shape
        batch_indices = batch_indices.expand(-1, M, K)
        return points[batch_indices, idx]
    else:
        B, M = idx.shape
        batch_indices = batch_indices.expand(-1, M)
        return points[batch_indices, idx]
    
def knn_indices(xyz: torch.Tensor, k: int) -> torch.Tensor:
    """Compute kNN indices within each batch using pairwise distance.
    xyz: (B, N, 3)
    returns idx: (B, N, k) of neighbor indices for each point.
    Note: includes self if distances tie; we explicitly drop self by taking topk on negative distances.
    """
    B, N, _ = xyz.shape
    # (B, N, N) pairwise distances (squared) – avoid sqrt for speed.
    with torch.no_grad():
        dist2 = torch.cdist(xyz, xyz, p=2)  # (B, N, N)
        # mask self distance to +inf so it won't be picked
        eye = torch.eye(N, device=xyz.device).unsqueeze(0)
        dist2 = dist2 + eye * 1e6
        _, idx = torch.topk(dist2, k, dim=-1, largest=False, sorted=False)  # smaller distance = nearer
    return idx  # (B, N, k)



def visualize_point_cloud(points: torch.Tensor, title: str = "Point Cloud"):
    """
    Visualize a point cloud in 3D using Plotly.

    Args:
        points (torch.Tensor): Shape (N, 3) tensor of xyz coordinates.
        title (str): Optional plot title.
    """
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()

    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=3,
            color=z,        # Color by z-coordinate for variety
            colorscale='Viridis',
            opacity=0.8
        )
    )])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    fig.show()
