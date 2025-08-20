# 🌀 PCT: Point Cloud Transformer

`pct` is a simple PyTorch-based package for learning latent representations from 3D point clouds.  
It includes a minimal **Point Cloud Transformer (PCT)** model, dataset utilities, and training scripts.  
The latent embeddings can be used for regression or other downstream tasks.

---

## 🚀 Features
- Dummy point cloud dataset generator (noisy spheres with variable radius).
- Collate functions for batching point clouds.
- Transformer-based point cloud encoder with regression head.
- Training utilities with PyTorch `DataLoader`.
- 3D visualization of point clouds using Plotly.

---

## 📦 Installation

Clone the repo and install locally:

```bash
git clone https://github.com/your-username/pct.git
cd pct
pip install .
``` 

## Citations
https://arxiv.org/pdf/2012.09688
