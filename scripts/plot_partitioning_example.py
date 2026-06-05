#!/usr/bin/env python3
"""Generate an illustrative figure comparing grid partitioning and
clustering-based region partitioning. Saves to reports/figures/partitioning_example.png
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def make_points(n=500, seed=0):
    rng = np.random.RandomState(seed)
    # Create a few cluster centers to make clustering visible
    centers = np.array([[2, 2], [7, 6], [6, 2], [3, 7]])
    pts = []
    for i, c in enumerate(centers):
        pts.append(c + 0.8 * rng.randn(n // len(centers), 2))
    pts = np.vstack(pts)
    # add some uniformly distributed background points
    pts = np.vstack([pts, rng.rand(100, 2) * 10 * 0.95 + 0.25])
    return pts


def plot_grid(ax, points, grid_size=(6, 6), bbox=(0, 10, 0, 10)):
    xmin, xmax, ymin, ymax = bbox
    nx, ny = grid_size
    xs = np.linspace(xmin, xmax, nx + 1)
    ys = np.linspace(ymin, ymax, ny + 1)

    # color points by grid cell index
    ix = np.searchsorted(xs, points[:, 0], side='right') - 1
    iy = np.searchsorted(ys, points[:, 1], side='right') - 1
    cell_id = (iy.clip(0, ny - 1) * nx) + ix.clip(0, nx - 1)

    ax.scatter(points[:, 0], points[:, 1], c=cell_id, cmap='tab20', s=15, alpha=0.85)
    for x in xs:
        ax.plot([x, x], [ymin, ymax], color='grey', lw=0.8, alpha=0.6)
    for y in ys:
        ax.plot([xmin, xmax], [y, y], color='grey', lw=0.8, alpha=0.6)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title('Uniform Grid Partitioning')
    ax.set_xticks([])
    ax.set_yticks([])


def plot_clustering(ax, points, n_clusters=8, seed=0):
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    labels = kmeans.fit_predict(points)
    ax.scatter(points[:, 0], points[:, 1], c=labels, cmap='tab10', s=15, alpha=0.9)
    centers = kmeans.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], marker='x', c='k', s=50, linewidths=2)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title('Clustering-based Partitioning')
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    out = Path('reports/figures')
    out.mkdir(parents=True, exist_ok=True)
    pts = make_points(n=500, seed=42)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    plot_grid(axs[0], pts, grid_size=(6, 6), bbox=(0, 10, 0, 10))
    plot_clustering(axs[1], pts, n_clusters=8, seed=42)

    # Add a shared annotation
    fig.suptitle('Region Partitioning: Grid vs Clustering', fontsize=14, fontweight='bold')

    save_path = out / 'partitioning_example.png'
    plt.savefig(save_path, dpi=200)
    print(f"Saved figure to {save_path}")


if __name__ == '__main__':
    main()
