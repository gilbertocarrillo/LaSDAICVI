# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "pandas",
#   "librosa",
#   "scikit-learn",
#   "matplotlib",
#   "pillow",
# ]
# ///
"""Render a rotating 3D t-SNE of the LaSDAICVI voice commands as a GIF.

Uses the same features as plot_keyword_clusters.py.

Usage (from the repository root):
    uv run scripts/plot_keyword_clusters_3d.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_keyword_clusters import COMMANDS, ROOT, features  # noqa: E402

PALETTE = [
    "#2F5FE0", "#FF7A6B", "#17935F", "#F2A900", "#8E44AD", "#00A6B4", "#E0457B",
    "#6B8E23", "#FF8C1A", "#3B2A20", "#5B8CFF", "#C0392B", "#16A085", "#7F8C8D",
    "#D35400", "#2C3E50", "#9B59B6", "#1ABC9C",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-keyword", type=int, default=150, help="samples drawn per keyword")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=ROOT / "assets" / "keyword-clusters-3d.gif")
    args = parser.parse_args()

    metadata = pd.read_csv(ROOT / "lasdaicvi.csv")
    subset = (
        metadata[metadata["label"].isin(COMMANDS)]
        .groupby("label")
        .sample(n=args.per_keyword, random_state=args.seed)
        .reset_index(drop=True)
    )
    matrix = np.stack([features(ROOT / "LaSDAICVI" / p) for p in subset["path"]])
    matrix = StandardScaler().fit_transform(matrix)
    matrix = PCA(n_components=30, random_state=args.seed).fit_transform(matrix)
    points = TSNE(
        n_components=3, perplexity=40, init="pca", random_state=args.seed
    ).fit_transform(matrix)
    labels = subset["label"]

    fig = plt.figure(figsize=(9, 6.4), facecolor="white")
    ax = fig.add_axes([0.0, 0.0, 0.78, 0.92], projection="3d")
    for index, keyword in enumerate(sorted(labels.unique())):
        mask = (labels == keyword).to_numpy()
        ax.scatter(
            points[mask, 0], points[mask, 1], points[mask, 2], s=9, alpha=0.7,
            color=PALETTE[index % len(PALETTE)], depthshade=False, label=keyword,
        )
    low, high = np.percentile(points, [1, 99], axis=0)
    ax.set_xlim(low[0], high[0])
    ax.set_ylim(low[1], high[1])
    ax.set_zlim(low[2], high[2])
    ax.set_box_aspect(None, zoom=1.25)
    ax.set_axis_off()
    fig.legend(
        *ax.get_legend_handles_labels(), loc="center left", bbox_to_anchor=(0.78, 0.47),
        frameon=False, fontsize=11, markerscale=2.2, handletextpad=0.3, labelspacing=0.45,
    )
    fig.text(0.03, 0.94, "LaSDAICVI · Voice commands", fontsize=16, fontweight="bold", color="#141925")
    fig.text(
        0.03, 0.905, f"3D t-SNE of time-normalized MFCC · {args.per_keyword} samples per keyword",
        fontsize=10.5, color="#586278",
    )

    def rotate(frame: int):
        ax.view_init(elev=20, azim=frame * 4)
        return (ax,)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    FuncAnimation(fig, rotate, frames=90, interval=80).save(
        args.out, writer=PillowWriter(fps=12), dpi=80
    )
    print(f"saved {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
