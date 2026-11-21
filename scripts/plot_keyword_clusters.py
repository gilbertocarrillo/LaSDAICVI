# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "pandas",
#   "librosa",
#   "scikit-learn",
#   "matplotlib",
# ]
# ///
"""Project LaSDAICVI samples to 2D and draw one cloud per keyword.

Each sample is summarized by its time-normalized MFCC trajectory, reduced with PCA and embedded
with t-SNE. Keywords that the children pronounce consistently form compact,
separate clouds.

Usage (from the repository root):
    uv run scripts/plot_keyword_clusters.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_RATE = 16_000
N_MFCC = 20
N_FRAMES = 24

DIGITS = [str(i) for i in range(10)]
COMMANDS = [
    "abajo", "actívate", "apágate", "aprender", "arriba", "atrás", "continúa",
    "derecha", "detente", "falso", "izquierda", "jugar", "no", "pepe",
    "repite", "si", "siguiente", "verdadero",
]
PRODUCTS = [
    "10", "12", "14", "15", "16", "18", "20", "21", "24", "25", "27", "28", "30", "32", "35", "36",
    "40", "42", "45", "48", "49", "50", "54", "56", "60", "63", "64", "70", "72", "80", "81", "90",
]


def features(path: Path) -> np.ndarray:
    """MFCC trajectory of the trimmed word, resampled to a fixed number of frames.

    Keeping the time axis matters: "arriba" and "abajo" share most of their
    sounds and differ mainly in their order.
    """
    audio, _ = librosa.load(path, sr=SAMPLE_RATE)
    audio, _ = librosa.effects.trim(audio, top_db=30)
    mfcc = librosa.feature.mfcc(
        y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=512, hop_length=128, n_mels=40
    )[1:]  # drop c0 (loudness)
    mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-6)
    grid = np.linspace(0, mfcc.shape[1] - 1, N_FRAMES)
    resampled = np.stack([np.interp(grid, np.arange(mfcc.shape[1]), row) for row in mfcc])
    return resampled.ravel()


def embed(frame: pd.DataFrame, seed: int) -> np.ndarray:
    matrix = np.stack([features(ROOT / "LaSDAICVI" / p) for p in frame["path"]])
    matrix = StandardScaler().fit_transform(matrix)
    matrix = PCA(n_components=30, random_state=seed).fit_transform(matrix)
    return TSNE(
        n_components=2, perplexity=40, init="pca", random_state=seed
    ).fit_transform(matrix)


def draw(ax: plt.Axes, points: np.ndarray, labels: pd.Series, title: str) -> None:
    keywords = sorted(dict.fromkeys(labels), key=lambda k: (not k.isdigit(), int(k) if k.isdigit() else 0, k))
    colors = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)
    for index, keyword in enumerate(keywords):
        mask = (labels == keyword).to_numpy()
        color = colors[index % len(colors)]
        ax.scatter(points[mask, 0], points[mask, 1], s=11, alpha=0.65, color=color, linewidths=0)
        cx, cy = np.median(points[mask], axis=0)
        ax.text(
            cx, cy, keyword, ha="center", va="center", fontsize=10, fontweight="bold",
            color="#141925",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, lw=1.2, alpha=0.9),
        )
    ax.set_title(title, fontsize=14, fontweight="bold", color="#141925", loc="left")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-keyword", type=int, default=150, help="samples drawn per keyword")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=ROOT / "assets" / "keyword-clusters.png")
    args = parser.parse_args()

    metadata = pd.read_csv(ROOT / "lasdaicvi.csv")
    groups = [
        ("Digits 0–9", DIGITS),
        ("Voice commands", COMMANDS),
        ("Multiplication-table results (tables 2–9)", PRODUCTS),
    ]

    fig = plt.figure(figsize=(16, 16), facecolor="white")
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 1.05))
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])]
    for ax, (title, keywords) in zip(axes, groups):
        subset = (
            metadata[metadata["label"].astype(str).isin(keywords)]
            .groupby("label")
            .sample(n=args.per_keyword, random_state=args.seed)
            .reset_index(drop=True)
        )
        points = embed(subset, args.seed)
        draw(ax, points, subset["label"].astype(str), title)
        print(f"{title}: {len(subset)} samples")

    fig.suptitle(
        "LaSDAICVI keywords projected with t-SNE (time-normalized MFCC)",
        fontsize=16, fontweight="bold", color="#141925", x=0.02, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"saved {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
