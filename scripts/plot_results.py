"""Generate the README figure from measured aggregate results."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="reports/results.json")
    parser.add_argument("--output", default="reports/benchmark.png")
    args = parser.parse_args()
    data = json.loads(Path(args.results).read_text())
    names = ["baseline", "no_content"]
    labels = ["Genre Transformer", "ID-only ablation", "Popularity"]
    recall, ne, err, cost = [], [], [], []
    for name in names:
        rows = [r for r in data["runs"] if r["variant"] == name and "test" in r]
        values = [r["test"]["recall@10"] for r in rows]
        recall.append(np.mean(values))
        err.append(np.std(values, ddof=1) if len(rows) > 1 else 0)
        ne.append(np.mean([r["test"]["normalized_entropy"] for r in rows]))
        cost.append(np.mean([r["cost"]["train_wall_seconds"] for r in rows]))
    recall.append(data["popularity"]["test"]["recall@10"])
    ne.append(data["popularity"]["test"]["normalized_entropy"])
    err.append(0)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2), facecolor="#f3f1e9")
    colors = ["#19796d", "#87b6a6", "#bac6c2"]
    for ax in axes:
        ax.set_facecolor("#f3f1e9")
        ax.grid(axis="y", alpha=0.2, zorder=0)
    axes[0].bar(labels, recall, yerr=err, capsize=4, color=colors, zorder=3)
    axes[0].set_title("Test Recall@10 ↑", loc="left", fontweight="bold")
    axes[0].set_ylim(0, max(recall) * 1.4)
    for i, val in enumerate(recall):
        axes[0].text(i, val + err[i] + 0.003, f"{val:.4f}", ha="center")
    axes[1].bar(labels, ne, color=colors, zorder=3)
    axes[1].set_title("Rating-label normalized entropy ↓", loc="left", fontweight="bold")
    axes[1].axhline(1, color="#687c78", ls="--", lw=1)
    axes[1].set_ylim(0, 1.12)
    for i, val in enumerate(ne):
        axes[1].text(i, val + 0.02, f"{val:.4f}", ha="center")
    axes[2].bar(labels[:2], cost, color=colors[:2], zorder=3)
    axes[2].set_title("Training wall seconds / run ↓", loc="left", fontweight="bold")
    axes[2].set_ylim(0, max(cost) * 1.3)
    for i, val in enumerate(cost):
        axes[2].text(i, val + max(cost) * 0.03, f"{val:.2f}s", ha="center")
    for ax in axes:
        ax.tick_params(axis="x", rotation=18)
    fig.suptitle(
        "FrameLab  /  Measured CPU benchmark",
        x=0.055,
        ha="left",
        fontsize=20,
        fontweight="bold",
        color="#183d39",
    )
    fig.text(
        0.055,
        0.02,
        "MovieLens 100K proxy · 3 seeds · recall error bars: seed SD · "
        "agent retained baseline · timings depend on hardware",
        color="#49635b",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.02, 0.08, 0.99, 0.92))
    fig.savefig(args.output, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
