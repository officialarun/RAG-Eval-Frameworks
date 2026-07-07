import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def plot_ragas_scores(
    comparison: pd.DataFrame,
    save_path: str = "data/cache/ragas_benchmark_chart.png",
    title: str = "RAGAS Scores by Retriever",
):
    """Render the per-metric bar chart comparing retrievers and save it to save_path."""

    metrics = list(comparison.columns)
    retrievers = list(comparison.index)
    x = np.arange(len(retrievers))

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 5), sharey=False)
    if len(metrics) == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for ax, metric, color in zip(axes, metrics, _COLORS):
        vals = comparison[metric].values.astype(float)
        bars = ax.bar(x, vals, color=color, alpha=0.82, edgecolor="white", linewidth=0.8)
        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(retrievers, rotation=35, ha="right", fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        ax.axhline(vals.mean(), color="gray", linestyle="--", linewidth=0.9, label="mean")
        ax.legend(fontsize=8)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
            )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
