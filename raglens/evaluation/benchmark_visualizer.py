import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

PALETTE = {
    "BM25": "#e74c3c",
    "Dense": "#3498db",
    "Hybrid": "#9b59b6",
    "Hierarchical": "#27ae60",
    "Hier+Neighbor": "#f39c12",
}


def build_comparison_table(flat_bench, hier_bench, nbr_bench, k_values=(1, 3, 5, 10)):
    """Build a colour-coded pandas Styler comparing all retrievers across all K values."""
    rows = []

    for name in ["BM25", "Dense", "Hybrid"]:
        for k in k_values:
            m = flat_bench[name][k]
            rows.append({
                "Retriever": name,
                "Type": "Flat / Lexical-Semantic",
                "K": k,
                "Hit@K": m["hit_at_k"],
                "MRR": m["mrr"],
                "NDCG@K": m["ndcg_at_k"],
                "Section Hit": None,
                "Sect+Child": None,
                "Latency (ms)": m["avg_latency_ms"],
            })

    for k in [v for v in k_values if v >= 3]:
        m = hier_bench[k]
        rows.append({
            "Retriever": "Hierarchical",
            "Type": "Hierarchical",
            "K": k,
            "Hit@K": m["child_hit_at_k"],
            "MRR": m["mrr"],
            "NDCG@K": None,
            "Section Hit": m["section_hit_at_k"],
            "Sect+Child": m["section_and_child_hit"],
            "Latency (ms)": m["avg_latency_ms"],
        })

    for k in [v for v in k_values if v >= 3]:
        m = nbr_bench[k]
        rows.append({
            "Retriever": "Hier+Neighbor",
            "Type": "Hierarchical+Context",
            "K": k,
            "Hit@K": m["child_hit_at_k"],
            "MRR": None,
            "NDCG@K": None,
            "Section Hit": None,
            "Sect+Child": None,
            "Latency (ms)": m["avg_latency_ms"],
        })

    df = pd.DataFrame(rows).set_index(["Retriever", "Type", "K"])

    def fmt(v, decimals=4):
        return f"{v:.{decimals}f}" if v is not None else "—"

    styled = (
        df.style
        .format({
            "Hit@K":        lambda v: fmt(v),
            "MRR":          lambda v: fmt(v),
            "NDCG@K":       lambda v: fmt(v),
            "Section Hit":  lambda v: fmt(v),
            "Sect+Child":   lambda v: fmt(v),
            "Latency (ms)": lambda v: fmt(v, 1),
        })
        .background_gradient(cmap="YlGn",    subset=["Hit@K"],       vmin=0, vmax=1)
        .background_gradient(cmap="Blues",   subset=["MRR"],          vmin=0, vmax=1)
        .background_gradient(cmap="Purples", subset=["NDCG@K"],       vmin=0, vmax=1)
        .background_gradient(cmap="Greens",  subset=["Section Hit"],  vmin=0, vmax=1)
        .background_gradient(cmap="Oranges", subset=["Sect+Child"],   vmin=0, vmax=1)
        .set_caption("Full Retrieval Benchmark — All Metrics Across All K Values")
        .set_table_styles([
            {"selector": "caption",
             "props": [("font-size", "15px"), ("font-weight", "bold"),
                       ("color", "#2c3e50"), ("padding", "10px")]},
            {"selector": "th",
             "props": [("background-color", "#2c3e50"), ("color", "white"),
                       ("font-size", "12px"), ("padding", "8px")]},
            {"selector": "td",
             "props": [("padding", "6px 10px"), ("font-size", "12px")]},
        ])
    )
    return styled


def plot_benchmark_chart(
    flat_bench,
    hier_bench,
    nbr_bench,
    k_values=(1, 3, 5, 10),
    save_path="experiments/data/benchmark_chart.png",
):
    """Render the 4-panel benchmark comparison figure and save it to save_path."""
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "Retrieval Benchmark: Comprehensive Comparison",
        fontsize=17, fontweight="bold", y=1.01,
    )

    hier_k = [v for v in k_values if v >= 3]

    # ── Panel 1: Hit@K line chart ──────────────────────────────────────────────
    ax1 = fig.add_subplot(2, 3, (1, 2))
    for name in ["BM25", "Dense", "Hybrid"]:
        hits = [flat_bench[name][k]["hit_at_k"] for k in k_values]
        ax1.plot(k_values, hits, marker="o", label=name,
                 color=PALETTE[name], linewidth=2.2, markersize=8)
    ax1.plot(hier_k, [hier_bench[k]["child_hit_at_k"] for k in hier_k],
             marker="s", linestyle="--", label="Hierarchical",
             color=PALETTE["Hierarchical"], linewidth=2.2, markersize=8)
    ax1.plot(hier_k, [nbr_bench[k]["child_hit_at_k"] for k in hier_k],
             marker="^", linestyle=":", label="Hier+Neighbor",
             color=PALETTE["Hier+Neighbor"], linewidth=2.2, markersize=9)

    for name, bench, key, offset in [
        ("Hybrid",        flat_bench["Hybrid"], "hit_at_k",       +0.01),
        ("Hierarchical",  {5: hier_bench[5]},   "child_hit_at_k", -0.03),
        ("Hier+Neighbor", {5: nbr_bench[5]},    "child_hit_at_k", +0.01),
    ]:
        v = bench[5][key]
        ax1.annotate(f"{v:.1%}", xy=(5, v), xytext=(5.2, v + offset),
                     fontsize=8.5, color=PALETTE[name], fontweight="bold")

    ax1.set_title("Hit@K Across K Values  (↗ = better retrieval coverage)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("K  (number of results retrieved)")
    ax1.set_ylabel("Hit@K  (fraction of queries answered)")
    ax1.set_xticks(k_values)
    ax1.legend(fontsize=9, loc="lower right")
    ax1.grid(True, alpha=0.25)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax1.set_ylim(0.4, 1.05)

    # ── Panel 2: MRR & NDCG@5 ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(2, 3, 3)
    names_p2 = ["BM25", "Dense", "Hybrid", "Hierarchical"]
    mrr_vals  = [flat_bench["BM25"][5]["mrr"], flat_bench["Dense"][5]["mrr"],
                 flat_bench["Hybrid"][5]["mrr"], hier_bench[5]["mrr"]]
    ndcg_vals = [flat_bench["BM25"][5]["ndcg_at_k"], flat_bench["Dense"][5]["ndcg_at_k"],
                 flat_bench["Hybrid"][5]["ndcg_at_k"], None]

    x2    = np.arange(len(names_p2))
    width = 0.38
    bars1 = ax2.bar(x2 - width / 2, mrr_vals, width,
                    label="MRR@5", color=[PALETTE[n] for n in names_p2], alpha=0.90)
    bars2 = ax2.bar(x2 + width / 2, [v if v is not None else 0 for v in ndcg_vals], width,
                    label="NDCG@5", color=[PALETTE[n] for n in names_p2], alpha=0.55, hatch="//")

    for bar in bars1:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, ndcg_vals):
        if val is not None:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax2.set_title("Rank Quality @ k=5\n(MRR vs NDCG — solid vs hatched)", fontsize=11, fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(names_p2, fontsize=9)
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("Score")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25, axis="y")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    # ── Panel 3: Hierarchical breakdown stacked bars ───────────────────────────
    ax3 = fig.add_subplot(2, 3, (4, 5))
    h5 = hier_bench[5]
    n5 = nbr_bench[5]
    categories = ["Hierarchical\n@ k=5", "Hier+Neighbor\n@ k=5"]
    sect_child = [h5["section_and_child_hit"], n5["child_hit_at_k"]]
    sect_only  = [h5["section_only_hit"], 0]
    miss_vals  = [h5["complete_miss"], 1 - n5["child_hit_at_k"]]

    x3, w3 = np.arange(len(categories)), 0.45
    b1 = ax3.bar(x3, sect_child, w3, label="Section + Exact Chunk Hit ✓", color="#27ae60")
    b2 = ax3.bar(x3, sect_only, w3, bottom=sect_child,
                 label="Section Hit only (chunk missed)", color="#f39c12")
    bottom2 = [s + o for s, o in zip(sect_child, sect_only)]
    b3 = ax3.bar(x3, miss_vals, w3, bottom=bottom2, label="Complete Miss ✗", color="#e74c3c", alpha=0.75)

    for bar, val in zip(b1, sect_child):
        if val > 0.03:
            ax3.text(bar.get_x() + bar.get_width() / 2, val / 2,
                     f"{val:.1%}", ha="center", va="center", fontsize=13, color="white", fontweight="bold")
    for bar, val, bot in zip(b2, sect_only, sect_child):
        if val > 0.03:
            ax3.text(bar.get_x() + bar.get_width() / 2, bot + val / 2,
                     f"{val:.1%}", ha="center", va="center", fontsize=11, color="white", fontweight="bold")
    for bar, val, bot in zip(b3, miss_vals, bottom2):
        if val > 0.01:
            ax3.text(bar.get_x() + bar.get_width() / 2, bot + val / 2,
                     f"{val:.1%}", ha="center", va="center", fontsize=11, color="white", fontweight="bold")

    ax3.annotate(
        "Neighbor expansion converts\n'Section Only' misses into full hits",
        xy=(0.85, sect_child[1] - 0.03), xytext=(0.5, 0.65),
        fontsize=9.5, color="#444",
        arrowprops=dict(arrowstyle="->", color="#444", lw=1.3), ha="center",
    )
    ax3.set_title(
        "Hierarchical Breakdown @ k=5\nHow Neighbor Expansion Recovers Missed Chunks",
        fontsize=11, fontweight="bold",
    )
    ax3.set_xticks(x3)
    ax3.set_xticklabels(categories, fontsize=12)
    ax3.set_ylabel("Fraction of Queries")
    ax3.set_ylim(0, 1.08)
    ax3.legend(fontsize=9, loc="upper right")
    ax3.grid(True, alpha=0.25, axis="y")
    ax3.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    # ── Panel 4: Latency bars ──────────────────────────────────────────────────
    ax4 = fig.add_subplot(2, 3, 6)
    lat_names = ["BM25", "Dense", "Hybrid", "Hierarchical", "Hier+Neighbor"]
    lats = [
        flat_bench["BM25"][5]["avg_latency_ms"],
        flat_bench["Dense"][5]["avg_latency_ms"],
        flat_bench["Hybrid"][5]["avg_latency_ms"],
        hier_bench[5]["avg_latency_ms"],
        nbr_bench[5]["avg_latency_ms"],
    ]
    bars4 = ax4.bar(lat_names, lats, color=[PALETTE[n] for n in lat_names],
                    alpha=0.88, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars4, lats):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(lats) * 0.01,
                 f"{val:.1f}ms", ha="center", va="bottom", fontsize=9)
    ax4.set_title("Average Retrieval Latency @ k=5\n(lower = faster)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Latency (ms)")
    ax4.set_xticklabels(lat_names, rotation=15, ha="right", fontsize=9)
    ax4.grid(True, alpha=0.25, axis="y")

    plt.tight_layout(pad=2.5)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Chart saved to {save_path}")

    return fig
