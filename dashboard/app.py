import json
from pathlib import Path

import pandas as pd
import streamlit as st

from raglens.evaluation import build_comparison_table, plot_benchmark_chart
from raglens.ragas import RagasScorer, plot_ragas_scores

DEMO_DATA_DIR = Path(__file__).parent / "demo_data"
DEMO_BENCHMARK_PATH = DEMO_DATA_DIR / "benchmark_results.json"
DEMO_SCORES_PATH = DEMO_DATA_DIR / "ragas_scores.jsonl"


def _resolve(configured_path: str, demo_path: Path):
    """Use the configured local path if it exists; otherwise fall back to the
    bundled reference results so a hosted/public deployment (no local corpus,
    no API keys) still has something real to show. Returns (path, is_demo)."""
    path = Path(configured_path)
    if path.exists():
        return path, False
    if demo_path.exists():
        return demo_path, True
    return path, False


st.set_page_config(page_title="raglens", layout="wide")

st.title("raglens")
st.caption(
    "Retrieval + RAGAS benchmarking dashboard. Runs are produced by the "
    "`raglens` CLI (ingest / index / benchmark / evaluate) -- this page "
    "only visualizes whatever those runs have already written to disk. "
    "If no local run is found, it falls back to bundled reference results "
    "from this project's own corpus (see README)."
)

with st.sidebar:
    st.header("Data sources")
    benchmark_path = st.text_input(
        "Retrieval benchmark JSON", value="experiments/experiments/data/benchmark_results.json"
    )
    scores_path = st.text_input(
        "RAGAS scores JSONL", value="experiments/data/cache/ragas_scores.jsonl"
    )
    retrievers = st.multiselect(
        "Retrievers",
        ["bm25", "dense", "hybrid", "hierarchical", "neighbor"],
        default=["bm25", "dense", "hybrid", "hierarchical", "neighbor"],
    )
    st.caption(
        "RAGAS scoring is judge-LLM-bound and can take hours for a few "
        "hundred samples -- this dashboard never triggers a run itself. "
        "Use `raglens evaluate` for that, ideally left running unattended."
    )

tab_retrieval, tab_ragas = st.tabs(["Retrieval Benchmark", "RAGAS Evaluation"])

with tab_retrieval:
    path, is_demo = _resolve(benchmark_path, DEMO_BENCHMARK_PATH)
    if not path.exists():
        st.info(f"No benchmark file at {path}. Run `raglens benchmark` first.")
    else:
        if is_demo:
            st.caption("Showing bundled reference results (no local run found at the configured path).")
        data = json.loads(path.read_text())
        flat_bench, hier_bench, nbr_bench = data["flat"], data["hierarchical"], data["neighbor"]
        hier_bench = {int(k): v for k, v in hier_bench.items()}
        nbr_bench = {int(k): v for k, v in nbr_bench.items()}
        for name in flat_bench:
            flat_bench[name] = {int(k): v for k, v in flat_bench[name].items()}

        k_values = sorted(next(iter(flat_bench.values())).keys())

        styled = build_comparison_table(flat_bench, hier_bench, nbr_bench, k_values=k_values)
        st.dataframe(styled, width="stretch")

        fig = plot_benchmark_chart(flat_bench, hier_bench, nbr_bench, k_values=k_values)
        st.pyplot(fig)

with tab_ragas:
    path, is_demo = _resolve(scores_path, DEMO_SCORES_PATH)
    if not path.exists():
        st.info(f"No RAGAS scores at {path}. Run `raglens evaluate` first.")
    else:
        if is_demo:
            st.caption("Showing bundled reference results (no local run found at the configured path).")
        scorer = RagasScorer(scores_path=str(path))
        results = scorer.aggregate(retrievers)
        comparison = scorer.build_comparison_table(results)

        st.subheader("Status")
        raw = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        counts = pd.Series([r["retriever"] for r in raw]).value_counts()
        status_cols = st.columns(len(retrievers))
        for col, name in zip(status_cols, retrievers):
            col.metric(name, f"{counts.get(name, 0)} scored")

        st.subheader("Mean scores")
        st.dataframe(comparison, width="stretch")

        scored_retrievers = [r for r in retrievers if comparison.loc[r].notna().any()]
        if scored_retrievers:
            fig = plot_ragas_scores(comparison.loc[scored_retrievers])
            st.pyplot(fig)
        else:
            st.info("No retriever has any scored samples yet.")
