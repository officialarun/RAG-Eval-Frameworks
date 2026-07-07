import json
import os

from raglens.evaluation.retrieval_evaluator import RetrievalEvaluator
from raglens.evaluation.hierarchical_retrieval_evaluator import HierarchicalRetrievalEvaluator
from raglens.evaluation.neighbor_hierarchical_retrieval_evaluator import NeighborHierarchicalRetrievalEvaluator


def run_benchmark(
    samples,
    bm25_retriever,
    dense_retriever,
    hybrid_retriever,
    hierarchical_retriever,
    neighbor_retriever,
    k_values=(1, 3, 5, 10),
    cache_path="experiments/data/benchmark_results.json",
):
    """
    Run the full retrieval benchmark across all five strategies and all k values.

    Returns (flat_bench, hier_bench, nbr_bench) where:
      flat_bench[name][k]  = evaluator result dict (BM25 / Dense / Hybrid)
      hier_bench[k]        = hierarchical evaluator result dict
      nbr_bench[k]         = neighbor evaluator result dict

    Also saves JSON to cache_path.
    """
    evaluator = RetrievalEvaluator()
    hier_evaluator = HierarchicalRetrievalEvaluator()
    nbr_evaluator = NeighborHierarchicalRetrievalEvaluator()

    print("Running full benchmark across all retrievers and K values...")
    print(f"Total samples: {len(samples)}\n")

    flat_bench = {}
    for name, retriever in [
        ("BM25", bm25_retriever),
        ("Dense", dense_retriever),
        ("Hybrid", hybrid_retriever),
    ]:
        flat_bench[name] = {}
        for k in k_values:
            print(f"  {name} @ k={k} ...", end="", flush=True)
            flat_bench[name][k] = evaluator.evaluate(retriever, samples, top_k=k)
            m = flat_bench[name][k]
            print(f"  hit={m['hit_at_k']:.3f}  mrr={m['mrr']:.3f}  ndcg={m['ndcg_at_k']:.3f}  lat={m['avg_latency_ms']:.1f}ms")

    hier_bench = {}
    for k in [v for v in k_values if v >= 3]:
        print(f"  Hierarchical @ k={k} ...", end="", flush=True)
        hier_bench[k] = hier_evaluator.evaluate(hierarchical_retriever, samples, k=k)
        m = hier_bench[k]
        print(f"  child_hit={m['child_hit_at_k']:.3f}  section_hit={m['section_hit_at_k']:.3f}  mrr={m['mrr']:.3f}  lat={m['avg_latency_ms']:.1f}ms")

    nbr_bench = {}
    for k in [v for v in k_values if v >= 3]:
        print(f"  Hier+Neighbor @ k={k} ...", end="", flush=True)
        nbr_bench[k] = nbr_evaluator.evaluate(neighbor_retriever, samples, k=k)
        m = nbr_bench[k]
        print(f"  child_hit={m['child_hit_at_k']:.3f}  lat={m['avg_latency_ms']:.1f}ms")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    serialisable = {
        "flat": {name: {str(k): v for k, v in kv.items()} for name, kv in flat_bench.items()},
        "hierarchical": {str(k): v for k, v in hier_bench.items()},
        "neighbor": {str(k): v for k, v in nbr_bench.items()},
    }
    with open(cache_path, "w") as f:
        json.dump(serialisable, f, indent=2)

    print(f"\nBenchmark complete! Results saved to {cache_path}")
    return flat_bench, hier_bench, nbr_bench
