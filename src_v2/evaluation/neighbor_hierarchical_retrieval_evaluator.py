import time


class NeighborHierarchicalRetrievalEvaluator:
    """
    Evaluates NeighborHierarchicalRetriever, which returns a flat list[Chunk]
    (center + neighboring chunks expanded from hierarchical results).
    Because the output is an unordered expanded set, only Hit@K is meaningful —
    there is no ranking to compute MRR against.
    """

    def evaluate(
        self,
        retriever,
        samples,
        k=5
    ):

        hits = 0
        latencies_ms = []

        total = len(samples)

        for sample in samples:

            gt_chunk = (
                sample.metadata[
                    "chunk_id"
                ]
            )

            t0 = time.perf_counter()

            results = (
                retriever.retrieve(
                    sample.question,
                    k=k
                )
            )

            latencies_ms.append(
                (time.perf_counter() - t0) * 1000
            )

            # NeighborHierarchicalRetriever returns list[Chunk] directly
            retrieved = {
                chunk.chunk_id
                for chunk in results
            }

            if gt_chunk in retrieved:
                hits += 1

        return {

            "questions":
                total,

            "child_hit_at_k":
                hits / total,

            "avg_latency_ms":
                sum(latencies_ms) / total
        }
