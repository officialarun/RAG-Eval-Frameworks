import time

from src_v2.evaluation import (
    hit_at_k,
    reciprocal_rank,
    ndcg_at_k
)


class RetrievalEvaluator:

    def evaluate(
        self,
        retriever,
        samples,
        top_k=5
    ):

        hit_scores = []
        rr_scores = []
        ndcg_scores = []
        latencies_ms = []

        total = len(
            samples
        )

        for index, sample in enumerate(
            samples,
            start=1
        ):

            t0 = time.perf_counter()

            results = retriever.retrieve(
                sample.question,
                k=top_k
            )

            latencies_ms.append(
                (time.perf_counter() - t0) * 1000
            )

            retrieved_chunk_ids = [

                result.chunk.chunk_id

                for result
                in results
            ]

            ground_truth = (
                sample.metadata[
                    "chunk_id"
                ]
            )

            hit_scores.append(

                hit_at_k(
                    ground_truth,
                    retrieved_chunk_ids
                )
            )

            rr_scores.append(

                reciprocal_rank(
                    ground_truth,
                    retrieved_chunk_ids
                )
            )

            ndcg_scores.append(

                ndcg_at_k(
                    ground_truth,
                    retrieved_chunk_ids
                )
            )

            if index % 50 == 0:

                print(
                    f"{index}/{total}"
                )

        return {

            "questions":
                total,

            "hit_at_k":
                sum(hit_scores)
                / total,

            "mrr":
                sum(rr_scores)
                / total,

            "ndcg_at_k":
                sum(ndcg_scores)
                / total,

            "avg_latency_ms":
                sum(latencies_ms)
                / total
        }
