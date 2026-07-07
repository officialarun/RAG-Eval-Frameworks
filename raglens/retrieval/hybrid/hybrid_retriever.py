from collections import defaultdict

from raglens.models import RetrievalResult
from raglens.retrieval.dense import DenseRetriever
from raglens.retrieval.bm25 import BM25Retriever


class HybridRetriever:
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        dense_weight: float = 0.5,
        bm25_weight: float = 0.5,
    ):
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

    def _normalize_scores(self, results):
        if not results:
            return {}

        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return {r.chunk.chunk_id: 1.0 for r in results}

        normalized = {}
        for r in results:
            normalized[r.chunk.chunk_id] = (r.score - min_score) / (max_score - min_score)

        return normalized

    def retrieve(self, query: str, k: int = 5):
        candidate_k = max(20, k * 3)

        dense_results = self.dense_retriever.retrieve(query=query, k=candidate_k)
        bm25_results = self.bm25_retriever.retrieve(query=query, k=candidate_k)

        dense_scores = self._normalize_scores(dense_results)
        bm25_scores = self._normalize_scores(bm25_results)

        all_chunks = {}
        for result in dense_results:
            all_chunks[result.chunk.chunk_id] = result.chunk
        for result in bm25_results:
            all_chunks[result.chunk.chunk_id] = result.chunk

        fused_scores = defaultdict(float)
        for chunk_id, score in dense_scores.items():
            fused_scores[chunk_id] += self.dense_weight * score
        for chunk_id, score in bm25_scores.items():
            fused_scores[chunk_id] += self.bm25_weight * score

        ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for chunk_id, score in ranked[:k]:
            results.append(
                RetrievalResult(chunk=all_chunks[chunk_id], score=float(score), retriever_name="hybrid")
            )

        return results
