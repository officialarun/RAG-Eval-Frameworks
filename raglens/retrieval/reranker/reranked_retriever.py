from raglens.models import RetrievalResult


class RerankedRetriever:
    """Wraps any retriever (BM25/Dense/Hybrid/Hierarchical/Neighbor) and adds
    a reranking pass: over-retrieve candidate_k candidates from the wrapped
    retriever, score all of them in one batched reranker call, return the
    top-k in the reranker's order.

    Composes exactly like HybridRetriever wraps Dense+BM25 -- satisfies the
    same retrieve(query, k) -> list[RetrievalResult] contract, so it can be
    passed anywhere an existing retriever is used (including
    RetrievalEvaluator, unchanged).
    """

    def __init__(
        self,
        retriever,
        reranker,
        retriever_name: str = "reranked",
        candidate_k: int = 25,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.retriever_name = retriever_name
        self.candidate_k = candidate_k

    def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]:

        candidates = self.retriever.retrieve(query, k=self.candidate_k)

        if not candidates:
            return []

        documents = [candidate.chunk.content for candidate in candidates]

        ranked = self.reranker.rerank(query, documents)

        return [
            RetrievalResult(
                chunk=candidates[index].chunk,
                score=score,
                retriever_name=self.retriever_name,
            )
            for index, score in ranked[:k]
        ]
