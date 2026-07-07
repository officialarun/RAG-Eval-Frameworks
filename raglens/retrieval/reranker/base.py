from typing import Protocol


class RerankerProvider(Protocol):
    """Minimal reranking contract: text pairs in, ranked (index, score) pairs out.

    Deliberately domain-agnostic -- a reranker never sees RetrievalResult or
    Chunk, only raw strings, matching how embedding/LLM providers stay
    unaware of the domain objects that wrap them (RerankedRetriever is the
    one place that maps back to RetrievalResult, same layering
    EmbeddingGenerator uses for Chunk <-> vector).

    The (index, score) shape -- rather than a parallel list of scores -- is
    deliberate: it matches both a local cross-encoder's predict()-then-sort
    and hosted rerank APIs (Cohere, Jina), which already return
    {index, relevance_score} sorted by relevance. Swapping providers later
    needs no interface change.
    """

    def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        ...
