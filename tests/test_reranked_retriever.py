from raglens.models import Chunk, RetrievalResult
from raglens.retrieval.reranker import RerankedRetriever, get_reranker_provider


def _chunk(chunk_id, content):
    return Chunk(
        chunk_id=chunk_id,
        parent_doc_id="doc_0",
        parent_section_id="s0",
        chunk_order=0,
        fragment_index=0,
        section_title="Intro",
        path="doc_0 > Intro",
        level=1,
        chunk_type="section_fragment",
        content=content,
    )


class FakeRetriever:
    """Returns a fixed candidate list, ignoring the query -- lets tests
    control exactly what RerankedRetriever has to work with."""

    def __init__(self, chunk_ids):
        self.chunk_ids = chunk_ids

    def retrieve(self, query, k=5):
        return [
            RetrievalResult(chunk=_chunk(cid, f"content-{cid}"), score=1.0, retriever_name="fake")
            for cid in self.chunk_ids[:k]
        ]


class FakeReranker:
    """Returns (index, score) pairs in a caller-specified order, so tests
    can assert the final output follows the *reranker's* ranking."""

    def __init__(self, order):
        self.order = order  # list of indices, already in desired rank order

    def rerank(self, query, documents):
        return [(i, float(len(self.order) - rank)) for rank, i in enumerate(self.order)]


def test_reranked_retriever_uses_reranker_order_not_wrapped_retriever_order():
    retriever = FakeRetriever(["c1", "c2", "c3", "c4", "c5"])
    reranker = FakeReranker(order=[4, 0, 2, 1, 3])  # c5 should come first
    reranked = RerankedRetriever(retriever, reranker, candidate_k=5)

    results = reranked.retrieve("query", k=3)

    assert [r.chunk.chunk_id for r in results] == ["c5", "c1", "c3"]


def test_reranked_retriever_truncates_to_k():
    retriever = FakeRetriever(["c1", "c2", "c3"])
    reranker = FakeReranker(order=[0, 1, 2])
    reranked = RerankedRetriever(retriever, reranker, candidate_k=3)

    results = reranked.retrieve("query", k=1)

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "c1"


def test_reranked_retriever_uses_candidate_k_not_final_k():
    retriever = FakeRetriever([f"c{i}" for i in range(10)])
    reranker = FakeReranker(order=list(range(4)))
    reranked = RerankedRetriever(retriever, reranker, candidate_k=4)

    results = reranked.retrieve("query", k=2)

    # FakeRetriever only ever sees k=candidate_k=4, never the final k=2
    assert len(results) == 2
    assert [r.chunk.chunk_id for r in results] == ["c0", "c1"]


def test_reranked_retriever_sets_retriever_name():
    retriever = FakeRetriever(["c1", "c2"])
    reranker = FakeReranker(order=[0, 1])
    reranked = RerankedRetriever(retriever, reranker, retriever_name="hybrid_reranked", candidate_k=2)

    results = reranked.retrieve("query", k=2)

    assert all(r.retriever_name == "hybrid_reranked" for r in results)


def test_reranked_retriever_handles_empty_candidates():
    retriever = FakeRetriever([])
    reranker = FakeReranker(order=[])
    reranked = RerankedRetriever(retriever, reranker, candidate_k=5)

    assert reranked.retrieve("query", k=5) == []


def test_get_reranker_provider_rejects_unknown_name():
    import pytest

    with pytest.raises(ValueError, match="Unknown reranker provider"):
        get_reranker_provider("not-a-real-provider")
