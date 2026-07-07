from raglens.config import RetrievalConfig
from raglens.models import Chunk
from raglens.retrieval.bm25.bm25_retriever import BM25Retriever


def _chunk(chunk_id, section_title, content):
    return Chunk(
        chunk_id=chunk_id,
        parent_doc_id="doc_0",
        parent_section_id="s0",
        chunk_order=0,
        fragment_index=0,
        section_title=section_title,
        path="doc_0 > " + section_title,
        level=1,
        chunk_type="section_fragment",
        content=content,
    )


def test_bm25_ranks_relevant_chunk_first():
    chunks = [
        _chunk("c1", "Intro", "gradient descent optimizes model parameters"),
        _chunk("c2", "Intro", "cats and dogs are common pets"),
    ]
    retriever = BM25Retriever(chunks)
    results = retriever.retrieve("gradient descent parameters", k=5)
    assert results[0].chunk.chunk_id == "c1"


def test_bm25_excludes_bad_sections_by_default():
    chunks = [
        _chunk("c1", "References", "gradient descent gradient descent gradient descent"),
        _chunk("c2", "Introduction", "gradient descent applied here"),
    ]
    retriever = BM25Retriever(chunks)
    results = retriever.retrieve("gradient descent", k=5)
    ids = [r.chunk.chunk_id for r in results]
    assert "c1" not in ids
    assert "c2" in ids


def test_bm25_custom_config_can_include_references():
    chunks = [
        _chunk("c1", "References", "gradient descent gradient descent gradient descent"),
    ]
    retriever = BM25Retriever(chunks, config=RetrievalConfig(exclude_reference_sections=False))
    results = retriever.retrieve("gradient descent", k=5)
    assert results[0].chunk.chunk_id == "c1"
