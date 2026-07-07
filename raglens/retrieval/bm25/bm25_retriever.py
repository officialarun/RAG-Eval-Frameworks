from rank_bm25 import BM25Okapi
import re

from raglens.models import Chunk, RetrievalResult
from raglens.retrieval.base import BaseRetriever
from raglens.config import RetrievalConfig, DEFAULT_CONFIG


class BM25Retriever(BaseRetriever):
    def __init__(self, chunks: list[Chunk], config: RetrievalConfig = DEFAULT_CONFIG):
        self.chunks = chunks
        self.config = config
        self.corpus = [chunk.content for chunk in chunks]
        self.tokenized_corpus = [self._tokenize(document) for document in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _tokenize(self, text: str):
        return re.findall(r"\w+", text.lower())

    def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]:
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(zip(self.chunks, scores), key=lambda x: x[1], reverse=True)

        retrieval_results: list[RetrievalResult] = []

        for chunk, score in ranked:
            if (
                self.config.exclude_reference_sections
                and self.config.is_bad_section(chunk.section_title)
            ):
                continue

            retrieval_results.append(
                RetrievalResult(chunk=chunk, score=float(score), retriever_name="bm25")
            )

            if len(retrieval_results) >= k:
                break

        return retrieval_results
