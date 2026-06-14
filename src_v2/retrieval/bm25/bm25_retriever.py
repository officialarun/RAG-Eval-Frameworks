from rank_bm25 import BM25Okapi
import re

from src_v2.models import Chunk, RetrievalResult
from src_v2.retrieval.base import BaseRetriever

# IMPORTANT:
# import module, not values
from src_v2.config import retrieval_config


class BM25Retriever(BaseRetriever):
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
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
                retrieval_config.EXCLUDE_REFERENCE_SECTIONS
                and retrieval_config.is_bad_section(chunk.section_title)
            ):
                continue

            retrieval_results.append(
                RetrievalResult(chunk=chunk, score=float(score), retriever_name="bm25")
            )

            if len(retrieval_results) >= k:
                break

        return retrieval_results
