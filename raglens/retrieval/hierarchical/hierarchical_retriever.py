from collections import defaultdict

from raglens.models import (
    Chunk,
    HierarchicalRetrievalResult,
)

from raglens.retrieval.hybrid import (
    HybridRetriever,
)


class HierarchicalRetriever:

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        all_chunks: list[Chunk]
    ):

        self.hybrid_retriever = hybrid_retriever

        self.parent_lookup = {}

        for chunk in all_chunks:

            if chunk.chunk_type == "parent_section":

                key = (
                    chunk.parent_doc_id,
                    chunk.parent_section_id
                )

                self.parent_lookup[key] = chunk

    def retrieve(
        self,
        query: str,
        k: int = 5,
        child_k: int = 20
    ):

        child_results = (
            self.hybrid_retriever.retrieve(
                query=query,
                k=child_k
            )
        )

        grouped_results = defaultdict(list)

        for result in child_results:

            child_chunk = result.chunk

            parent_key = (
                child_chunk.parent_doc_id,
                child_chunk.parent_section_id
            )

            if parent_key not in self.parent_lookup:
                continue

            grouped_results[parent_key].append(
                result
            )

        hierarchical_results = []

        for parent_key, results in grouped_results.items():

            parent_chunk = (
                self.parent_lookup[
                    parent_key
                ]
            )

            best_child = max(
                results,
                key=lambda x: x.score
            )

            hierarchical_results.append(
                HierarchicalRetrievalResult(
                    parent_chunk=parent_chunk,
                    child_chunk=best_child.chunk,
                    score=best_child.score,
                    retriever_name="hierarchical"
                )
            )

        hierarchical_results.sort(
            key=lambda x: x.score,
            reverse=True
        )

        return hierarchical_results[:k]