from src_v2.models import (
    Chunk,
    NeighborRetrievalResult
)

from src_v2.retrieval.hybrid import (
    HybridRetriever
)


class NeighborRetriever:

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        embedding_chunks: list[Chunk]
    ):

        self.hybrid_retriever = (
            hybrid_retriever
        )

        self.chunk_lookup = {}

        for chunk in embedding_chunks:

            key = (
                chunk.parent_doc_id,
                chunk.chunk_order
            )

            self.chunk_lookup[
                key
            ] = chunk

    def retrieve(
        self,
        query: str,
        k: int = 5
    ):

        base_results = (
            self.hybrid_retriever.retrieve(
                query=query,
                k=k
            )
        )

        results = []

        for result in base_results:

            center_chunk = (
                result.chunk
            )

            neighbors = []

            for offset in [
                -1,
                0,
                1
            ]:

                key = (
                    center_chunk.parent_doc_id,
                    center_chunk.chunk_order + offset
                )

                if key in self.chunk_lookup:

                    neighbors.append(
                        self.chunk_lookup[key]
                    )

            results.append(
                NeighborRetrievalResult(
                    center_chunk=center_chunk,
                    neighbor_chunks=neighbors,
                    score=result.score,
                    retriever_name="neighbor"
                )
            )

        return results