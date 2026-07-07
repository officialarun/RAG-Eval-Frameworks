from raglens.config import RetrievalConfig, DEFAULT_CONFIG

from raglens.models import (
    Chunk,
    RetrievalResult
)

from raglens.retrieval.base import (
    BaseRetriever
)


class DenseRetriever(
    BaseRetriever
):

    def __init__(
        self,
        embedding_model,
        vector_store,
        config: RetrievalConfig = DEFAULT_CONFIG
    ):

        self.embedding_model = (
            embedding_model
        )

        self.vector_store = (
            vector_store
        )

        self.config = config

    def retrieve(
        self,
        query: str,
        k: int = 5
    ):

        query_embedding = (
            self.embedding_model.embed_query(
                query
            )
        )

        results = (
            self.vector_store.similarity_search(
                query_embedding=query_embedding,
                k=k * 3
            )
        )

        retrieval_results = []

        docs = results["documents"][0]

        metas = results["metadatas"][0]

        distances = results["distances"][0]

        for doc, meta, distance in zip(
            docs,
            metas,
            distances
        ):
            if (
                self.config.exclude_reference_sections
                and
                self.config.is_bad_section(meta["section_title"])
            ):
                continue

            chunk = Chunk(
                chunk_id=
                    meta["chunk_id"],

                parent_doc_id=
                    meta["parent_doc_id"],

                parent_section_id=
                    meta["parent_section_id"],

                chunk_order=
                    meta["chunk_order"],

                fragment_index=
                    meta["fragment_index"],

                section_title=
                    meta["section_title"],

                path=
                    meta["path"],

                level=
                    meta["level"],

                chunk_type=
                    meta["chunk_type"],

                content=doc
            )

            retrieval_results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=1 - distance,
                    retriever_name="dense"
                )
            )

            if len(
                retrieval_results
            ) >= k:

                break

        return retrieval_results