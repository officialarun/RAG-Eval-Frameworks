from langchain_core.embeddings import Embeddings

from raglens.models import (
    Chunk,
    ChunkEmbedding
)


class EmbeddingGenerator:

    def __init__(
        self,
        embeddings: Embeddings,
        batch_size: int = 50
    ):

        self.embeddings = (
            embeddings
        )
        self.batch_size = (
            batch_size
        )

    def generate(
        self,
        chunks: list[Chunk]
    ) -> list[ChunkEmbedding]:

        texts = [
            chunk.content
            for chunk in chunks
        ]

        all_embeddings = []

        for i in range(
            0,
            len(texts),
            self.batch_size
        ):
            batch = texts[
                i:i + self.batch_size
            ]
            batch_embeddings = (
                self.embeddings.embed_documents(
                    batch
                )
            )
            all_embeddings.extend(
                batch_embeddings
            )
            print(
                f"Processed "
                f"{min(i + self.batch_size, len(texts))}"
                f"/{len(texts)} chunks"
            )

        chunk_embeddings = []

        for chunk, embedding in zip(
            chunks,
            all_embeddings
        ):

            chunk_embeddings.append(
                ChunkEmbedding(
                    chunk_id=chunk.chunk_id,
                    embedding=embedding,
                    chunk=chunk
                )
            )

        return chunk_embeddings
