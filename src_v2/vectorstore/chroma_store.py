from chromadb import PersistentClient

from src_v2.models import (
    ChunkEmbedding
)


class ChromaStore:

    def __init__(
        self,
        persist_directory="./chroma_db",
        collection_name="rag_chunks"
    ):

        self.client = (
            PersistentClient(
                path=persist_directory
            )
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=collection_name
            )
        )

    def count(self):

        return self.collection.count()

    def reset(self):

        self.client.delete_collection(
            self.collection.name
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=self.collection.name
            )
        )
    
    def similarity_search(
        self,
        query_embedding,
        k=5
    ):
        return self.collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=k
        )

    def add_embeddings(
        self,
        chunk_embeddings: list[ChunkEmbedding]
    ):
        ids = []

        embeddings = []

        documents = []

        metadatas = []

        for item in chunk_embeddings:

            chunk = item.chunk

            ids.append(
                chunk.chunk_id
            )

            embeddings.append(
                item.embedding
            )

            documents.append(
                chunk.content
            )

            metadatas.append(
                {
                    "chunk_id":
                        chunk.chunk_id,

                    "chunk_order":
                        chunk.chunk_order,

                    "parent_doc_id":
                        chunk.parent_doc_id,

                    "parent_section_id":
                        chunk.parent_section_id,

                    "section_title":
                        chunk.section_title,

                    "fragment_index":
                        chunk.fragment_index,

                    "path":
                        chunk.path,

                    "level":
                        chunk.level,

                    "chunk_type":
                        chunk.chunk_type
                }
            )

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
    
    # similarity search method to find the most similar chunks based on the query embedding
    def similarity_search(
        self,
        query_embedding,
        k=5
    ):
        return self.collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=k
        )