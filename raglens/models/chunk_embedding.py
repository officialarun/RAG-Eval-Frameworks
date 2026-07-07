from dataclasses import dataclass

from raglens.models import Chunk


@dataclass
class ChunkEmbedding:

    chunk_id: str

    embedding: list[float]

    chunk: Chunk