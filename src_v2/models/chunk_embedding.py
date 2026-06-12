from dataclasses import dataclass

from src_v2.models import Chunk


@dataclass
class ChunkEmbedding:

    chunk_id: str

    embedding: list[float]

    chunk: Chunk