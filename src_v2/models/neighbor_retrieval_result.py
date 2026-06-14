from dataclasses import dataclass

from src_v2.models import Chunk


@dataclass
class NeighborRetrievalResult:

    center_chunk: Chunk

    neighbor_chunks: list[Chunk]

    score: float

    retriever_name: str