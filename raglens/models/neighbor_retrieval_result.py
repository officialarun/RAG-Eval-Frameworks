from dataclasses import dataclass

from raglens.models import Chunk


@dataclass
class NeighborRetrievalResult:

    center_chunk: Chunk

    neighbor_chunks: list[Chunk]

    score: float

    retriever_name: str