from dataclasses import dataclass

from src_v2.models import (
    Chunk
)


@dataclass
class HierarchicalRetrievalResult:

    parent_chunk: Chunk

    child_chunk: Chunk

    score: float

    retriever_name: str