from dataclasses import dataclass

from raglens.models import (
    Chunk
)


@dataclass
class HierarchicalRetrievalResult:

    parent_chunk: Chunk

    child_chunk: Chunk

    score: float

    retriever_name: str