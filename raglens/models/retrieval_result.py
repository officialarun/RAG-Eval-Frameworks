from dataclasses import dataclass

from raglens.models import Chunk


@dataclass
class RetrievalResult:

    chunk: Chunk

    score: float

    retriever_name: str