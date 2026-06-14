from dataclasses import dataclass

from src_v2.models import Chunk


@dataclass
class RetrievalResult:

    chunk: Chunk

    score: float

    retriever_name: str