from dataclasses import dataclass
from typing import Any
from typing import Optional


@dataclass
class RagasSample:

    question: str

    reference_answer: str

    contexts: list[str]

    generated_answer: Optional[str] = None

    metadata: dict[str, Any] = None