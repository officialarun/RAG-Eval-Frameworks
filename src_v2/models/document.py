from dataclasses import dataclass, field
from typing import List
@dataclass
class Document:

    doc_id: str

    title: str

    source_type: str

    source_path: str

    sections: List["Section"] = field(
        default_factory=list
    )

    metadata: dict = field(
        default_factory=dict
    )