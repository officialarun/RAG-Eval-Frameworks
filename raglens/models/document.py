from dataclasses import dataclass, field
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from raglens.models.section import Section


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