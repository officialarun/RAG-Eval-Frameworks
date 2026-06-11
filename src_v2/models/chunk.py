from dataclasses import dataclass, field


@dataclass
class Chunk:

    chunk_id: str

    parent_doc_id: str

    parent_section_id: str

    chunk_order: int

    fragment_index: int

    section_title: str

    path: str

    level: int

    chunk_type: str

    content: str

    metadata: dict = field(
        default_factory=dict
    )