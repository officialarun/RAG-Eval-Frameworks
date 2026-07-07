from dataclasses import dataclass


@dataclass
class FlattenedSection:

    section_id: str

    document_id: str

    document_title: str

    section_title: str

    level: int

    path: str

    content: str

    parent_section_id: str | None