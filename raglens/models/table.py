from dataclasses import dataclass, field
from typing import List
@dataclass
class Table:

    table_id: str

    title: str

    headers: List[str]

    rows: List[List[str]]

    context: str

    metadata: dict = field(
        default_factory=dict
    )