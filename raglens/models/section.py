from dataclasses import dataclass, field
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from raglens.models.table import Table
    from raglens.models.equation import Equation


@dataclass
class Section:

    section_id: str

    title: str

    level: int

    content: str

    parent_section_id: str | None = None

    subsections: List["Section"] = field(
        default_factory=list
    )

    tables: List["Table"] = field(
        default_factory=list
    )

    equations: List["Equation"] = field(
        default_factory=list
    )