import re
from dataclasses import dataclass, field

_DEFAULT_BAD_SECTIONS = frozenset({
    "references",
    "bibliography",
    "contents",
    "index",
    "list of figures",
    "list of tables",
})


@dataclass
class RetrievalConfig:

    bad_sections: frozenset[str] = field(
        default_factory=lambda: _DEFAULT_BAD_SECTIONS
    )
    default_top_k: int = 5
    exclude_reference_sections: bool = True

    def is_bad_section(self, section_title: str) -> bool:

        normalized = re.sub(
            r"[^a-z0-9 ]",
            "",
            section_title.lower()
        ).strip()

        return any(
            bad in normalized
            for bad in self.bad_sections
        )


DEFAULT_CONFIG = RetrievalConfig()
