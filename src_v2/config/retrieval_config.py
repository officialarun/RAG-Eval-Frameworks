
import re
BAD_SECTIONS = {
    "references",
    "bibliography",
    "contents",
    "index",
    "list of figures",
    "list of tables"
}

DEFAULT_TOP_K = 5

EXCLUDE_REFERENCE_SECTIONS = True


def is_bad_section(
    section_title: str
):

    normalized = re.sub(
        r"[^a-z0-9 ]",
        "",
        section_title.lower()
    ).strip()

    return any(
        bad in normalized
        for bad in BAD_SECTIONS
    )