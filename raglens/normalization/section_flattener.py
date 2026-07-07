from raglens.models import (
    Document,
    Section,
    FlattenedSection
)


class SectionFlattener:

    def flatten(
        self,
        document: Document
    ) -> list[FlattenedSection]:

        flattened = []

        roots = [
            section
            for section in document.sections
            if section.parent_section_id is None
        ]

        for section in roots:

            self._flatten_section(
            section=section,
            document=document,
            path=document.title,
            flattened=flattened
        )

        return flattened

    def _flatten_section(
        self,
        section: Section,
        document: Document,
        path: str,
        flattened: list[FlattenedSection]
    ):

        current_path = (
            f"{path} > {section.title}"
        )

        flattened.append(
            FlattenedSection(
                section_id=section.section_id,
                section_title=section.title,
                document_id=document.doc_id,
                document_title=document.title,
                level=section.level,
                path=current_path,
                content=section.content,
                parent_section_id=section.parent_section_id
            )
        )

        for subsection in section.subsections:

            self._flatten_section(
                subsection,
                document,
                current_path,
                flattened
            )