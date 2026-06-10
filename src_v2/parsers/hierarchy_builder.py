from src_v2.models import (
    Document,
    Section
)


class HierarchyBuilder:

    def build(
        self,
        document: Document
    ) -> Document:

        for section in document.sections:

            section.parent_section_id = None

            section.subsections = []

        stack = []

        for section in document.sections:

            while (
                stack
                and stack[-1].level >= section.level
            ):
                stack.pop()

            if stack:

                parent = stack[-1]

                section.parent_section_id = (
                    parent.section_id
                )

                parent.subsections.append(
                    section
                )

            stack.append(
                section
            )

        return document