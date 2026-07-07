import re
import html

from raglens.models.document import Document
from raglens.models.section import Section


class MarkdownSectionParser:

    def parse(
        self,
        markdown: str,
        document: Document
    ) -> Document:

        lines = markdown.splitlines()

        root_sections = []

        stack = []

        current_section = None

        current_content = []

        for line in lines:

            heading_match = re.match(
                r"^(#{1,6})\s+(.*)$",
                line
            )

            if heading_match:

                if current_section:

                    current_section.content = (
                        "\n".join(
                            current_content
                        ).strip()
                    )

                level = len(
                    heading_match.group(1)
                )

                title = html.unescape(
                    heading_match.group(2)
                    .strip()
                )

                section = Section(
                    section_id=title,
                    title=title,
                    level=level,
                    content=""
                )

                while (
                    stack
                    and stack[-1].level >= level
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

                else:

                    root_sections.append(
                        section
                    )

                stack.append(
                    section
                )

                current_section = section

                current_content = []

            else:

                if current_section:

                    current_content.append(
                        line
                    )

        if current_section:

            current_section.content = html.unescape(
                "\n".join(
                    current_content
                ).strip()
            )

        document.sections = (
            root_sections
        )

        return document