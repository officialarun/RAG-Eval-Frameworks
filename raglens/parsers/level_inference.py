import re

from raglens.models import Document


class LevelInference:

    def infer(
        self,
        document: Document
    ) -> Document:

        for section in document.sections:

            match = re.match(
                r"^(\d+(?:\.\d+)*)",
                section.title
            )

            if match:

                number = match.group(1)

                section.level = (
                    number.count(".") + 1
                )

        return document