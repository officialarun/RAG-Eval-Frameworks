import re


class StructurePreserver:

    def protect(
        self,
        text: str
    ):

        replacements = {}

        counter = 0

        # Formula blocks
        formula_pattern = r"\$\$(.*?)\$\$"

        def replace_formula(match):

            nonlocal counter

            key = (
                f"__FORMULA_{counter}__"
            )

            replacements[key] = (
                match.group(0)
            )

            counter += 1

            return key

        text = re.sub(
            formula_pattern,
            replace_formula,
            text,
            flags=re.DOTALL
        )

        # Markdown tables
        table_pattern = (
            r"(\|.*\|\n)+"
        )

        def replace_table(match):

            nonlocal counter

            key = (
                f"__TABLE_{counter}__"
            )

            replacements[key] = (
                match.group(0)
            )

            counter += 1

            return key

        text = re.sub(
            table_pattern,
            replace_table,
            text
        )

        return text, replacements

    def restore(
        self,
        text: str,
        replacements: dict
    ):

        for key, value in replacements.items():

            text = text.replace(
                key,
                value
            )

        return text