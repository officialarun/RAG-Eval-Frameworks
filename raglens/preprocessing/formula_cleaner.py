import re

class FormulaCleaner:
    def clean_formula(
        self,
        formula: str
    ) -> str:

        formula = re.sub(
            r"&&\s*\(\d+\)",
            "",
            formula
        )

        formula = formula.replace(
            "&=",
            "="
        )

        formula = formula.replace(
            "&",
            ""
        )

        formula = formula.replace(
            "\\\\",
            " "
        )

        pattern = r'(?<=\b[A-Za-z])\s+(?=[A-Za-z]\b)'

        previous = None

        while previous != formula:

            previous = formula

            formula = re.sub(
                pattern,
                '',
                formula
            )

        formula = re.sub(
            r'\s+',
            ' ',
            formula
        )

        return formula.strip()
    
    def process_markdown(
        self,
        markdown: str
    ) -> str:

        formulas = re.findall(
            r"\$\$(.*?)\$\$",
            markdown,
            flags=re.DOTALL
        )

        for formula in formulas:

            cleaned = self.clean_formula(
                formula
            )

            markdown = markdown.replace(
                f"$${formula}$$",
                f"$${cleaned}$$"
            )

        return markdown