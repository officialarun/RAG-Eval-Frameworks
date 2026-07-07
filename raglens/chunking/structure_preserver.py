import re


class StructurePreserver:
    MAX_ATOMIC_TABLE_LENGTH = 3000
    TABLE_FRAGMENT_SIZE = 1500

    def protect(self, text: str):
        replacements = {}
        counter = 0

        # --------------------------------------------------
        # Formula blocks
        # --------------------------------------------------
        formula_pattern = r"\$\$(.*?)\$\$"

        def replace_formula(match):
            nonlocal counter
            key = f"__FORMULA_{counter}__"
            replacements[key] = {
                "content": match.group(0)
            }
            counter += 1
            return key

        text = re.sub(formula_pattern, replace_formula, text, flags=re.DOTALL)

        # --------------------------------------------------
        # Tables
        # --------------------------------------------------
        lines = text.splitlines(keepends=True)
        output = []
        i = 0

        while i < len(lines):
            line = lines[i]

            # Detect markdown table start
            if "|" in line and i + 1 < len(lines) and "|" in lines[i + 1]:
                table_lines = []
                while i < len(lines) and "|" in lines[i]:
                    table_lines.append(lines[i])
                    i += 1

                table_text = "".join(table_lines)

                # Small table -> atomic
                if len(table_text) <= self.MAX_ATOMIC_TABLE_LENGTH:
                    key = f"__TABLE_{counter}__"
                    replacements[key] = {
                        "content": table_text
                    }
                    output.append(key + "\n")
                    counter += 1

                # Large table -> split by rows
                else:
                    table_chunks = self._split_large_table(
                            table_text
                    )
                    table_id = counter

                    for fragment_idx, chunk in enumerate(table_chunks):

                        key = (
                            f"__TABLE_{table_id}_FRAGMENT_{fragment_idx}__"
                        )

                        replacements[key] = {
                            "content": chunk,
                            "table_id": f"table_{table_id}",
                            "table_fragment_index": fragment_idx,
                            "table_fragment_count": len(table_chunks)
                        }

                        output.append(
                            key + "\n"
                        )
                    counter += 1
            else:
                output.append(line)
                i += 1

        text = "".join(output)
        return text, replacements

    def restore(self, text: str, replacements: dict):
        # Replace keys with their saved content
        for key, value in replacements.items():
            text = text.replace(key, value["content"])
        return text

    def _split_large_table(self, table_text: str):
        lines = table_text.strip().splitlines()
        if len(lines) < 3:
            return [table_text]

        header = lines[0]
        separator = lines[1]
        rows = lines[2:]

        fragments = []
        current_rows = []
        current_size = 0

        for row in rows:
            row_size = len(row)
            if current_rows and current_size + row_size > self.TABLE_FRAGMENT_SIZE:
                fragment = header + "\n" + separator + "\n" + "\n".join(current_rows)
                fragments.append(fragment)
                current_rows = []
                current_size = 0

            current_rows.append(row)
            current_size += row_size

        if current_rows:
            fragment = header + "\n" + separator + "\n" + "\n".join(current_rows)
            fragments.append(fragment)

        return fragments
