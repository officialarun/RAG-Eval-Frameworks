from raglens.question_generation import (
    append_record,
    get_completed_chunk_ids
)

from raglens.question_generation import (
    QuestionGenerator
)


class QuestionDatasetBuilder:

    def __init__(self, generator: QuestionGenerator | None = None):

        self.generator = (
            generator or QuestionGenerator()
        )

    def build(
        self,
        chunks
    ):

        completed_ids = (
            get_completed_chunk_ids()
        )

        total = len(
            chunks
        )

        for index, chunk in enumerate(
            chunks,
            start=1
        ):

            if (
                chunk.chunk_id
                in completed_ids
            ):

                print(
                    f"[{index}/{total}] "
                    f"SKIPPING "
                    f"{chunk.chunk_id}"
                )

                continue

            try:

                result = (
                    self.generator.generate(
                        chunk
                    )
                )

                record = {

                    "chunk_id":
                        chunk.chunk_id,

                    "section_title":
                        chunk.section_title,

                    "path":
                        chunk.path,

                    **result
                }

                append_record(
                    record
                )

                print(
                    f"[{index}/{total}] "
                    f"{result['status'].upper()} "
                    f"{chunk.chunk_id}"
                )

            except Exception as error:

                append_record(
                    {
                        "chunk_id":
                            chunk.chunk_id,

                        "section_title":
                            chunk.section_title,

                        "path":
                            chunk.path,

                        "status":
                            "failed",

                        "error":
                            str(error)
                    }
                )

                print(
                    f"[{index}/{total}] "
                    f"FAILED "
                    f"{chunk.chunk_id}"
                )