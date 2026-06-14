import json

from pathlib import Path

from src_v2.models import (
    QuestionSample
)

from .question_cache import (
    get_cache_file
)


class QuestionDatasetLoader:

    def load(self):

        samples = []
        cache_file = get_cache_file()
        with open(
            cache_file,
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                record = json.loads(
                    line
                )

                if (
                    record["status"]
                    != "success"
                ):
                    continue

                sample = QuestionSample(

                    sample_id=
                        record["chunk_id"],

                    document_title="",

                    question=
                        record["question"],

                    reference_answer=
                        record[
                            "reference_answer"
                        ],

                    supporting_context="",

                    relevant_sections=[
                        record[
                            "path"
                        ]
                    ],

                    metadata={

                        "chunk_id":
                            record[
                                "chunk_id"
                            ],

                        "section_title":
                            record[
                                "section_title"
                            ],

                        "path":
                            record[
                                "path"
                            ]
                    }
                )

                samples.append(
                    sample
                )

        return samples