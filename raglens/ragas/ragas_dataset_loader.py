import json

from ragas.dataset_schema import (
    EvaluationDataset,
    SingleTurnSample
)

from .answer_cache import (
    get_cache_file
)


class RagasDatasetLoader:

    def load(self, retriever=None, chunk_ids=None):

        samples = []

        cache_file = (
            get_cache_file()
        )

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

                if (
                    retriever is not None
                    and record.get("retriever")
                    != retriever
                ):
                    continue

                if (
                    chunk_ids is not None
                    and record.get("chunk_id")
                    not in chunk_ids
                ):
                    continue

                sample = (
                    SingleTurnSample(

                        user_input=
                            str(record[
                                "question"
                            ]),

                        retrieved_contexts=
                            record[
                                "contexts"
                            ],

                        response=
                            str(record[
                                "generated_answer"
                            ]),

                        reference=
                            str(record[
                                "reference_answer"
                            ])
                    )
                )

                samples.append(
                    sample
                )

        return (
            EvaluationDataset(
                samples=samples
            )
        )