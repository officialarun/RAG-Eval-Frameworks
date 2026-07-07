from .answer_cache import (
    load_completed_keys,
    append_record
)


class AnswerDatasetBuilder:

    def __init__(
        self,
        answer_generator
    ):

        self.answer_generator = (
            answer_generator
        )

    def build(
        self,
        ragas_samples
    ):

        completed = (
            load_completed_keys()
        )

        total = len(
            ragas_samples
        )

        for index, sample in enumerate(
            ragas_samples,
            start=1
        ):

            key = (

                sample.metadata[
                    "chunk_id"
                ],

                sample.metadata[
                    "retriever"
                ]
            )

            if key in completed:

                continue

            try:

                answer = (

                    self.answer_generator
                    .generate(

                        sample.question,

                        sample.contexts
                    )
                )

                append_record({

                    "chunk_id":
                        sample.metadata[
                            "chunk_id"
                        ],

                    "retriever":
                        sample.metadata[
                            "retriever"
                        ],

                    "question":
                        sample.question,

                    "contexts":
                        sample.contexts,

                    "generated_answer":
                        answer,

                    "reference_answer":
                        sample.reference_answer,

                    "status":
                        "success"
                })

            except Exception as error:

                append_record({

                    "chunk_id":
                        sample.metadata[
                            "chunk_id"
                        ],

                    "retriever":
                        sample.metadata[
                            "retriever"
                        ],

                    "question":
                        sample.question,

                    "status":
                        "failed",

                    "error":
                        str(error)
                })

            if index % 10 == 0:

                print(
                    f"{index}/{total}"
                )