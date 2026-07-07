from raglens.models import RagasSample

class EvaluationDatasetBuilder:
    def _extract_context(self, result):
        # Flat retrievers
        if hasattr(result, "chunk"):
            return result.chunk.content

        # Hierarchical retriever
        if hasattr(result, "child_chunk"):
            return result.child_chunk.content

        # NeighborHierrarchicalRetriever
        if hasattr(result, "content"):
            return result.content

        raise ValueError("Unknown retrieval result type")


    def build(self, samples, retriever, retriever_name, k=5):
        dataset = []
        total = len(samples)
        max_contexts = 10
        for index, sample in enumerate(samples, start=1):
            results = retriever.retrieve(sample.question, k=k)

            contexts = [self._extract_context(result) for result in results]
            contexts = contexts[:max_contexts]
            dataset.append(
                RagasSample(
                    question=sample.question,
                    reference_answer=sample.reference_answer,
                    contexts=contexts,
                    generated_answer=getattr(sample, "generated_answer", None),
                    metadata={
                        "retriever": retriever_name,
                        "chunk_id": sample.metadata["chunk_id"],
                        "section_title": sample.metadata["section_title"],
                        "path": sample.metadata["path"],
                    },
                )
            )

            if index % 50 == 0:
                print(f"{index}/{total}")

        return dataset
