class CrossEncoderReranker:

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):

        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:

        if not documents:
            return []

        pairs = [[query, doc] for doc in documents]

        # One batched call -- never loop per-document.
        scores = self.model.predict(pairs)

        ranked = sorted(
            enumerate(scores),
            key=lambda pair: pair[1],
            reverse=True,
        )

        return [(index, float(score)) for index, score in ranked]
