from langchain_ollama import OllamaEmbeddings
import numpy as np


class SemanticSimilarity:

    def __init__(self):

        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text"
        )

    def cosine_similarity(
        self,
        vec1,
        vec2
    ):

        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        similarity = np.dot(
            vec1,
            vec2
        ) / (
            np.linalg.norm(vec1)
            * np.linalg.norm(vec2)
        )

        return float(similarity)

    def score(
        self,
        text1,
        text2
    ):

        emb1 = self.embeddings.embed_query(
            text1
        )

        emb2 = self.embeddings.embed_query(
            text2
        )

        similarity = self.cosine_similarity(
            emb1,
            emb2
        )

        return round(
            similarity,
            4
        )

    def answer_similarity(
        self,
        ground_truth,
        generated_answer
    ):

        return self.score(
            ground_truth,
            generated_answer
        )

    def context_similarity(
        self,
        original_context,
        retrieved_context
    ):

        return self.score(
            original_context,
            retrieved_context
        )