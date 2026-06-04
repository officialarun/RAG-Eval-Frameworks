from langchain_ollama import ChatOllama
import re


class LLMJudge:

    def __init__(self):

        self.llm = ChatOllama(
            model="qwen3:8b",
            temperature=0
        )

    def score_answer(
        self,
        question,
        ground_truth,
        generated_answer
    ):

        prompt = f"""
        You are evaluating a RAG system.

        Question:
        {question}

        Ground Truth Answer:
        {ground_truth}

        Generated Answer:
        {generated_answer}

        Evaluate how correct the generated answer is compared to the ground truth.

        Scoring Guidelines:

        1.0 = Completely correct

        0.8 = Mostly correct

        0.6 = Partially correct

        0.4 = Limited correctness

        0.2 = Mostly incorrect

        0.0 = Completely incorrect

        Return ONLY a number between 0 and 1.

        Examples:

        0.92

        0.75

        0.34
        """

        try:

            response = self.llm.invoke(prompt)

            score_text = response.content.strip()

            match = re.search(
                r"\d*\.?\d+",
                score_text
            )

            if match:

                score = float(
                    match.group()
                )

                score = max(
                    0.0,
                    min(score, 1.0)
                )

                return round(score, 4)

            return 0.0

        except Exception as e:

            print(
                f"LLM Judge Error: {e}"
            )

            return 0.0


if __name__ == "__main__":

    judge = LLMJudge()

    score = judge.score_answer(
        "What is self attention?",
        "Self attention allows tokens to attend to each other.",
        "Self attention allows tokens in a sequence to interact with one another."
    )

    print(score)