from raglens.llm import LLMProvider, get_llm_provider


class AnswerGenerator:

    def __init__(self, llm_provider: LLMProvider | None = None):

        self.llm_provider = (
            llm_provider or get_llm_provider("openai")
        )
    
    def build_prompt(
    self,
    question,
    contexts
    ):

        context_text = (
            "\n\n".join(
                contexts
            )
        )

        return f"""
        Answer the question using ONLY the provided context.

        If the answer cannot be determined from the context,
        reply exactly:

        Insufficient information

        Question:
        {question}

        Context:
        {context_text}

        Answer:
        """

    def generate(
    self,
    question,
    contexts
    ):

        prompt = (
            self.build_prompt(
                question,
                contexts
            )
        )

        return (
            self.llm_provider.generate(
                prompt
            )
        )