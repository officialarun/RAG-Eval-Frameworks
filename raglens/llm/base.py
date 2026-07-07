from typing import Protocol


class LLMProvider(Protocol):
    """Minimal text-generation contract used by QuestionGenerator/AnswerGenerator.

    Deliberately narrow (single prompt in, string out) since that's all these
    callers need -- the RAGAS judge LLM has its own separate factory
    (raglens.ragas) because it must satisfy ragas's own LLM wrapper interface.
    """

    def generate(self, prompt: str) -> str:
        ...
