import os

from dotenv import load_dotenv
from ragas.llms import llm_factory
from ragas.metrics._context_recall import ContextRecall
from ragas.metrics._factual_correctness import FactualCorrectness
from ragas.metrics._faithfulness import Faithfulness

from raglens.ragas.fast_context_precision import FastContextPrecision

load_dotenv()

_DEFAULT_JUDGE_MODELS = {
    "openai": "gpt-4o-mini",
    "groq": "llama-3.3-70b-versatile",
    "ollama": "qwen3:8b",
}


def get_ragas_judge_llm(provider: str = "openai", model: str | None = None, **kwargs):
    """Build a ragas-compatible judge LLM (has .generate/.agenerate) for the
    chosen provider. Ollama is reached through its OpenAI-compatible /v1
    endpoint since ragas' llm_factory doesn't have a native ollama provider.
    """

    provider = provider.lower()
    model = model or _DEFAULT_JUDGE_MODELS.get(provider)

    temperature = kwargs.pop("temperature", 0)
    max_tokens = kwargs.pop("max_tokens", 1024)

    if provider == "openai":
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=kwargs.pop("api_key", None) or os.getenv("OPENAI_API_KEY"))
        return llm_factory(
            model=model, provider="openai", client=client,
            temperature=temperature, max_tokens=max_tokens, **kwargs,
        )

    if provider == "groq":
        from groq import AsyncGroq

        client = AsyncGroq(api_key=kwargs.pop("api_key", None) or os.getenv("GROQ_API_KEY"))
        return llm_factory(
            model=model, provider="groq", client=client,
            temperature=temperature, max_tokens=max_tokens, **kwargs,
        )

    if provider == "ollama":
        from openai import AsyncOpenAI

        base_url = kwargs.pop("base_url", None) or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        client = AsyncOpenAI(base_url=base_url, api_key="ollama")
        return llm_factory(
            model=model, provider="openai", client=client,
            temperature=temperature, max_tokens=max_tokens, **kwargs,
        )

    raise ValueError(
        f"Unknown ragas judge provider: {provider!r}. "
        f"Choose from {sorted(_DEFAULT_JUDGE_MODELS)}."
    )


def get_default_metrics():
    return [
        Faithfulness(),
        FactualCorrectness(),
        FastContextPrecision(),
        ContextRecall(),
    ]
