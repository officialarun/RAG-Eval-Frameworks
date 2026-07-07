import os

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

_DEFAULT_MODELS = {
    "ollama": "nomic-embed-text",
    "openai": "text-embedding-3-small",
}


def get_embedding_provider(
    provider: str,
    model: str | None = None,
    **kwargs,
) -> Embeddings:
    """Return a LangChain-compatible Embeddings instance (embed_query/embed_documents).

    The same instance can be used both for indexing (EmbeddingGenerator) and
    retrieval (DenseRetriever) -- there is only one embedding object per run.
    """

    provider = provider.lower()

    if provider not in _DEFAULT_MODELS:
        raise ValueError(
            f"Unknown embedding provider: {provider!r}. "
            f"Choose from {sorted(_DEFAULT_MODELS)}."
        )

    model = model or _DEFAULT_MODELS[provider]

    if provider == "ollama":
        base_url = kwargs.pop("base_url", None) or os.getenv("OLLAMA_HOST")
        return OllamaEmbeddings(model=model, base_url=base_url, **kwargs)

    if provider == "openai":
        api_key = kwargs.pop("api_key", None) or os.getenv("OPENAI_API_KEY")
        return OpenAIEmbeddings(model=model, api_key=api_key, **kwargs)

    raise AssertionError("unreachable")
