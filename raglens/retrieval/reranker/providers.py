_KNOWN_PROVIDERS = {"cross_encoder"}
# "cohere", "jina" -- not implemented yet (API keys, cost, not asked for).
# Adding one later is a new branch here plus a new *_provider.py file,
# not a redesign of RerankedRetriever or the RerankerProvider interface.


def get_reranker_provider(provider: str = "cross_encoder", model: str | None = None, **kwargs):

    provider = provider.lower()

    if provider not in _KNOWN_PROVIDERS:
        raise ValueError(
            f"Unknown reranker provider: {provider!r}. "
            f"Choose from {sorted(_KNOWN_PROVIDERS)}."
        )

    from raglens.retrieval.reranker.cross_encoder_provider import CrossEncoderReranker

    return CrossEncoderReranker(model_name=model) if model else CrossEncoderReranker()
