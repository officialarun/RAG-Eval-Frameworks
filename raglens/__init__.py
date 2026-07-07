__version__ = "0.1.0"

__all__ = ["RagLensPipeline"]


def __getattr__(name):
    # Lazy: raglens.pipeline pulls in Docling/transformers, which is heavy
    # and irrelevant to callers that only need e.g. raglens.evaluation or
    # raglens.ragas (CLI report command, the dashboard, unit tests, ...).
    if name == "RagLensPipeline":
        from raglens.pipeline import RagLensPipeline
        return RagLensPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
