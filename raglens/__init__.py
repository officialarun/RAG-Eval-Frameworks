from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("raglens-toolkit")
except PackageNotFoundError:
    # Editable/dev install without build metadata (e.g. running from a git
    # clone that was never `pip install -e .`'d) -- not a real version, but
    # better than crashing or silently hardcoding a string that goes stale
    # the moment pyproject.toml's version is bumped without updating here too.
    __version__ = "0.0.0-dev"

__all__ = ["RagLensPipeline"]


def __getattr__(name):
    # Lazy: raglens.pipeline pulls in Docling/transformers, which is heavy
    # and irrelevant to callers that only need e.g. raglens.evaluation or
    # raglens.ragas (CLI report command, the dashboard, unit tests, ...).
    if name == "RagLensPipeline":
        from raglens.pipeline import RagLensPipeline
        return RagLensPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
