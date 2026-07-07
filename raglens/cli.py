import asyncio

import typer

# Only base-dependency imports at module level -- `report`/`info`/`--help`
# must work with a lean `pip install raglens-toolkit`, without the `full`
# extra (docling/chromadb/langchain-ollama/sentence-transformers/...) that
# ingest/index/questions/benchmark/evaluate need. Those commands import
# their own heavy deps lazily, guarded by _require_full() below so a
# missing extra produces one clear line instead of a raw traceback.
from raglens.ragas import RagasScorer, get_ragas_judge_llm, plot_ragas_scores, stratified_sample_by_document

# Judge-LLM latency is inherently variable (rate limits, time of day, provider
# load) -- these are the best/worst-case minutes-per-sample observed across
# real runs of this project, used only to print a rough upfront estimate.
_MIN_MINUTES_PER_SAMPLE = 2
_MAX_MINUTES_PER_SAMPLE = 8

ALL_RETRIEVERS = ["bm25", "dense", "hybrid", "hierarchical", "neighbor"]

_DOCS_URL = "https://github.com/officialarun/RAG-Eval-Frameworks"
_PYPI_URL = "https://pypi.org/project/raglens-toolkit/"

# Rich reflows/collapses whitespace in the epilog (no preserved line breaks,
# no lists), and [bracket] text is interpreted as markup unless escaped --
# so this stays short prose, not a formatted example block. The full,
# exactly-formatted workflow walkthrough lives in `raglens info` instead,
# which uses plain typer.echo() and isn't subject to this reflow.
_EPILOG = (
    "New here? Run `raglens info` for the full workflow, install status, "
    "and links -- it's meant to answer everything without needing the README. "
    "Run `raglens COMMAND --help` for that command's options. One thing to "
    "know upfront: ingest/index/questions/benchmark/evaluate need the "
    "\\[full] extra (pip install raglens-toolkit\\[full]); `report` and the "
    "dashboard (streamlit run dashboard/app.py) work without it. "
    f"Docs: {_DOCS_URL}"
)

app = typer.Typer(
    help="raglens: benchmark and RAGAS-evaluate a RAG pipeline over your own PDF corpus.",
    epilog=_EPILOG,
)


def _version_callback(show_version: bool):
    if show_version:
        import raglens

        typer.echo(f"raglens-toolkit {raglens.__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None, "--version", callback=_version_callback, is_eager=True,
        help="Show the installed version and exit.",
    ),
):
    pass


def _require_full():
    """Call at the top of any command needing the `[full]` extra. Turns a
    raw ModuleNotFoundError into one actionable line instead of a traceback
    a first-time user has no way to interpret."""

    try:
        import docling  # noqa: F401
    except ModuleNotFoundError:
        typer.secho(
            "\nThis command needs the `[full]` extra (Docling, ChromaDB, "
            "Ollama client, rank-bm25, sentence-transformers, ...), which "
            "isn't installed.\n\n"
            "  pip install raglens-toolkit[full]\n",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)


def _full_is_installed() -> bool:
    try:
        import docling  # noqa: F401

        return True
    except ModuleNotFoundError:
        return False


@app.command()
def info():
    """Show version, what's installed, and where to go next.

    Run this first if you're new here or something isn't working --
    it's meant to answer "what did I actually just install?" without
    needing to open the README.
    """

    import raglens

    full_installed = _full_is_installed()

    typer.echo(f"raglens-toolkit {raglens.__version__}\n")

    typer.echo("Install status:")
    typer.echo("  base (dashboard + `report`)                         : installed")
    typer.echo(f"  [full] (ingest/index/questions/benchmark/evaluate) : {'installed' if full_installed else 'NOT installed'}")
    if not full_installed:
        typer.echo("\n  Missing commands need: pip install raglens-toolkit[full]")

    typer.echo("\nCommands:")
    typer.echo("  ingest     Parse a PDF corpus into structure-aware chunks.        [needs full]")
    typer.echo("  index      Embed + store chunks in a vector index.               [needs full]")
    typer.echo("  questions  Generate Q&A ground truth (resumable).                [needs full]")
    typer.echo("  benchmark  Benchmark retrieval quality across all 5 strategies.  [needs full]")
    typer.echo("  evaluate   Run RAGAS end-to-end evaluation (resumable).          [needs full]")
    typer.echo("  report     Aggregate whatever's been scored so far.             [base install OK]")

    typer.echo("\nNot a CLI command, run separately:")
    typer.echo("  streamlit run dashboard/app.py     Visualize benchmark/RAGAS results in a browser")

    typer.echo("\nAdvanced (Python API, not exposed as a flag except `evaluate --reranker`):")
    typer.echo("  from raglens.retrieval import RerankedRetriever, get_reranker_provider")

    typer.echo(f"\nFull docs: {_DOCS_URL}")
    typer.echo(f"PyPI: {_PYPI_URL}")


def _ingest_or_exit(pipeline, docs: str, no_cache: bool):
    """Wraps pipeline.ingest() so a bad --docs path (empty dir, typo, wrong
    path) prints one clear line instead of a deep traceback from whatever
    downstream stage first chokes on an empty chunk list."""

    try:
        return pipeline.ingest(docs, use_cache=not no_cache)
    except ValueError as e:
        typer.secho(f"\n{e}\n", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


def _prepare(docs: str, embedding_provider: str, persist_dir: str, no_cache: bool):
    from raglens.embedding import get_embedding_provider
    from raglens.pipeline import RagLensPipeline
    from raglens.vectorstore import ChromaStore

    pipeline = RagLensPipeline()
    all_chunks = _ingest_or_exit(pipeline, docs, no_cache)

    provider = get_embedding_provider(embedding_provider)
    store = ChromaStore(persist_directory=persist_dir)

    if store.count() == 0:
        embedding_chunks, _ = pipeline.build_index(all_chunks, provider, store)
    else:
        embedding_chunks = [c for c in all_chunks if c.chunk_type != "parent_section"]

    retrievers = pipeline.build_retrievers(all_chunks, embedding_chunks, provider, store)
    return pipeline, all_chunks, retrievers


@app.command()
def ingest(
    docs: str = typer.Option(..., help="Directory of PDFs to parse and chunk."),
    no_cache: bool = typer.Option(False, help="Ignore any existing parsed/chunk cache."),
):
    """Parse a PDF corpus into structure-aware chunks (Docling -> chunking).

    First step of the pipeline. Example:

      raglens ingest --docs ./my_pdfs
    """

    _require_full()
    from raglens.pipeline import RagLensPipeline

    pipeline = RagLensPipeline()
    chunks = _ingest_or_exit(pipeline, docs, no_cache)
    typer.echo(f"{len(chunks)} chunks created from {docs}")
    typer.echo("Next: raglens index --docs " + docs)


@app.command()
def index(
    docs: str = typer.Option(..., help="Directory of PDFs to parse and chunk."),
    embedding_provider: str = typer.Option("ollama", help="ollama | openai"),
    persist_dir: str = typer.Option("./chroma_db", help="ChromaDB persistence directory."),
    no_cache: bool = typer.Option(False, help="Ignore any existing parsed/chunk cache."),
):
    """Build the vector index (embed + store) for a PDF corpus.

    Second step, after `ingest`. Ollama is free/local (needs a running
    Ollama server); OpenAI needs OPENAI_API_KEY set. Example:

      raglens index --docs ./my_pdfs --embedding-provider ollama
    """

    _require_full()
    from raglens.embedding import get_embedding_provider
    from raglens.pipeline import RagLensPipeline
    from raglens.vectorstore import ChromaStore

    pipeline = RagLensPipeline()
    all_chunks = _ingest_or_exit(pipeline, docs, no_cache)

    provider = get_embedding_provider(embedding_provider)
    store = ChromaStore(persist_directory=persist_dir)
    embedding_chunks, chunk_embeddings = pipeline.build_index(all_chunks, provider, store)

    typer.echo(f"Indexed {len(chunk_embeddings)} chunks into {persist_dir}")
    typer.echo("Next: raglens questions --docs " + docs)


@app.command()
def questions(
    docs: str = typer.Option(..., help="Directory of PDFs (for chunks, via cache if already ingested)."),
    llm_provider: str = typer.Option("openai", help="openai | groq | ollama"),
    llm_model: str = typer.Option(None, help="Override the default model for llm_provider."),
    no_cache: bool = typer.Option(False, help="Ignore any existing parsed/chunk cache."),
):
    """Generate Q&A ground truth for every usable chunk (resumable).

    Third step. Safe to Ctrl-C and re-run -- already-completed chunks are
    skipped. Example:

      raglens questions --docs ./my_pdfs --llm-provider openai
    """

    _require_full()
    from raglens.llm import get_llm_provider
    from raglens.pipeline import RagLensPipeline
    from raglens.question_generation import QuestionGenerator

    pipeline = RagLensPipeline()
    all_chunks = _ingest_or_exit(pipeline, docs, no_cache)

    generator = QuestionGenerator(llm_provider=get_llm_provider(llm_provider, model=llm_model))
    samples = pipeline.generate_questions(all_chunks, generator=generator)
    typer.echo(f"{len(samples)} question samples available")
    typer.echo("Next: raglens benchmark --docs " + docs)


@app.command()
def benchmark(
    docs: str = typer.Option(..., help="Directory of PDFs (for chunks, via cache if already ingested)."),
    embedding_provider: str = typer.Option("ollama", help="ollama | openai"),
    persist_dir: str = typer.Option("./chroma_db", help="ChromaDB persistence directory."),
    output: str = typer.Option("data/benchmark_results.json", help="Where to write benchmark results JSON."),
    no_cache: bool = typer.Option(False, help="Ignore any existing parsed/chunk cache."),
):
    """Benchmark retrieval quality (Hit@K, MRR, NDCG) across all 5 strategies.

    Fourth step, needs `raglens questions` to have run first. Example:

      raglens benchmark --docs ./my_pdfs --embedding-provider ollama
    """

    _require_full()
    from raglens.question_generation import QuestionDatasetLoader

    pipeline, all_chunks, retrievers = _prepare(docs, embedding_provider, persist_dir, no_cache)
    samples = QuestionDatasetLoader().load()

    if not samples:
        typer.echo("No question samples found -- run `raglens questions` first.")
        raise typer.Exit(1)

    pipeline.benchmark_retrieval(samples, retrievers, cache_path=output)
    typer.echo(f"Benchmark results written to {output}")


@app.command()
def evaluate(
    docs: str = typer.Option(..., help="Directory of PDFs (for chunks, via cache if already ingested)."),
    embedding_provider: str = typer.Option("ollama", help="ollama | openai"),
    persist_dir: str = typer.Option("./chroma_db", help="ChromaDB persistence directory."),
    judge_provider: str = typer.Option("openai", help="openai | groq | ollama"),
    judge_model: str = typer.Option(None, help="Override the default judge model."),
    sample_size: int = typer.Option(150, help="Stratified sample size to score."),
    seed: int = typer.Option(42, help="Sampling seed (must match prior runs to resume correctly)."),
    retrievers: str = typer.Option(
        ",".join(ALL_RETRIEVERS), help="Comma-separated retriever names to evaluate."
    ),
    reranker: str = typer.Option(
        None,
        help="Optional: rerank each selected retriever's results before scoring "
             "(e.g. 'cross_encoder'). Scored under new '<name>_reranked' entries, "
             "so it never collides with un-reranked scores already in scores_path.",
    ),
    scores_path: str = typer.Option(None, help="Where RAGAS scores are cached/appended."),
    answer_cache_path: str = typer.Option(None, help="Where generated answers are cached/appended."),
    batch_size: int = typer.Option(10, help="Samples per judge-LLM batch (checkpointed after each)."),
    no_cache: bool = typer.Option(False, help="Ignore any existing parsed/chunk cache."),
):
    """Run RAGAS end-to-end evaluation. Fully resumable: safe to interrupt and re-run.

    Fifth step, needs `raglens questions` to have run first. This is the
    slow, judge-LLM-bound stage -- expect an upfront time estimate before
    anything is sent to the judge. Example:

      raglens evaluate --docs ./my_pdfs --embedding-provider ollama --judge-provider openai
    """

    _require_full()
    from raglens.question_generation import QuestionDatasetLoader

    pipeline, all_chunks, all_retrievers = _prepare(docs, embedding_provider, persist_dir, no_cache)
    samples = QuestionDatasetLoader().load()

    if not samples:
        typer.echo("No question samples found -- run `raglens questions` first.")
        raise typer.Exit(1)

    selected = {name: all_retrievers[name] for name in retrievers.split(",")}

    if reranker:
        from raglens.retrieval import RerankedRetriever, get_reranker_provider

        reranker_provider = get_reranker_provider(reranker)
        selected = {
            f"{name}_reranked": RerankedRetriever(r, reranker_provider, retriever_name=f"{name}_reranked")
            for name, r in selected.items()
        }
        typer.echo(f"Reranking enabled ({reranker}) -- scoring as: {', '.join(selected)}")

    _, target_chunk_ids = stratified_sample_by_document(samples, n=sample_size, seed=seed)
    status = RagasScorer(scores_path=scores_path).status(list(selected), target_chunk_ids)
    remaining = sum(s["remaining"] for s in status.values())
    if remaining:
        low_h = remaining * _MIN_MINUTES_PER_SAMPLE / 60
        high_h = remaining * _MAX_MINUTES_PER_SAMPLE / 60
        typer.echo(
            f"\n{remaining} samples still need scoring across {len(selected)} retriever(s). "
            f"Rough estimate: {low_h:.1f}-{high_h:.1f}h (judge-LLM latency varies a lot "
            f"with rate limits and time of day -- already-scored samples are skipped).\n"
        )
    else:
        typer.echo("\nAll selected retrievers are already fully scored for this sample.\n")

    judge_llm = get_ragas_judge_llm(judge_provider, judge_model)

    results = asyncio.run(
        pipeline.run_ragas_async(
            samples,
            selected,
            judge_llm,
            sample_size=sample_size,
            seed=seed,
            scores_path=scores_path,
            answer_cache_path=answer_cache_path,
            batch_size=batch_size,
        )
    )

    for name, scores in results.items():
        typer.echo(f"{name:15s} {scores}")

    typer.echo("\nNext: raglens report" + (f" --retrievers {','.join(selected)}" if reranker else ""))


@app.command()
def report(
    scores_path: str = typer.Option(None, help="Where RAGAS scores are read from."),
    retrievers: str = typer.Option(",".join(ALL_RETRIEVERS), help="Comma-separated retriever names."),
    output_csv: str = typer.Option("data/cache/ragas_results.csv", help="Where to write the comparison table."),
    output_chart: str = typer.Option(None, help="Where to write the comparison chart (PNG). Skipped if omitted."),
):
    """Aggregate whatever has been scored so far into a comparison table/chart.

    Works on a base install -- no `\\[full]` extra needed. If you used
    `evaluate --reranker`, pass the matching `<name>_reranked` names via
    --retrievers to include them. Example:

      raglens report --retrievers bm25,dense,hybrid,hybrid_reranked
    """

    scorer = RagasScorer(scores_path=scores_path)
    results = scorer.aggregate(retrievers.split(","))
    comparison = scorer.build_comparison_table(results)

    comparison.to_csv(output_csv)
    typer.echo(comparison.round(4).to_string())
    typer.echo(f"\nSaved -> {output_csv}")

    if output_chart:
        plot_ragas_scores(comparison, save_path=output_chart)
        typer.echo(f"Saved -> {output_chart}")


if __name__ == "__main__":
    app()
