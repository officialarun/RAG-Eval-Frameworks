import asyncio

import typer

# Only base-dependency imports at module level -- `report` (and `--help`) must
# work with a lean `pip install raglens-toolkit`, without the `full` extra
# (docling/chromadb/langchain-ollama/...) that ingest/index/questions/
# benchmark/evaluate need. Those commands import their own heavy deps lazily.
from raglens.ragas import RagasScorer, get_ragas_judge_llm, plot_ragas_scores, stratified_sample_by_document

# Judge-LLM latency is inherently variable (rate limits, time of day, provider
# load) -- these are the best/worst-case minutes-per-sample observed across
# real runs of this project, used only to print a rough upfront estimate.
_MIN_MINUTES_PER_SAMPLE = 2
_MAX_MINUTES_PER_SAMPLE = 8

app = typer.Typer(help="raglens: benchmark and RAGAS-evaluate a RAG pipeline over your own PDF corpus.")

ALL_RETRIEVERS = ["bm25", "dense", "hybrid", "hierarchical", "neighbor"]


def _prepare(docs: str, embedding_provider: str, persist_dir: str, no_cache: bool):
    from raglens.embedding import get_embedding_provider
    from raglens.pipeline import RagLensPipeline
    from raglens.vectorstore import ChromaStore

    pipeline = RagLensPipeline()
    all_chunks = pipeline.ingest(docs, use_cache=not no_cache)

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
    """Parse a PDF corpus into structure-aware chunks (Docling -> chunking)."""

    from raglens.pipeline import RagLensPipeline

    pipeline = RagLensPipeline()
    chunks = pipeline.ingest(docs, use_cache=not no_cache)
    typer.echo(f"{len(chunks)} chunks created from {docs}")


@app.command()
def index(
    docs: str = typer.Option(..., help="Directory of PDFs to parse and chunk."),
    embedding_provider: str = typer.Option("ollama", help="ollama | openai"),
    persist_dir: str = typer.Option("./chroma_db", help="ChromaDB persistence directory."),
    no_cache: bool = typer.Option(False, help="Ignore any existing parsed/chunk cache."),
):
    """Build the vector index (embed + store) for a PDF corpus."""

    from raglens.embedding import get_embedding_provider
    from raglens.pipeline import RagLensPipeline
    from raglens.vectorstore import ChromaStore

    pipeline = RagLensPipeline()
    all_chunks = pipeline.ingest(docs, use_cache=not no_cache)

    provider = get_embedding_provider(embedding_provider)
    store = ChromaStore(persist_directory=persist_dir)
    embedding_chunks, chunk_embeddings = pipeline.build_index(all_chunks, provider, store)

    typer.echo(f"Indexed {len(chunk_embeddings)} chunks into {persist_dir}")


@app.command()
def questions(
    docs: str = typer.Option(..., help="Directory of PDFs (for chunks, via cache if already ingested)."),
    llm_provider: str = typer.Option("openai", help="openai | groq | ollama"),
    llm_model: str = typer.Option(None, help="Override the default model for llm_provider."),
    no_cache: bool = typer.Option(False, help="Ignore any existing parsed/chunk cache."),
):
    """Generate Q&A ground truth for every usable chunk (resumable)."""

    from raglens.llm import get_llm_provider
    from raglens.pipeline import RagLensPipeline
    from raglens.question_generation import QuestionGenerator

    pipeline = RagLensPipeline()
    all_chunks = pipeline.ingest(docs, use_cache=not no_cache)

    generator = QuestionGenerator(llm_provider=get_llm_provider(llm_provider, model=llm_model))
    samples = pipeline.generate_questions(all_chunks, generator=generator)
    typer.echo(f"{len(samples)} question samples available")


@app.command()
def benchmark(
    docs: str = typer.Option(..., help="Directory of PDFs (for chunks, via cache if already ingested)."),
    embedding_provider: str = typer.Option("ollama", help="ollama | openai"),
    persist_dir: str = typer.Option("./chroma_db", help="ChromaDB persistence directory."),
    output: str = typer.Option("data/benchmark_results.json", help="Where to write benchmark results JSON."),
    no_cache: bool = typer.Option(False, help="Ignore any existing parsed/chunk cache."),
):
    """Benchmark retrieval quality (Hit@K, MRR, NDCG) across all 5 strategies."""

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
    scores_path: str = typer.Option(None, help="Where RAGAS scores are cached/appended."),
    answer_cache_path: str = typer.Option(None, help="Where generated answers are cached/appended."),
    batch_size: int = typer.Option(10, help="Samples per judge-LLM batch (checkpointed after each)."),
    no_cache: bool = typer.Option(False, help="Ignore any existing parsed/chunk cache."),
):
    """Run RAGAS end-to-end evaluation. Fully resumable: safe to interrupt and re-run."""

    from raglens.question_generation import QuestionDatasetLoader

    pipeline, all_chunks, all_retrievers = _prepare(docs, embedding_provider, persist_dir, no_cache)
    samples = QuestionDatasetLoader().load()

    if not samples:
        typer.echo("No question samples found -- run `raglens questions` first.")
        raise typer.Exit(1)

    selected = {name: all_retrievers[name] for name in retrievers.split(",")}

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


@app.command()
def report(
    scores_path: str = typer.Option(None, help="Where RAGAS scores are read from."),
    retrievers: str = typer.Option(",".join(ALL_RETRIEVERS), help="Comma-separated retriever names."),
    output_csv: str = typer.Option("data/cache/ragas_results.csv", help="Where to write the comparison table."),
    output_chart: str = typer.Option(None, help="Where to write the comparison chart (PNG). Skipped if omitted."),
):
    """Aggregate whatever has been scored so far into a comparison table/chart."""

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
