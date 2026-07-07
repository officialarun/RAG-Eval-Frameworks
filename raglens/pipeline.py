from pathlib import Path

from raglens.cache import (
    chunks_exist,
    load_chunks,
    load_parsed_documents,
    parsed_documents_exist,
    save_chunks,
    save_parsed_documents,
)
from raglens.chunking import SectionChunker
from raglens.config import DEFAULT_CONFIG, RetrievalConfig
from raglens.embedding import EmbeddingGenerator
from raglens.evaluation import run_benchmark
from raglens.models.document import Document
from raglens.normalization import SectionFlattener
from raglens.parsers import (
    DoclingParser,
    HierarchyBuilder,
    LevelInference,
    MarkdownSectionParser,
)
from raglens.preprocessing import FormulaCleaner
from raglens.question_generation import (
    QuestionDatasetBuilder,
    QuestionDatasetLoader,
    QuestionGenerator,
)
from raglens.ragas import (
    AnswerDatasetBuilder,
    AnswerGenerator,
    EvaluationDatasetBuilder,
    RagasScorer,
    stratified_sample_by_document,
)
from raglens.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HierarchicalRetriever,
    HybridRetriever,
    NeighborHierarchicalRetriever,
)
from raglens.validation import ChunkAuditor
from raglens.vectorstore import ChromaStore


class RagLensPipeline:
    """End-to-end orchestration over your own PDF corpus.

    Mirrors the pipeline validated interactively in
    experiments/pipeline_validation.ipynb, extracted into reusable stages so
    it can be driven from the CLI or a script instead of cell-by-cell.
    """

    def __init__(self, config: RetrievalConfig = DEFAULT_CONFIG):
        self.config = config

    # ── Stage 1-2: ingest + structure-aware chunking ────────────────────────

    def ingest(self, pdf_dir: str, use_cache: bool = True) -> list:
        """Parse every PDF in pdf_dir into structure-aware Chunks.

        When use_cache is True, both the raw parsed markdown and the final
        chunk list are cached on disk -- a second call (e.g. from a later
        CLI invocation) skips Docling parsing entirely and returns the same
        Chunk objects via the chunk cache.
        """

        if use_cache and chunks_exist():
            return load_chunks()

        sources = [str(p) for p in Path(pdf_dir).glob("*.pdf")]

        if use_cache and parsed_documents_exist():
            all_markdowns = load_parsed_documents()
        else:
            docling = DoclingParser()
            all_markdowns = []
            for source in sources:
                try:
                    all_markdowns.append(docling.parse(source))
                except Exception as e:
                    print(f"Failed to parse {source}: {e}")
            if use_cache:
                save_parsed_documents(all_markdowns)

        cleaner = FormulaCleaner()
        cleaned_markdowns = [cleaner.process_markdown(md) for md in all_markdowns]

        documents = [
            Document(
                doc_id=f"doc_{idx}",
                title=Path(source).stem,
                source_type="pdf",
                source_path=source,
            )
            for idx, source in enumerate(sources)
        ]

        section_parser = MarkdownSectionParser()
        level_inference = LevelInference()
        hierarchy_builder = HierarchyBuilder()
        flattener = SectionFlattener()
        chunker = SectionChunker()

        all_chunks = []
        for document, markdown in zip(documents, cleaned_markdowns):
            parsed_doc = section_parser.parse(markdown, document)
            inferred_doc = level_inference.infer(parsed_doc)
            hierarchy_doc = hierarchy_builder.build(inferred_doc)
            flattened_sections = flattener.flatten(hierarchy_doc)
            all_chunks.extend(
                chunker.chunk(document=hierarchy_doc, sections=flattened_sections)
            )

        ChunkAuditor().audit(all_chunks)

        if use_cache:
            save_chunks(all_chunks)

        return all_chunks

    # ── Stage 3: embedding + vector index ───────────────────────────────────

    def build_index(self, all_chunks: list, embedding_provider, vector_store: ChromaStore):
        """Embed and index every non-parent_section chunk. Returns
        (embedding_chunks, chunk_embeddings) -- embedding_chunks is what
        BM25/Neighbor retrievers are built from downstream."""

        embedding_chunks = [c for c in all_chunks if c.chunk_type != "parent_section"]

        generator = EmbeddingGenerator(embedding_provider)
        chunk_embeddings = generator.generate(embedding_chunks)
        vector_store.add_embeddings(chunk_embeddings)

        return embedding_chunks, chunk_embeddings

    # ── Stage 4: retriever construction ─────────────────────────────────────

    def build_retrievers(
        self,
        all_chunks: list,
        embedding_chunks: list,
        embedding_provider,
        vector_store: ChromaStore,
    ) -> dict:
        """Build the five canonical retrievers used for benchmarking/RAGAS."""

        dense = DenseRetriever(embedding_model=embedding_provider, vector_store=vector_store, config=self.config)
        bm25 = BM25Retriever(embedding_chunks, config=self.config)
        hybrid = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25)
        hierarchical = HierarchicalRetriever(hybrid_retriever=hybrid, all_chunks=all_chunks)
        neighbor = NeighborHierarchicalRetriever(
            hierarchical_retriever=hierarchical, all_chunks=all_chunks, window=1
        )

        return {
            "bm25": bm25,
            "dense": dense,
            "hybrid": hybrid,
            "hierarchical": hierarchical,
            "neighbor": neighbor,
        }

    # ── Stage 5: Q&A ground-truth generation ────────────────────────────────

    def generate_questions(self, all_chunks: list, generator: QuestionGenerator | None = None):
        """Generate one Q&A pair per usable section_fragment chunk. Resumable
        via the on-disk question cache -- already-completed chunk_ids are
        skipped automatically."""

        usable_chunks = [
            c for c in all_chunks
            if c.chunk_type == "section_fragment"
            and not (self.config.exclude_reference_sections and self.config.is_bad_section(c.section_title))
            and len(c.content.strip()) >= 200
        ]

        QuestionDatasetBuilder(generator=generator).build(usable_chunks)
        return QuestionDatasetLoader().load()

    # ── Stage 6: retrieval benchmarking (Hit@K / MRR / NDCG) ────────────────

    def benchmark_retrieval(
        self,
        samples: list,
        retrievers: dict,
        k_values=(1, 3, 5, 10),
        cache_path: str = "experiments/data/benchmark_results.json",
    ):
        return run_benchmark(
            samples,
            retrievers["bm25"],
            retrievers["dense"],
            retrievers["hybrid"],
            retrievers["hierarchical"],
            retrievers["neighbor"],
            k_values=k_values,
            cache_path=cache_path,
        )

    # ── Stage 7: RAGAS end-to-end evaluation ────────────────────────────────

    async def run_ragas_async(
        self,
        samples: list,
        retrievers: dict,
        judge_llm,
        sample_size: int = 150,
        seed: int = 42,
        answer_generator: AnswerGenerator | None = None,
        metrics=None,
        scores_path: str | None = None,
        answer_cache_path: str | None = None,
        batch_size: int = 10,
    ):
        """Retrieve -> generate answers (cached/resumable) -> score with RAGAS
        (cached/resumable). Safe to interrupt and re-run at any point."""

        eval_samples, target_chunk_ids = stratified_sample_by_document(samples, n=sample_size, seed=seed)

        eval_builder = EvaluationDatasetBuilder()
        answer_builder = AnswerDatasetBuilder(answer_generator or AnswerGenerator())

        for name, retriever in retrievers.items():
            ragas_samples = eval_builder.build(eval_samples, retriever, name, k=5)
            answer_builder.build(ragas_samples)

        scorer = RagasScorer(
            llm=judge_llm,
            metrics=metrics,
            scores_path=scores_path,
            answer_cache_path=answer_cache_path,
            batch_size=batch_size,
        )

        return await scorer.score_async(list(retrievers.keys()), target_chunk_ids)
