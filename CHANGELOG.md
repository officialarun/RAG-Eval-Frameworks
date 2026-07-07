# Changelog

All notable changes to this project are documented here. Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.0] - 2026-07-07

### Added
- Rebuilt `src_v2` as `raglens` (PyPI distribution name `raglens-toolkit`): a pip-installable, provider-agnostic RAG benchmarking and RAGAS-evaluation toolkit.
- Embedding provider abstraction (`raglens.embedding`): Ollama or OpenAI, selectable per-run, one shared instance used for both indexing and query-time retrieval.
- LLM provider abstraction (`raglens.llm`): Ollama, OpenAI, or Groq for question generation and answer generation.
- `RagLensPipeline` (`raglens/pipeline.py`): the notebook's cell-by-cell orchestration (ingest, index, build retrievers, generate questions, benchmark, RAGAS-evaluate) extracted into reusable, testable stages.
- `RagasScorer` (`raglens/ragas/ragas_scorer.py`): the resumable, checkpointed RAGAS scoring loop, extracted from the reference notebook — verified to correctly resume an in-progress multi-retriever scoring run without re-scoring already-completed samples.
- `raglens` CLI (`ingest` / `index` / `questions` / `benchmark` / `evaluate` / `report`), with heavy dependencies (Docling, ChromaDB, Ollama) imported lazily per-command so a lean install still supports `report`.
- Local Streamlit dashboard (`dashboard/app.py`) visualizing retrieval-benchmark and RAGAS results.
- `RetrievalConfig` — bad-section filtering and default top-k are now an injectable dataclass instead of module-level globals.
- Dependency extras: base install is dashboard/report-only; `[full]` adds the ingest/parsing/indexing stack; `[dev]` adds test/lint/build tooling.
- Test suite (pytest, 24 tests) covering retrieval metrics, `RetrievalConfig`, stratified sampling, and `RagasScorer`'s resume/aggregate logic; GitHub Actions CI running lint + tests.
- `LICENSE` (MIT), `.env.example`, `CONTRIBUTING.md`.
- Upfront time estimate printed before `raglens evaluate` starts a long judge-LLM scoring run.

### Fixed
- Upstream `ragas`/`langchain-community` version incompatibility (`ragas==0.4.3` imports a module `langchain-community>=0.4.2` removed) that blocked RAGAS entirely.
- `DenseRetriever` required an `embed_query` method the embedding wrapper it was given didn't have; the new provider abstraction closes that gap directly.
- Duplicate `similarity_search` method in `ChromaStore`.
- Dead `WeightedRetriever` stub (empty file, never implemented, never referenced) removed.
- `plot_benchmark_chart` didn't return its `Figure`, which would have broken any caller (e.g. the dashboard) trying to reuse it programmatically.
