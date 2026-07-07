# Changelog

All notable changes to this project are documented here. Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.2.1] - 2026-07-08

### Added
- `raglens --version` and `raglens info` (version, install status, next steps, links) — a first-time user shouldn't have to open the README to understand what they installed.
- Rich `--help` epilog pointing at `raglens info` and `raglens COMMAND --help`, flagging upfront that `ingest`/`index`/`questions`/`benchmark`/`evaluate` need the `[full]` extra and that the Streamlit dashboard is a separate command, not part of this CLI.
- `evaluate --reranker <name>` (e.g. `cross_encoder`) exposes the reranker via the CLI for the first time — scores under new `<name>_reranked` entries so it never collides with existing un-reranked scores.
- Friendly, actionable errors instead of raw tracebacks: running a `[full]`-only command without the extra installed now prints one line with the fix (`pip install raglens-toolkit[full]`) instead of a bare `ModuleNotFoundError`; pointing `ingest`/`index`/`questions`/`benchmark`/`evaluate` at a directory with no PDFs now fails fast with a clear message instead of crashing several frames deep in `ChunkAuditor` on an empty list.

### Fixed
- `raglens.__version__` was hardcoded and had already drifted (still said `0.1.0` after two releases) — now reads from installed package metadata (`importlib.metadata.version`), so it can't go stale again.

## [0.2.0] - 2026-07-08

### Added
- Cross-encoder reranker (`raglens/retrieval/reranker/`): `RerankerProvider` (Protocol, matching the embedding/LLM provider pattern), `CrossEncoderReranker` (local, free, `BAAI/bge-reranker-base` via `sentence-transformers`), and `RerankedRetriever`, which wraps any existing retriever (BM25/Dense/Hybrid/Hierarchical/Neighbor) with a `Retriever → Reranker → LLM` stage — opt-in, not part of the default retriever set. `sentence-transformers` added to the `[full]` extra.
- `Dockerfile` for the Hugging Face Spaces deployment (Docker SDK — HF's "New Space" UI no longer offers a native Streamlit card).
- Tag-triggered PyPI publish workflow (`.github/workflows/publish.yml`): pushing a `v*` tag now runs tests/lint, builds, and publishes via PyPI Trusted Publishing (OIDC, no stored token) automatically — replaces manual `twine upload`.
- `lean-install` CI job asserting `raglens.evaluation`/`raglens.ragas` still import correctly with zero extras installed.

### Fixed
- Dashboard crash on Hugging Face Spaces: `plot_benchmark_chart`/`plot_ragas_scores` unconditionally wrote a PNG to a hardcoded path that doesn't exist in the slim Docker image — `save_path` now accepts `None` to skip the disk write. Found live on the deployed Space, not by local testing (which always had the full repo present).
- CI: the initial workflow only installed `[dev]`, so `tests/test_bm25_retriever.py` (needs `rank_bm25`, a `[full]`-only dependency) failed on the very first real run — now installs `[dev,full]`.
- HF Spaces deploy: the repo's git history contains old binary files (pre-`.gitignore` `chroma_db/` artifacts) that HF's pre-receive hook rejects on a plain `git push`; deploys now push a single orphan commit containing only the files the Docker build needs.

### Changed
- README rewritten with current state (PyPI package, live HF Spaces demo, CLI, tests/CI) and restructured for both a first-time visitor and interview prep — added "From Research Pipeline to Open-Source Tool," "Bugs Found and Fixed," and "Engineering Decisions Worth Discussing" sections.
- `artifacts/pipeline_animation.svg` updated to match current state (multi-provider LLM, `FactualCorrectness` metric, in-progress RAGAS status, productized footer).

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
