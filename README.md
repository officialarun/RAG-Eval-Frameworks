---
title: raglens
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---
<!-- The block above is Hugging Face Spaces config metadata (read only when
this repo is pushed to a Space) -- see docs/deploy-hf-space.md. It's inert
everywhere else, including on GitHub. HF Spaces no longer offers a native
Streamlit SDK card in its "New Space" UI, so this runs via Dockerfile. -->

<p align="center">
  <img src="artifacts/pipeline_animation.svg" alt="Pipeline Workflow Animation" width="1100">
</p>

<h1 align="center">raglens</h1>

<p align="center">
  <a href="https://pypi.org/project/raglens-toolkit/"><img src="https://img.shields.io/pypi/v/raglens-toolkit?label=raglens-toolkit" alt="PyPI"></a>
  <a href="https://github.com/officialarun/RAG-Eval-Frameworks/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/officialarun/RAG-Eval-Frameworks/ci.yml?label=CI" alt="CI"></a>
  <a href="https://huggingface.co/spaces/Arun-56/raglens-demo"><img src="https://img.shields.io/badge/demo-live-brightgreen" alt="Live Demo"></a>
  <img src="https://img.shields.io/badge/python-3.12-blue" alt="Python 3.12">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="MIT License"></a>
</p>

<p align="center"><b>
  <a href="https://huggingface.co/spaces/Arun-56/raglens-demo">Live Demo</a> ·
  <a href="https://pypi.org/project/raglens-toolkit/">PyPI</a> ·
  <a href="#quickstart--run-this-on-your-own-corpus">Quickstart</a> ·
  <a href="#table-of-contents">Full Docs Below</a>
</b></p>

> A ground-up, provider-agnostic toolkit for building, benchmarking, and rigorously evaluating Retrieval-Augmented Generation systems — structure-aware PDF ingestion, five retrieval strategies, and RAGAS end-to-end evaluation, wrapped in a `pip install`-able CLI and a Streamlit dashboard.

## TL;DR

```bash
pip install raglens-toolkit[full]
raglens ingest --docs ./my_pdfs && raglens index --docs ./my_pdfs --embedding-provider ollama
```

That's the whole install-to-first-result path. No API keys needed if you use Ollama locally. Or skip installing anything and click through the **[live demo](https://huggingface.co/spaces/Arun-56/raglens-demo)** — it shows this project's own real benchmark and RAGAS numbers.

**What it is, in one paragraph:** most RAG tutorials measure "did the chatbot answer nicely." This project instead asks a narrower, harder question — *for a given PDF corpus, which retrieval strategy actually finds the right information, and does the final generated answer stay faithful to it?* — and answers it with a from-scratch pipeline (no LangChain retrieval abstractions, no off-the-shelf chunker) that's traceable from source chunk → generated question → retrieval result → RAGAS score at every step.

**Where it stands right now:** retrieval benchmarking across 5 strategies is complete (627 Q&A pairs, full results below). RAGAS end-to-end evaluation is in progress — 2 of 5 retrievers fully scored, the rest answer-generated and queued, resumable at any time (`raglens evaluate`).

---

## Table of Contents

- [Quickstart](#quickstart--run-this-on-your-own-corpus)
- [Why We Rebuilt (V1 → V2)](#why-we-rebuilt-v1--v2)
- [Pipeline Stages](#pipeline-stages)
- [Strategies at Each Stage](#strategies-at-each-stage) — chunking, retrieval, reranking, question generation design rationale
- [Benchmark Results](#benchmark-results)
- [RAGAS End-to-End Evaluation](#ragas-end-to-end-evaluation--in-progress)
- [From Research Pipeline to Open-Source Tool](#from-research-pipeline-to-open-source-tool) — the productization story
- [Bugs Found and Fixed](#bugs-found-and-fixed) — concrete war stories
- [Engineering Decisions Worth Discussing](#engineering-decisions-worth-discussing) — interview cheat-sheet
- [raglens Folder Stories](#raglens-folder-stories)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Version 1 Notes (Historical Context)](#version-1-notes-historical-context)

---

## Quickstart — Run This On Your Own Corpus

```bash
pip install raglens-toolkit[full]   # or: git clone + pip install -e ".[full]"
cp .env.example .env                # fill in whichever provider(s) you'll use

# 1. Parse + structure-aware chunk your PDFs
raglens ingest --docs ./my_pdfs

# 2. Embed + index (local & free via Ollama, or --embedding-provider openai)
raglens index --docs ./my_pdfs --embedding-provider ollama

# 3. Generate Q&A ground truth (resumable — safe to Ctrl-C and re-run)
raglens questions --docs ./my_pdfs --llm-provider openai

# 4. Benchmark retrieval quality (Hit@K, MRR, NDCG across 5 strategies)
raglens benchmark --docs ./my_pdfs --embedding-provider ollama

# 5. RAGAS end-to-end evaluation — judge-LLM-bound, checkpointed every batch
raglens evaluate --docs ./my_pdfs --embedding-provider ollama --judge-provider openai

# 6. Visualize everything
streamlit run dashboard/app.py
```

Every long-running stage (`questions`, `evaluate`) is resumable: it's keyed by `(chunk_id, retriever)` and skips whatever's already cached, so it's safe to run for a few hours a night and pick back up exactly where it left off. Supported providers today: **Ollama** (local, free) and **OpenAI** for embeddings; **Ollama, OpenAI, and Groq** for generation/judging.

`pip install raglens-toolkit` alone (no `[full]`) installs just enough for the dashboard and `raglens report` — the ingest/index/benchmark/evaluate pipeline needs the `[full]` extra (Docling, ChromaDB, Ollama client). See [From Research Pipeline to Open-Source Tool](#from-research-pipeline-to-open-source-tool) for why that split exists.

Want a public, zero-install link instead of running it locally? **[huggingface.co/spaces/Arun-56/raglens-demo](https://huggingface.co/spaces/Arun-56/raglens-demo)** — the dashboard falls back to bundled reference results automatically, no API keys or local corpus needed for a hosted demo. See `docs/deploy-hf-space.md` to deploy your own copy.

---

## Why We Rebuilt (V1 → V2)

Three root causes forced the rebuild from scratch.

**1. Naive chunking broke semantic units.**
`RecursiveCharacterTextSplitter` with a fixed size and overlap rolled a window across the raw document text, cutting across section boundaries, splitting tables mid-row, and bisecting LaTeX formulas. Chunks had no awareness of document structure — a paragraph could end in one chunk and its concluding sentence begin in the next.

**2. Wikipedia was the wrong corpus.**
Wikipedia articles are noisy: citation markers, redirect stubs, heavily cross-linked prose, and mathematical notation that does not survive plain-text extraction. The corpus introduced irrelevant matches that polluted retrieval results and made evaluation misleading.

**3. The evaluation methodology measured the wrong thing.**
Matching a retrieved chunk ID against one "expected" chunk from hundreds of candidates — when a question about transformers could reasonably be answered by many different chunks — produced near-zero recall that did not reflect actual retrieval quality. The metric was too strict in the wrong way.

**V2 addressed all three:** curated PDF corpus + Docling-based structure-aware parsing + section-aware chunking + ground truth generated from the exact source chunk so each question has one verifiable, traceable correct answer.

---

## Pipeline Stages

Eight stages transform raw PDFs into benchmark-ready retrieval results.

| # | Stage | Tool(s) | Input | Output |
|---|-------|---------|-------|--------|
| 1 | PDF Ingestion | Docling | Raw PDFs (13 docs) | Markdown with LaTeX formulae intact |
| 2 | Section Parsing + Hierarchy | MarkdownSectionParser → LevelInference → HierarchyBuilder | Markdown string | `Document` with nested `Section` tree |
| 3 | Normalization | SectionFlattener | Nested Section tree | `List[FlattenedSection]` with pre-computed breadcrumb paths |
| 4 | Section-Aware Chunking | SectionChunker + StructurePreserver | FlattenedSections | `List[Chunk]` — three types: `parent_section`, `table_fragment`, `section_fragment` |
| 5 | Embedding | EmbeddingGenerator (Ollama `nomic-embed-text` or OpenAI) | Chunks | `List[ChunkEmbedding]` |
| 6 | Vector Store | ChromaStore (ChromaDB) | ChunkEmbeddings | Persisted similarity index |
| 7 | Synthetic Q&A Generation | Configurable LLM (Ollama / OpenAI / Groq) | Chunks | 627 Q&A pairs in JSONL cache |
| 8 | Retrieval + Evaluation | 5 Retrievers + 3 Evaluators | QuestionSamples | Hit@K, MRR, NDCG, Section Hit, Latency |

**The data transformation story:** A raw PDF becomes a `Document` (root container with metadata). The document's heading structure is parsed into a nested `Section` tree. That tree is flattened into `List[FlattenedSection]` — each section pre-annotated with its full breadcrumb path (e.g. `"lbdl > Chapter 3 > Optimization"`). The chunker then transforms each section into one or more `Chunk` objects — the atomic unit for embedding and retrieval. Finally, each chunk is embedded into a vector and stored in ChromaDB.

---

## Strategies at Each Stage

### Chunking Strategies

The chunker applies a three-tier decision based on section size:

```
Section content length?
├── ≤ 2,500 chars
│   └── Single parent_section chunk  (fragment_index = -1, meaning "atomic, not split")
│
└── > 2,500 chars
    ├── Always create: parent_section chunk  (full section text — used by hierarchical retrieval)
    │
    └── Also create fragment children:
        │
        ├── Tables  (StructurePreserver)
        │   ├── Table ≤ 3,000 chars → single atomic table_fragment chunk
        │   └── Table > 3,000 chars → row-by-row table_fragment chunks
        │         └── Header Duplication: every fragment = header_row + separator_row + data_rows
        │             (a row of "| 0.75 | 0.83 |" is meaningless without its column names)
        │
        └── Remaining text → RecursiveCharacterTextSplitter (size=1,200, overlap=200)
              └── section_fragment chunks, fragment_index = 0, 1, 2 ...
                    └── Sequential fragment_index enables neighbor retrieval later
```

**Table-aware extraction** — Markdown tables are detected by the presence of `|` pipe characters. Small tables are preserved as a single atomic chunk (schema intact). Large tables are split at row boundaries, never mid-row, into ~1,500-character fragments.

**Header duplication** — When a large table is fragmented, every fragment gets the original header row and separator row prepended: `header + "\n" + separator + "\n" + data_rows`. This makes each fragment self-interpreting when retrieved in isolation.

**Formula protection** — LaTeX blocks (`$$...$$`) are replaced with `__FORMULA_N__` placeholders before text splitting and restored afterward. This prevents a formula from being cut mid-expression by the character-based splitter.

**Why not a sliding window?** — Fixed-size windows roll across the full document, crossing section boundaries. Section-aware chunking treats each section as the atomic semantic unit — small sections are never fragmented, and fragments never contain content from two different sections.

---

### Retrieval Strategies

Five strategies are implemented, each building on the previous:

| Strategy | Approach | Score | Key Design |
|----------|----------|-------|------------|
| BM25 | Lexical (Okapi BM25) | Raw, 0–∞ | In-memory; no vector store dependency |
| Dense | Semantic (nomic-embed-text) | Cosine similarity, 0–1 | Retrieves k×3 before filtering bad sections |
| Hybrid | Dense + BM25 fusion | Fused, 0–1 | Min-max normalization per retriever, then 50/50 weighted sum |
| Hierarchical | Hybrid → parent section context | Fused, 0–1 | Pre-built `(doc_id, section_id) → Chunk` dict for O(1) parent lookup |
| Hier+Neighbor | Hierarchical + fragment window | — | Expands to ±1 fragment siblings using `fragment_index` (section-local, never crosses boundaries) |

**Score normalization in Hybrid** — BM25 scores are unbounded [0, ∞); Dense scores are cosine similarity [0, 1]. Without normalization, BM25 magnitudes would dominate the fusion. Min-Max normalization applied independently per retriever: `(score − min) / (max − min)`, then `fused = 0.5 × dense_norm + 0.5 × bm25_norm`.

**Two flavours of neighbor retrieval** — `NeighborRetriever` uses global `chunk_order` (a monotonic counter) to find adjacent chunks; this can cross section boundaries. `NeighborHierarchicalRetriever` uses `fragment_index` (section-local counter) to expand only within the same section — the version used in benchmarking.

**Candidate over-retrieval** — All retrievers retrieve more candidates than k before filtering. Reference sections, bibliographies, and indices are excluded via an injectable `RetrievalConfig`. Over-retrieval guarantees k clean results remain after filtering.

---

### Reranking — an Optional Pipeline Stage, Not a 6th Strategy

`Retriever → Reranker → LLM`. Reranking isn't a peer alternative to BM25/Dense/Hybrid/etc. — it's a stage that wraps *any* of them: over-retrieve a candidate pool, score every candidate against the query with a cross-encoder (which sees the query and document together, unlike embedding-based retrieval which scores them independently), and keep only the reordered top-k. `RerankedRetriever` (`raglens/retrieval/reranker/`) composes with an existing retriever exactly the way `HybridRetriever` composes Dense+BM25:

```python
from raglens.retrieval import RerankedRetriever, get_reranker_provider

reranker = get_reranker_provider("cross_encoder")  # BAAI/bge-reranker-base, local, free
reranked_hybrid = RerankedRetriever(retrievers["hybrid"], reranker, candidate_k=25)
results = reranked_hybrid.retrieve("your query", k=5)
```

It's opt-in, not part of the default retriever set `raglens benchmark`/`evaluate` build — reranking adds real inference cost and a new dependency (`sentence-transformers`, in the `[full]` extra), so it shouldn't silently attach itself to every existing run.

**Why a `Protocol`, not a subclass of `BaseRetriever`.** A reranker is a *provider* (an interchangeable scoring backend), matching `get_embedding_provider`/`get_llm_provider`'s existing pattern, not a retriever. `RerankerProvider.rerank(query, documents) -> list[(index, score)]` stays domain-agnostic (raw strings in, ranked indices out) — it never sees `RetrievalResult`/`Chunk`; `RerankedRetriever` is the one place that maps back to domain objects, the same layering `EmbeddingGenerator` uses for `Chunk` ↔ vector. The `(index, score)` return shape (rather than a parallel list of scores) is deliberate too: it matches both a local cross-encoder's `predict()`-then-sort *and* how hosted rerank APIs (Cohere, Jina) already respond — so adding one of those later via `get_reranker_provider("cohere")` is a new branch in `raglens/retrieval/reranker/providers.py`, not an interface redesign.

---

### Question Generation Strategy

- **One Q&A per chunk** — avoids questions where multiple chunks give equivalent answers, which would make ground truth ambiguous.
- **Prompt engineering** — the generation prompt explicitly bans phrases like "According to the text" or "In this section," forcing standalone natural-language questions a real user might ask.
- **Skip logic** — chunks with insufficient content (very short fragments, context-free table rows) return `{"status": "skip"}` and are excluded from the evaluation dataset.
- **Resumable JSONL cache** — generation results are appended one record at a time to a JSONL file. If the process is interrupted, it resumes from the last completed chunk — no LLM calls are repeated, no data is lost.

---

## Benchmark Results

Results from `experiments/experiments/data/benchmark_results.json`, measured across 627 evaluation samples.

### Numbers

| Retriever | Type | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@5 | NDCG@5 | Section Hit@5 | Latency@5 |
|-----------|------|-------|-------|-------|--------|-------|--------|---------------|-----------|
| BM25 | Flat | 75.6% | 87.0% | 90.2% | 92.9% | 81.3% | 83.5% | — | 2.1 ms |
| Dense | Flat | 51.9% | 67.2% | 73.2% | 79.3% | 60.4% | 63.6% | — | 13.9 ms |
| Hybrid | Flat | 73.0% | 87.8% | 91.5% | 93.9% | 80.6% | 83.3% | — | 19.3 ms |
| Hierarchical | Hierarchical | — | 82.2% | 83.4% | 85.1% | 78.7% | — | 95.0% | 24.4 ms |
| Hier+Neighbor | Hier+Context | — | 87.8% | 89.4% | 91.3% | — | — | — | 23.0 ms |

![Benchmark Results Table](experiments/experiments/data/Retrieval_Benchmark_Reults.png)

![Benchmark Chart — Hit@K, MRR, Hierarchical Breakdown, Latency](experiments/experiments/data/benchmark_chart.png)

### Are These Numbers Good? — An Objective Assessment

**What the numbers show:**
- Hybrid at 91.5% Hit@5 means 9 out of 10 queries find the right chunk in the top 5 results.
- Hierarchical at 95.0% Section Hit@5 means the correct document section is retrieved almost every time.
- Hier+Neighbor at 89.4% vs Hierarchical at 83.4% (+6%) confirms that expanding the context window to neighboring fragments meaningfully recovers missed chunks.

**Why these numbers are optimistic — three honest caveats:**

**1. BM25 has a structural advantage in this benchmark.**
The ground truth questions were generated by an LLM *from* the source chunks. The LLM naturally uses terminology from the chunk text, so BM25 (keyword matching) is predisposed to score well. The 75.6% vs 51.9% Hit@1 gap (BM25 vs Dense) likely overstates BM25's real-world advantage — on natural user queries that don't mirror chunk vocabulary, the gap would be smaller.

**2. Dense retrieval underperforms its potential here.**
`nomic-embed-text` runs locally via Ollama on CPU. Frontier embedding models (`text-embedding-3-large`, `BGE-M3`, `MXBAI`) would likely close the gap with BM25 significantly. The 51.9% Hit@1 reflects this local model limitation, not a ceiling on semantic retrieval.

**3. Single ground truth per question underestimates retrieval quality.**
Each question maps to exactly one "correct" chunk ID. Many questions can be correctly answered by multiple semantically similar chunks in the corpus. When a retriever returns a different-but-equally-valid chunk, the evaluation marks it as a miss. Hit@K underestimates actual retrieval usefulness — a point first identified in EXP01 and still true in V2.

**Bottom line:** The retrieval layer is working correctly within this benchmark setup. These numbers confirm the pipeline is functioning and that the five strategies are meaningfully differentiated. RAGAS gives a more honest end-to-end assessment — it measures whether the final generated answer is actually correct, not whether a specific chunk ID was found.

---

## RAGAS End-to-End Evaluation — In Progress

RAGAS (Retrieval Augmented Generation Assessment) measures end-to-end RAG quality — not just whether the right chunk was retrieved, but whether the system produced a correct, faithful, relevant answer.

### The Four Metrics Actually Used

| Metric | Question it answers |
|--------|-------------------|
| **Faithfulness** | Does the generated answer stay within what the retrieved context actually says? Detects hallucination. |
| **Factual Correctness** | Is the generated answer factually aligned with the reference answer? (Used in place of Answer Relevancy — a better fit here since every question already has a verified reference answer.) |
| **Context Precision** | Of the retrieved chunks, what fraction are genuinely useful for answering the question? Scored with a custom `FastContextPrecision` metric that parallelizes ragas's default sequential per-context loop. |
| **Context Recall** | Does the retrieved context contain all the information needed to construct the correct answer? |

Judge LLM: `gpt-4o-mini` by default (configurable — `raglens evaluate --judge-provider ollama|openai|groq`).

### How It Runs

1. **Retrieve + generate answers** (`raglens.ragas.EvaluationDatasetBuilder` + `AnswerDatasetBuilder`) — for a stratified 150-question sample (proportional per source document, seed 42) × each retriever, retrieve top-5 chunks and generate an answer. Cached to JSONL, resumable.
2. **Score with RAGAS** (`raglens.ragas.RagasScorer`) — batches of 10 samples sent to the judge LLM, with results appended to a JSONL cache after every batch and keyed by `(retriever, chunk_id)`. An interrupted run — expected, since judge-LLM scoring runs at roughly 1-8 minutes per sample — resumes exactly where it left off; already-scored pairs are never re-sent.
3. **Aggregate** (`raglens report`) — mean scores per retriever → comparison table + chart.

### Status

| Retriever | Answers generated | RAGAS scored |
|-----------|:---:|:---:|
| BM25 | ✅ 150/150 | ✅ 150/150 |
| Dense | ✅ 150/150 | ✅ 150/150 |
| Hybrid | ✅ 150/150 (622/622 full set) | ⏳ 0/150 |
| Hierarchical | ✅ 150/150 | ⏳ 0/150 |
| Hier+Neighbor | ✅ 150/150 | ⏳ 0/150 |

Resume scoring with `raglens evaluate` — it picks up on `hybrid` next automatically. At an observed ~1-8 minutes per judge-LLM call, the CLI prints an upfront time estimate before committing to a run (450 samples remaining ≈ 15-60 hours, hence the resumable/checkpointed design rather than a single long-lived process).

### The Key Insight RAGAS Unlocks

Hit@K tells us *a chunk was retrieved*. RAGAS tells us *whether the LLM used that chunk to produce a correct, faithful answer*. A retriever that scores high on Hit@K may still produce hallucinations if its retrieved context is noisy or misaligned with the question. That is the gap this stage closes.

---

## From Research Pipeline to Open-Source Tool

Everything above was built to answer a research question. The pipeline that answers it, though, was originally just notebook cells (`experiments/pipeline_validation.ipynb`) — great for exploration, useless to anyone who isn't the original author. This section is about the second phase of the project: turning that notebook into something a stranger can `pip install` and point at their own PDFs.

**Provider abstraction.** Every LLM/embedding call used to be a hardcoded `OpenAI()` client instantiated inline. `raglens/embedding/providers.py` and `raglens/llm/providers.py` now expose `get_embedding_provider("ollama"|"openai")` and `get_llm_provider("ollama"|"openai"|"groq")` factories; every class that needs a model takes one as a constructor argument instead of building its own client. This is what makes the whole pipeline runnable for free (Ollama, local) or with better quality (OpenAI/Groq) with a single CLI flag.

**The CLI** (`raglens/cli.py`, built with Typer) — `ingest`, `index`, `questions`, `benchmark`, `evaluate`, `report` — is the notebook's cell-by-cell logic extracted into `RagLensPipeline` (`raglens/pipeline.py`) and driven from the command line. The expensive commands (`questions`, `evaluate`) inherit the same resumable-cache discipline the research pipeline already needed for multi-hour LLM calls, so interrupting a `raglens evaluate` run overnight and resuming it the next day is a first-class, tested behavior — not an afterthought.

**Dependency-extras split.** `pip install raglens-toolkit` alone installs a *lean* set (pandas, matplotlib, ragas, openai, groq, streamlit, typer) — enough for the dashboard and `raglens report`. Docling, ChromaDB, and Ollama's client (all needed only for `ingest`/`index`/`benchmark`/`evaluate`) live behind a `[full]` extra. This exists because a public deployment (PyPI, Hugging Face Spaces) shouldn't force a multi-GB install of a PDF-parsing/ML stack just to render a chart — and because `raglens/cli.py` lazily imports its heavy dependencies per-command rather than at module load, `raglens report` and `raglens --help` work correctly on the lean install alone. A dedicated CI job (`lean-install` in `.github/workflows/ci.yml`) asserts this stays true.

**Tests and CI.** 24 pytest tests cover the pure-logic surfaces that don't need a live LLM or vector store: retrieval metrics (`hit_at_k`, `reciprocal_rank`, `ndcg_at_k`), the injectable `RetrievalConfig`, stratified sampling, and `RagasScorer`'s resume/aggregate logic (verified directly against the real, live `ragas_scores.jsonl` cache, not a mock). GitHub Actions runs the full suite plus lint on every push, in two jobs — one with `[dev,full]` for correctness, one with zero extras to guard the lean-install promise above.

**Distribution.** The package is published on PyPI as [`raglens-toolkit`](https://pypi.org/project/raglens-toolkit/) (the plain `raglens` name was already taken by an unrelated package — the import path is unaffected, only the install name differs). The dashboard is deployed to Hugging Face Spaces at [huggingface.co/spaces/Arun-56/raglens-demo](https://huggingface.co/spaces/Arun-56/raglens-demo) via Docker (see `docs/deploy-hf-space.md`), with a demo-mode fallback (`dashboard/app.py`'s `_resolve()`) that serves this project's own bundled reference results (`dashboard/demo_data/`) whenever no local corpus or API keys are present — so the public demo needs neither.

---

## Bugs Found and Fixed

Concrete, specific things that broke and how they were diagnosed — the kind of detail that's easy to forget by the time an interview asks "tell me about a bug you fixed."

| Bug | Root cause | Fix |
|---|---|---|
| RAGAS entirely unimportable | `ragas==0.4.3` unconditionally imports `langchain_community.chat_models.vertexai`, a module removed in `langchain-community==0.4.2` — an upstream incompatibility, not project code | Pinned `langchain-community>=0.4.1,<0.4.2` in `pyproject.toml`; this would have silently blocked the very next attempt to resume RAGAS scoring had it gone unnoticed |
| `DenseRetriever` crashed on `embed_query` | The retriever called `.embed_query()` on whatever embedding object it was handed, but the original `OllamaEmbeddingGenerator` wrapper only implemented `.generate()` — callers worked around it by reaching into a private `.model` attribute | The new `EmbeddingGenerator` + `get_embedding_provider()` design returns a single object that implements both `embed_documents` (indexing) and `embed_query` (retrieval) correctly, closing the gap instead of working around it |
| Duplicate `similarity_search` method | `ChromaStore` had the same method defined twice; a first fix attempt accidentally kept both copies (just moved which one had a comment) | Caught properly on a second pass via `ruff` (`F811: redefinition of unused name`) — a good example of why lint in CI catches what manual review misses |
| Dead `WeightedRetriever` | An empty, 0-byte stub file with an `__init__.py` that imported a class that was never implemented — looked at first like a missing export, was actually never-finished code with zero references anywhere | Deleted outright rather than "fixed"; confirmed via repo-wide grep that nothing referenced it |
| Dashboard crashed on Hugging Face Spaces | `plot_benchmark_chart`/`plot_ragas_scores` unconditionally called `plt.savefig()` to a hardcoded path (`experiments/data/...`) that only exists in the full dev repo — the slim Docker image only ships `raglens/`, `dashboard/`, and bundled demo data | Found live, in production, by actually opening the deployed Space (not caught by local testing, which always had the full repo present) — `save_path` now accepts `None` to skip the disk write; the dashboard only ever needed the returned `Figure` for `st.pyplot()` |
| HF Space deploy rejected by a plain `git push` | This repo's git history has binary files (old `chroma_db/` artifacts) committed early on, before `.gitignore` excluded them — Hugging Face's pre-receive hook blocks any binary content not stored via their LFS/Xet system, and `git push` sends full history, not just the current tree | Deploy via a single orphan commit containing only the files the Docker build needs (`Dockerfile`, `raglens/`, `dashboard/`, `pyproject.toml`, `README.md`), force-pushed to the Space's `main` — sidesteps the history entirely rather than trying to scrub it |

---

## Engineering Decisions Worth Discussing

A dense, scannable list for interview prep — each is a decision that had a real alternative, not just "the obvious way to do it."

- **Why min-max normalize before fusing BM25 + Dense, not z-score or raw sum?** BM25 is unbounded; min-max guarantees both signals land in [0,1] before the 50/50 weighted sum, so neither can silently dominate regardless of corpus size. See [Retrieval Strategies](#strategies-at-each-stage).
- **Why is every cache keyed by `(chunk_id, retriever)` and append-only JSONL, not a database?** A crash mid-run (or an intentional Ctrl-C after a few hours) must lose zero completed work. Append-only JSONL means a completed record is durable the instant it's written; a real database would need transactions to get the same guarantee for no real benefit at this scale.
- **Why swap RAGAS's `AnswerRelevancy` for `FactualCorrectness`?** Every question already has a human-verifiable reference answer (traced back to its source chunk) — `FactualCorrectness` directly compares generated vs. reference answer, which is strictly more informative here than relevancy-to-question alone.
- **Why a custom `FastContextPrecision` metric?** RAGAS's default `ContextPrecision` scores each retrieved context sequentially; wrapping the same prompt/logic in `asyncio.gather` parallelizes it, cutting real wall-clock time on a metric that's otherwise O(k) LLM calls per sample.
- **Why lazy-import Docling/ChromaDB inside CLI command functions instead of at module top?** A module-level `from raglens.pipeline import RagLensPipeline` at the top of `cli.py` would force every command — including `raglens report`, which never touches the ingest pipeline — to pay Docling's (multi-GB, `transformers`/`torch`-pulling) import cost. Moving the import inside each command function keeps `pip install raglens-toolkit` (no extras) actually lean, and it's covered by a dedicated CI job rather than just claimed in a docstring.
- **Why does `raglens evaluate` print a time estimate before starting?** Judge-LLM latency for RAGAS scoring is inherently variable (rate limits, time of day) and can span hours to days for a few hundred samples — not something to silently discover after committing to a run. The estimate is deliberately a range (best/worst observed), not a false-precision single number.
- **Why deploy via an orphan git commit instead of cleaning the repo's history?** Rewriting history (`git filter-repo`, BFG) to remove old binary blobs would rewrite every commit hash on a repo with existing PRs/branches/tags — high blast radius for a problem that a single clean orphan commit solves without touching anything else.

---

## raglens Folder Stories

Every folder in `raglens/` has a specific responsibility. Here is the story of each one.

### `models/` — The Schema Layer

Pure dataclasses with no behavior. Defines the full data hierarchy:

```
Document
└── Section (nested tree, preserves heading hierarchy)
    └── FlattenedSection (tree → flat list, path pre-computed)
        └── Chunk (atomic unit for embedding + retrieval)
            └── ChunkEmbedding (Chunk + its dense vector)
```

Three result types exist because three retrieval paradigms return different shapes of data:
- `RetrievalResult` — flat retrievers (Dense, BM25, Hybrid): `chunk + score + retriever_name`
- `HierarchicalRetrievalResult` — Hierarchical retriever: `parent_chunk + child_chunk + score`
- `NeighborRetrievalResult` — NeighborRetriever: `center_chunk + neighbor_chunks + score`

Everything between pipeline stages is typed. No raw dicts are passed between components.

---

### `parsers/` — The Document Understanding Layer

Four classes form a sequential pipeline:

1. **DoclingParser** — converts a PDF file to a Markdown string using Docling with formula enrichment enabled. This preserves heading levels as `#`/`##` and LaTeX as `$$...$$`.
2. **MarkdownSectionParser** — regex-matches heading lines and builds a nested `Section` tree, assigning `parent_section_id` from a stack as it descends through heading levels.
3. **LevelInference** — infers section depth from title numbering patterns (e.g. "3.2.1 Introduction" → level 3) for documents that use numbered headings instead of Markdown heading levels.
4. **HierarchyBuilder** — validates and rebuilds parent-child relationships from section levels; corrects any inconsistencies introduced by the parser.

Together they transform raw PDF bytes into a typed, hierarchical `Document` object.

---

### `normalization/` — The Bridge Layer

`SectionFlattener` converts the nested `Section` tree into a flat `List[FlattenedSection]`. For each section it pre-computes the full breadcrumb path (e.g. `"lbdl > Chapter 3 > Stochastic Gradient Descent"`) and records `parent_section_id`.

This layer exists so chunking receives a simple flat list. The chunker never needs to traverse the tree — all the ancestry information is already embedded in each `FlattenedSection`.

---

### `chunking/` — The Fragmentation Layer

`SectionChunker` orchestrates the three-tier strategy (small section → atomic, large section → parent + fragments). `StructurePreserver` handles everything structure-sensitive: table detection, large-table fragmentation with header duplication, and formula placeholder substitution.

The key output fields on every `Chunk`:
- `chunk_type` — `parent_section`, `table_fragment`, or `section_fragment`
- `fragment_index` — `-1` for atomic parent sections; `0, 1, 2…` for ordered fragments
- `parent_section_id` — links every fragment back to its originating section

These three fields are what make hierarchical and neighbor retrieval possible downstream.

---

### `embedding/` — The Vectorization Layer

Stateless: takes a `Chunk`, returns a vector. `EmbeddingGenerator` wraps whatever LangChain `Embeddings` instance `get_embedding_provider("ollama" | "openai")` returns — the same instance is used for both indexing (`embed_documents`) and query-time retrieval (`embed_query`), so there's exactly one embedding object per run. Ollama is free/local/deterministic with no API keys; OpenAI is available for higher-quality embeddings.

---

### `llm/` — The Generation Layer

Mirrors `embedding/` for text generation. `get_llm_provider("ollama" | "openai" | "groq")` returns an object implementing a single `.generate(prompt) -> str` method; `QuestionGenerator` and `AnswerGenerator` take one as a constructor argument instead of instantiating their own client. The RAGAS judge LLM has its own, separate factory (`raglens/ragas/judge.py`) since it must satisfy `ragas`'s own LLM wrapper interface, not this simpler one.

---

### `vectorstore/` — The Index Layer

`ChromaStore` wraps ChromaDB. Each entry stores `(chunk_id, embedding, metadata_dict)` where the metadata dict contains all `Chunk` fields. This inline metadata storage means retrieved results can reconstruct full `Chunk` objects without a secondary database lookup — `ChromaStore.get_all_chunks()` uses exactly this to let the CLI rebuild retrievers from an existing index without re-parsing or re-embedding.

---

### `config/` — The Settings Layer

`RetrievalConfig` is an injectable dataclass (constructor-passed to `BM25Retriever`/`DenseRetriever`, defaulting to a shared `DEFAULT_CONFIG` instance) with three fields:
- `bad_sections` — section titles to exclude: `references`, `bibliography`, `contents`, `index`, `list of figures`, `list of tables`
- `exclude_reference_sections` — boolean flag
- `is_bad_section(title)` — normalizes title to lowercase alphanumeric, checks membership

All retrievers accept a `config` parameter. Without this filtering, BM25 retrieves bibliography entries because they contain topic-related keywords; Dense retrieves index pages because they share vocabulary with queries. Passing a custom `RetrievalConfig` lets a different corpus define its own excluded sections.

---

### `retrieval/` — The Strategy Layer

Five strategies arranged by composition depth:

```
BM25Retriever          ← in-memory lexical matching, no vector store
DenseRetriever         ← semantic matching via embedding + ChromaDB
HybridRetriever        ← composes Dense + BM25, normalizes + fuses scores
HierarchicalRetriever  ← wraps Hybrid, promotes results to parent section context
NeighborHierarchicalRetriever  ← wraps Hierarchical, expands to sibling fragments
```

`BM25Retriever` and `DenseRetriever` extend `BaseRetriever` (abstract class enforcing `retrieve(query, k)`). The remaining three compose by duck typing — they accept any object with a `retrieve()` method.

---

### `evaluation/` — The Metrics Layer

`retrieval_metrics.py` contains three pure functions: `hit_at_k`, `reciprocal_rank`, `ndcg_at_k`. No state, no side effects.

Three evaluator classes exist because each retriever returns a different result type:

| Evaluator | Used with | Result type | Metrics |
|-----------|-----------|-------------|---------|
| `RetrievalEvaluator` | Dense, BM25, Hybrid | `List[RetrievalResult]` | Hit@K, MRR, NDCG@K, Latency |
| `HierarchicalRetrievalEvaluator` | Hierarchical | `List[HierarchicalRetrievalResult]` | Child Hit@K, Section Hit@K, Sect+Child, MRR, Latency |
| `NeighborHierarchicalRetrievalEvaluator` | NeighborHierarchical | `List[Chunk]` | Child Hit@K, Latency |

`benchmark_runner.py` orchestrates all three evaluators across all five retrievers and every k value in one call; `benchmark_visualizer.py` renders the comparison table and 4-panel chart shown above.

---

### `question_generation/` — The Dataset Layer

Turns `List[Chunk]` into 627 evaluation Q&A pairs with full ground truth traceability.

- `QuestionGenerator` — wraps an injectable `LLMProvider` (Ollama/OpenAI/Groq, defaults to OpenAI); generates one Q&A per chunk with engineered prompts
- `QuestionDatasetBuilder` — orchestrates generation with resumability via `get_completed_chunk_ids()`
- `question_cache.py` — JSONL persistence; append-only so the file survives mid-run interruptions
- `QuestionDatasetLoader` — filters the cache to `status=success` records and returns `List[QuestionSample]`

---

### `ragas/` — The End-to-End Evaluation Layer

The productized version of the notebook's RAGAS cells:

- `EvaluationDatasetBuilder` + `AnswerGenerator` + `AnswerDatasetBuilder` — retrieve contexts and generate answers per `(retriever, question)` pair, cached and resumable
- `RagasScorer` — the batched, checkpointed scoring loop described in [RAGAS End-to-End Evaluation](#ragas-end-to-end-evaluation--in-progress); also exposes `.status()` and `.aggregate()` for reporting without triggering any new scoring
- `judge.py` — builds the ragas-compatible judge LLM per provider; `sampling.py` — the stratified-by-source-document sampling used to pick the 150-question evaluation subset
- `ragas_visualizer.py` — the per-metric bar chart shown by the dashboard

---

### `cache/` — The Persistence Layer

Three different caching concerns, one pattern (append-only where it matters):

- **Pickle (`parser_cache.py`)** — binary, fast, stores `Document` objects post-Docling. Docling parsing can take 30+ seconds per PDF; the cache avoids re-running it every session.
- **Pickle (`chunk_cache.py`)** — stores the full `List[Chunk]` after chunking, so the CLI's later stages (`index`, `benchmark`, `evaluate`) can skip re-parsing entirely if a corpus has already been ingested.
- **JSONL (`question_cache.py`, `answer_cache.py`)** — text, append-only, human-readable. LLM generation can fail mid-run; JSONL means each completed record is immediately safe. A full-file rewrite (as JSON would require) would lose data on a crash.

---

### `preprocessing/` — The Cleaning Layer

`formula_cleaner.py` pre-processes LaTeX content before it reaches Docling, removing or normalizing malformed formula syntax that would produce parsing artifacts in the rendered Markdown output.

---

### `validation/` — The Audit Layer

`ChunkAuditor` validates chunk integrity after chunking completes. It checks for empty chunks, missing IDs, malformed `fragment_index` values, and parent-child consistency. Catches issues before they silently corrupt the vector store.

---

### `pipeline.py` and `cli.py` — The Orchestration Layer

`RagLensPipeline` is the notebook's cell-by-cell logic (ingest → index → build retrievers → generate questions → benchmark → RAGAS-evaluate) as callable, testable methods. `cli.py` wraps each stage as a Typer command, lazily importing heavy dependencies per-command (see [From Research Pipeline to Open-Source Tool](#from-research-pipeline-to-open-source-tool)) so the package stays lean when the full pipeline isn't needed.

---

## Project Structure

```
rag-evaluation-framework/
│
├── data/
│   └── documents/               # Your PDF corpus (BYO — 13 curated GenAI/RAG PDFs used for the reference run)
│
├── experiments/
│   ├── pipeline_validation.ipynb   # Reference notebook — all stages run cell-by-cell
│   └── experiments/data/
│       ├── benchmark_results.json         # Saved benchmark numbers
│       ├── Retrieval_Benchmark_Reults.png # Pandas styled results table
│       └── benchmark_chart.png            # 4-panel comparison chart
│
├── raglens/                      # Core library (pip-installable as raglens-toolkit)
│   ├── models/                  # Dataclasses: Document, Section, Chunk, results
│   ├── parsers/                 # Docling → Markdown → Section tree
│   ├── normalization/           # Section tree → flat FlattenedSection list
│   ├── chunking/                # SectionChunker + StructurePreserver
│   ├── embedding/               # Ollama / OpenAI embedding providers + EmbeddingGenerator
│   ├── llm/                     # Ollama / OpenAI / Groq text-generation providers
│   ├── vectorstore/             # ChromaDB wrapper
│   ├── config/                  # Injectable RetrievalConfig (bad sections, top-k)
│   ├── retrieval/               # BM25, Dense, Hybrid, Hierarchical, Neighbor
│   ├── evaluation/              # Hit@K, MRR, NDCG, benchmark runner + charts
│   ├── question_generation/     # Q&A generation + JSONL cache
│   ├── ragas/                   # RagasScorer, judge factory, sampling, dataset builders
│   ├── cache/                   # Resumable caches (parsed docs, chunks, answers)
│   ├── preprocessing/           # Formula cleaner
│   ├── validation/              # Chunk auditor
│   ├── pipeline.py              # RagLensPipeline — orchestrates every stage
│   └── cli.py                   # `raglens` command (ingest/index/questions/benchmark/evaluate/report)
│
├── dashboard/
│   ├── app.py                   # Streamlit dashboard — visualizes CLI-produced results
│   └── demo_data/                # Bundled reference results for the public HF Spaces demo
│
├── tests/                       # pytest — pure-logic unit tests (metrics, config, sampling, scorer)
│
├── docs/
│   └── deploy-hf-space.md       # Hugging Face Spaces deployment guide
│
├── .github/workflows/ci.yml     # Test + lint + lean-install CI, on every push
├── Dockerfile                    # Hugging Face Spaces (Docker SDK) build
├── pyproject.toml                # base / [full] / [dev] dependency extras
├── LICENSE                       # MIT
├── CONTRIBUTING.md
├── CHANGELOG.md
│
├── src/                          # V1 (legacy, kept for historical reference — see "Why We Rebuilt")
│
└── artifacts/
    └── processtillnow.png       # V1 architecture diagram
```

---

## Tech Stack

| Component | Tool | Why |
|-----------|------|-----|
| PDF Parsing | [Docling](https://github.com/DS4SD/docling) | Formula-aware; preserves heading hierarchy from complex PDFs |
| LLM (Q&A gen / answers / judge) | Ollama, OpenAI, or Groq (your choice per-command) | Free local default via Ollama; cloud providers for quality/speed when needed |
| Embedding | Ollama (`nomic-embed-text`) or OpenAI | Local, deterministic, no cloud dependency — or OpenAI when you want it |
| Vector Store | ChromaDB | Lightweight, local persistence, no infrastructure required |
| Lexical Retrieval | rank-bm25 (Okapi BM25) | Fast in-memory BM25, no external service |
| Reranking (optional) | `sentence-transformers` CrossEncoder (`BAAI/bge-reranker-base`) | Local, free; wraps any retriever via `RerankedRetriever` |
| Text Splitting | LangChain `RecursiveCharacterTextSplitter` | Used only for text fragments within large sections |
| Evaluation Framework | RAGAS | Standard RAG evaluation library |
| CLI | Typer | `raglens` command: ingest/index/questions/benchmark/evaluate/report |
| Dashboard | Streamlit | Local + hosted (Hugging Face Spaces) visualization |
| Packaging | setuptools, [PyPI](https://pypi.org/project/raglens-toolkit/) | `pip install raglens-toolkit`, base/`[full]`/`[dev]` extras |
| CI | GitHub Actions | Test + lint + lean-install regression guard on every push |
| Deployment | Docker, Hugging Face Spaces | [Live demo](https://huggingface.co/spaces/Arun-56/raglens-demo), no API keys needed |
| Language | Python 3.12 | — |

---

## Version 1 Notes (Historical Context)

The following summarises the first iteration of this project, which used a LangChain-based pipeline on Wikipedia data. It is preserved here as context for the transition to V2.

- **Data source**: ~20 Wikipedia articles on ML/AI topics (Transformers, LLMs, Embeddings, Vector Databases, etc.) fetched via the Wikipedia API
- **Tech stack**: LangChain + ChromaDB + nomic-embed-text (same embedding model) + Qwen3:8B
- **Chunking**: `RecursiveCharacterTextSplitter` at chunk_size=1000, overlap=200
- **Retrieval evaluated**: Vector similarity search and BM25 (LangChain abstractions)
- **Evaluation metrics**: Exact Chunk Recall, Semantic Similarity (cosine ~0.63 average), LLM-as-a-Judge (~0.74 average)
- **Key finding**: Exact chunk recall severely underestimated retrieval quality — relevant answers came from chunks other than the "expected" source chunk
- **Key finding**: LLM-as-a-Judge scores correlated better with answer quality than embedding cosine similarity
- **Identified limitations**: No hybrid retrieval, no hallucination detection, Wikipedia noise, naive chunking, no section awareness
- **Outcome**: All identified limitations became the design requirements for V2
- **Reference links**: [Hallucination Detection Guide](https://oneuptime.com/blog/post/32026-01-30-hallucination-detection/view) · [LLM Eval Projects Guide](https://docs.google.com/document/d/1JrwCVhrDScXxZ_1l9s04TmSUj4BaptTU/mobilebasic)
