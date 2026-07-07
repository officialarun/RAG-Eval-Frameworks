# Contributing

## Setup

```bash
git clone https://github.com/officialarun/RAG-Eval-Frameworks.git
cd RAG-Eval-Frameworks
uv sync --extra dev --extra full   # or: pip install -e ".[dev,full]"
cp .env.example .env               # fill in whichever provider(s) you'll use
```

`full` pulls in the ingest/parsing/indexing stack (Docling, ChromaDB, Ollama); without it you get the lean install (dashboard + `raglens report` only). `dev` adds `pytest`, `ruff`, and `build`.

## Running checks

```bash
uv run pytest tests/ -v
uv run ruff check raglens tests
```

Both must pass before opening a PR — they're also what CI runs (`.github/workflows/ci.yml`).

## Project layout

See the "Project Structure" section in `README.md` for what each `raglens/` subpackage is responsible for. A few conventions worth knowing before changing code:

- **Provider abstractions live in `raglens/embedding/` and `raglens/llm/`.** New embedding/LLM providers should implement the same interface those factories already return (`get_embedding_provider`, `get_llm_provider`) rather than special-casing call sites.
- **Retrievers accept an injectable `RetrievalConfig`** (`raglens/config/retrieval_config.py`), not module-level globals — don't reintroduce global mutable config.
- **Cache modules** (`raglens/cache/`, `raglens/question_generation/question_cache.py`, `raglens/ragas/answer_cache.py`) are resumable and keyed by `(chunk_id, retriever)` or similar — any change to their file formats must stay backward-compatible with existing cached data, since RAGAS scoring runs can take hours to days and users depend on being able to resume them.
- **`raglens/cli.py` keeps heavy imports (Docling, ChromaDB, Ollama) lazy inside each command function**, not at module level — this is what keeps `pip install raglens-toolkit` (no extras) usable for the dashboard/report path. Don't move those imports back to the top of the file.

## Tests

Unit tests in `tests/` cover pure logic only (retrieval metrics, `RetrievalConfig`, stratified sampling, `RagasScorer`'s resume/aggregate logic) — nothing that needs a live Ollama server or API key. If you're adding a new pure-logic component, add a test alongside it in the same style (no mocking frameworks, just plain fixtures/tmp_path).

## Commit messages / PRs

Explain the *why*, not just the *what* — especially for anything touching the resumable caches, since a subtle format change there can silently break someone's in-progress multi-day RAGAS run.
