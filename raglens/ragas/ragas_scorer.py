import asyncio
import json
import os
import warnings
from pathlib import Path

import pandas as pd
from ragas import aevaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.run_config import RunConfig

from raglens.ragas.answer_cache import get_cache_file as get_answer_cache_file
from raglens.ragas.judge import get_default_metrics

SCORE_COLS = [
    "faithfulness",
    "factual_correctness(mode=f1)",
    "context_precision",
    "context_recall",
]

DISPLAY_NAMES = {
    "faithfulness": "Faithfulness",
    "factual_correctness(mode=f1)": "Factual Correctness",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
}

_DEFAULT_SCORES_PATH = "data/cache/ragas_scores.jsonl"


class RagasScorer:
    """Resumable, checkpointed RAGAS scoring across multiple retrievers.

    Ported unchanged in behavior from the notebook's batched evaluation loop:
    scores are appended to `scores_path` after every batch, keyed by
    (retriever, chunk_id), so an interrupted run resumes exactly where it
    left off -- already-scored pairs are never re-sent to the judge LLM.
    """

    def __init__(
        self,
        llm=None,
        metrics=None,
        scores_path: str | Path | None = None,
        answer_cache_path: str | Path | None = None,
        batch_size: int = 10,
        run_config: RunConfig | None = None,
    ):
        self.llm = llm
        self.metrics = metrics or get_default_metrics()
        self.scores_path = Path(
            scores_path
            or os.getenv("RAGLENS_RAGAS_SCORES_PATH", _DEFAULT_SCORES_PATH)
        )
        self.answer_cache_path = (
            Path(answer_cache_path) if answer_cache_path else get_answer_cache_file()
        )
        self.batch_size = batch_size
        self.run_config = run_config or RunConfig(
            timeout=1200, max_retries=3, max_workers=2
        )

    def _load_done_keys(self):
        if not self.scores_path.exists():
            return set()
        done = set()
        with open(self.scores_path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done.add((r["retriever"], r["chunk_id"]))
        return done

    def _load_pending_samples(self, retriever_name, chunk_ids, done_keys):
        pending = sorted(
            cid for cid in chunk_ids
            if (retriever_name, cid) not in done_keys
        )
        chunk_to_rec = {}
        with open(self.answer_cache_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if (rec.get("status") == "success"
                        and rec.get("retriever") == retriever_name
                        and rec.get("chunk_id") in set(pending)):
                    chunk_to_rec[rec["chunk_id"]] = rec

        ragas_samples, ordered_ids = [], []
        for cid in pending:
            if cid not in chunk_to_rec:
                continue
            r = chunk_to_rec[cid]
            ragas_samples.append(SingleTurnSample(
                user_input=str(r["question"]),
                retrieved_contexts=r["contexts"],
                response=str(r["generated_answer"]),
                reference=str(r["reference_answer"]),
            ))
            ordered_ids.append(cid)
        return ragas_samples, ordered_ids

    def _append_scores(self, retriever_name, chunk_ids, batch_df):
        self.scores_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.scores_path, "a", encoding="utf-8") as f:
            for i, chunk_id in enumerate(chunk_ids):
                row = batch_df.iloc[i]
                rec = {"retriever": retriever_name, "chunk_id": chunk_id}
                for col in SCORE_COLS:
                    val = row[col] if col in batch_df.columns else None
                    rec[col] = float(val) if (val is not None and pd.notna(val)) else None
                f.write(json.dumps(rec) + "\n")

    def status(self, retrievers, target_chunk_ids):
        """Report cached/remaining counts per retriever without scoring anything."""
        done_keys = self._load_done_keys()
        report = {}
        for retriever_name in retrievers:
            already_done = sum(
                1 for (r, c) in done_keys
                if r == retriever_name and c in target_chunk_ids
            )
            report[retriever_name] = {
                "cached": already_done,
                "total": len(target_chunk_ids),
                "remaining": len(target_chunk_ids) - already_done,
            }
        return report

    async def score_async(self, retrievers, target_chunk_ids):
        """Score all retrievers against target_chunk_ids, resuming from
        scores_path. Returns {retriever_name: {metric: mean_score}}."""

        done_keys = self._load_done_keys()
        all_results = {}

        for retriever_name in retrievers:
            already_done = sum(
                1 for (r, c) in done_keys
                if r == retriever_name and c in target_chunk_ids
            )
            pending_samples, pending_ids = self._load_pending_samples(
                retriever_name, target_chunk_ids, done_keys
            )

            print(f"\n{'='*60}")
            print(f"Retriever : {retriever_name}")
            print(f"Cached    : {already_done} / {len(target_chunk_ids)} | Remaining: {len(pending_samples)}")

            for batch_start in range(0, len(pending_samples), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(pending_samples))
                batch_samples = pending_samples[batch_start:batch_end]
                batch_ids = pending_ids[batch_start:batch_end]

                print(f"  [{batch_start+1:03d}:{batch_end:03d}] evaluating {len(batch_samples)} samples ...", flush=True)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", DeprecationWarning)
                        batch_result = await aevaluate(
                            dataset=EvaluationDataset(samples=batch_samples),
                            metrics=self.metrics,
                            llm=self.llm,
                            run_config=self.run_config,
                            show_progress=True,
                        )
                    batch_df = batch_result.to_pandas()
                    self._append_scores(retriever_name, batch_ids, batch_df)
                    done_keys.update((retriever_name, cid) for cid in batch_ids)
                    means = {col: round(batch_df[col].mean(), 4)
                             for col in SCORE_COLS if col in batch_df.columns}
                    print(f"    saved {len(batch_ids)} samples | means: {means}")
                except Exception as e:
                    print(f"    BATCH FAILED ({type(e).__name__}: {e}) — skipping, will retry on next run")

            all_results[retriever_name] = self._aggregate_one(retriever_name, target_chunk_ids)
            print(f"  Mean → {all_results[retriever_name]}")

        return all_results

    def _aggregate_one(self, retriever_name, target_chunk_ids=None):
        scores = {col: [] for col in SCORE_COLS}
        if not self.scores_path.exists():
            return {col: None for col in SCORE_COLS}
        with open(self.scores_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec["retriever"] != retriever_name:
                    continue
                if target_chunk_ids is not None and rec["chunk_id"] not in target_chunk_ids:
                    continue
                for col in SCORE_COLS:
                    val = rec.get(col)
                    if val is not None:
                        scores[col].append(val)

        return {
            col: round(sum(v) / len(v), 4) if v else None
            for col, v in scores.items()
        }

    def aggregate(self, retrievers, target_chunk_ids=None) -> dict:
        """Aggregate mean scores per retriever from scores_path, without
        scoring anything -- usable standalone (e.g. a `report` CLI command)
        even without a judge LLM configured."""
        return {name: self._aggregate_one(name, target_chunk_ids) for name in retrievers}

    def score(self, retrievers, target_chunk_ids):
        """Sync convenience wrapper around score_async. Not usable from inside
        an already-running event loop (e.g. Jupyter) -- use score_async there."""
        return asyncio.run(self.score_async(retrievers, target_chunk_ids))

    @staticmethod
    def build_comparison_table(all_results) -> pd.DataFrame:
        comparison = pd.DataFrame(all_results).T
        comparison.columns = [DISPLAY_NAMES.get(c, c) for c in comparison.columns]
        comparison.index.name = "Retriever"
        return comparison
