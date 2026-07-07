import json

from raglens.ragas.ragas_scorer import RagasScorer


def _write_scores(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_status_reports_zero_when_no_scores_file(tmp_path):
    scores_path = tmp_path / "scores.jsonl"
    scorer = RagasScorer(scores_path=scores_path)
    status = scorer.status(["bm25"], {"c1", "c2"})
    assert status["bm25"] == {"cached": 0, "total": 2, "remaining": 2}


def test_status_recognizes_already_scored_pairs(tmp_path):
    scores_path = tmp_path / "scores.jsonl"
    _write_scores(scores_path, [
        {"retriever": "bm25", "chunk_id": "c1", "faithfulness": 0.9},
        {"retriever": "bm25", "chunk_id": "c2", "faithfulness": 0.8},
        {"retriever": "dense", "chunk_id": "c1", "faithfulness": 0.7},
    ])
    scorer = RagasScorer(scores_path=scores_path)

    status = scorer.status(["bm25", "dense", "hybrid"], {"c1", "c2"})
    assert status["bm25"] == {"cached": 2, "total": 2, "remaining": 0}
    assert status["dense"] == {"cached": 1, "total": 2, "remaining": 1}
    assert status["hybrid"] == {"cached": 0, "total": 2, "remaining": 2}


def test_aggregate_computes_mean_per_retriever(tmp_path):
    scores_path = tmp_path / "scores.jsonl"
    _write_scores(scores_path, [
        {"retriever": "bm25", "chunk_id": "c1", "faithfulness": 1.0, "factual_correctness(mode=f1)": 0.5,
         "context_precision": 1.0, "context_recall": 1.0},
        {"retriever": "bm25", "chunk_id": "c2", "faithfulness": 0.5, "factual_correctness(mode=f1)": 0.5,
         "context_precision": 1.0, "context_recall": 0.5},
    ])
    scorer = RagasScorer(scores_path=scores_path)

    results = scorer.aggregate(["bm25", "dense"])
    assert results["bm25"]["faithfulness"] == 0.75
    assert results["dense"]["faithfulness"] is None


def test_aggregate_respects_target_chunk_ids_filter(tmp_path):
    scores_path = tmp_path / "scores.jsonl"
    _write_scores(scores_path, [
        {"retriever": "bm25", "chunk_id": "c1", "faithfulness": 1.0},
        {"retriever": "bm25", "chunk_id": "c2", "faithfulness": 0.0},
    ])
    scorer = RagasScorer(scores_path=scores_path)

    results = scorer.aggregate(["bm25"], target_chunk_ids={"c1"})
    assert results["bm25"]["faithfulness"] == 1.0
