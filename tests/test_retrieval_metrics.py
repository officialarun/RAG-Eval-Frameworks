import math

from raglens.evaluation.retrieval_metrics import hit_at_k, ndcg_at_k, reciprocal_rank


def test_hit_at_k_found():
    assert hit_at_k("c2", ["c1", "c2", "c3"]) == 1


def test_hit_at_k_not_found():
    assert hit_at_k("c9", ["c1", "c2", "c3"]) == 0


def test_reciprocal_rank_first_position():
    assert reciprocal_rank("c1", ["c1", "c2", "c3"]) == 1.0


def test_reciprocal_rank_third_position():
    assert reciprocal_rank("c3", ["c1", "c2", "c3"]) == 1.0 / 3


def test_reciprocal_rank_not_found():
    assert reciprocal_rank("c9", ["c1", "c2", "c3"]) == 0.0


def test_ndcg_at_k_first_position():
    assert ndcg_at_k("c1", ["c1", "c2", "c3"]) == 1.0 / math.log2(2)


def test_ndcg_at_k_not_found():
    assert ndcg_at_k("c9", ["c1", "c2", "c3"]) == 0.0


def test_ndcg_decreases_with_rank():
    assert ndcg_at_k("c1", ["x", "c1", "y"]) < ndcg_at_k("c1", ["c1", "x", "y"])
