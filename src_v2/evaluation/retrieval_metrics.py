import math


def hit_at_k(
    ground_truth_chunk_id: str,
    retrieved_chunk_ids: list[str]
) -> int:

    return int(
        ground_truth_chunk_id
        in retrieved_chunk_ids
    )


def reciprocal_rank(
    ground_truth_chunk_id: str,
    retrieved_chunk_ids: list[str]
) -> float:

    for rank, chunk_id in enumerate(
        retrieved_chunk_ids,
        start=1
    ):

        if (
            chunk_id
            == ground_truth_chunk_id
        ):

            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    ground_truth_chunk_id: str,
    retrieved_chunk_ids: list[str]
) -> float:
    """NDCG with single relevant doc: DCG / IDCG = (1/log2(rank+1)) / 1.0"""

    for rank, chunk_id in enumerate(
        retrieved_chunk_ids,
        start=1
    ):

        if (
            chunk_id
            == ground_truth_chunk_id
        ):

            return 1.0 / math.log2(rank + 1)

    return 0.0
