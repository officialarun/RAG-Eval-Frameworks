import random
from collections import defaultdict


def stratified_sample_by_document(samples, n: int, seed: int = 42):
    """Proportionally sample n items from `samples`, stratified by source
    document (parsed from the `doc_N_...` chunk_id convention), so every
    document contributes roughly its fair share to the evaluation subset.

    Returns (eval_samples, target_chunk_ids).
    """

    rng = random.Random(seed)

    doc_buckets = defaultdict(list)
    for i, s in enumerate(samples):
        doc_num = s.metadata["chunk_id"].split("_", 2)[1]
        doc_buckets[doc_num].append(i)

    eval_indices = []
    for doc_num, indices in sorted(doc_buckets.items(), key=lambda x: int(x[0])):
        k = max(1, round(len(indices) / len(samples) * n))
        eval_indices.extend(rng.sample(indices, min(k, len(indices))))

    if len(eval_indices) > n:
        eval_indices = rng.sample(eval_indices, n)
    elif len(eval_indices) < n:
        pool = [i for i in range(len(samples)) if i not in set(eval_indices)]
        eval_indices.extend(rng.sample(pool, min(n - len(eval_indices), len(pool))))

    eval_indices = sorted(eval_indices)
    eval_samples = [samples[i] for i in eval_indices]
    target_chunk_ids = {s.metadata["chunk_id"] for s in eval_samples}

    return eval_samples, target_chunk_ids
