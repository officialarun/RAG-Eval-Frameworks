from dataclasses import dataclass, field

from raglens.ragas.sampling import stratified_sample_by_document


@dataclass
class FakeSample:
    metadata: dict = field(default_factory=dict)


def _make_samples(doc_counts: dict[str, int]):
    samples = []
    for doc_num, count in doc_counts.items():
        for i in range(count):
            samples.append(FakeSample(metadata={"chunk_id": f"doc_{doc_num}_section_{i}"}))
    return samples


def test_stratified_sample_respects_requested_size():
    samples = _make_samples({"0": 50, "1": 50})
    eval_samples, target_chunk_ids = stratified_sample_by_document(samples, n=20, seed=42)
    assert len(eval_samples) == 20
    assert len(target_chunk_ids) == 20


def test_stratified_sample_is_deterministic_given_seed():
    samples = _make_samples({"0": 100, "1": 100, "2": 100})
    _, ids_a = stratified_sample_by_document(samples, n=30, seed=42)
    _, ids_b = stratified_sample_by_document(samples, n=30, seed=42)
    assert ids_a == ids_b


def test_stratified_sample_covers_every_document():
    samples = _make_samples({"0": 10, "1": 10, "2": 10})
    _, target_chunk_ids = stratified_sample_by_document(samples, n=15, seed=42)
    docs_represented = {cid.split("_")[1] for cid in target_chunk_ids}
    assert docs_represented == {"0", "1", "2"}


def test_stratified_sample_handles_n_larger_than_available():
    samples = _make_samples({"0": 5, "1": 5})
    eval_samples, target_chunk_ids = stratified_sample_by_document(samples, n=100, seed=42)
    assert len(eval_samples) == 10
    assert len(target_chunk_ids) == 10
