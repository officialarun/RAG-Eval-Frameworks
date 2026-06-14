import time


class HierarchicalRetrievalEvaluator:

    def evaluate(
        self,
        retriever,
        samples,
        k=5
    ):

        child_hits = 0
        section_hits = 0

        section_and_child = 0
        section_only = 0
        complete_miss = 0

        rr_scores = []
        latencies_ms = []

        total = len(samples)

        for sample in samples:

            gt_chunk = (
                sample.metadata["chunk_id"]
            )

            gt_section = (
                sample.metadata["section_title"]
            )

            # path is document-qualified (e.g. "lbdl > Introduction"),
            # so it avoids false positives when multiple documents share
            # the same section_title.
            gt_path = (
                sample.metadata.get("path")
                or gt_section
            )

            t0 = time.perf_counter()

            results = retriever.retrieve(
                sample.question,
                k=k
            )

            latencies_ms.append(
                (time.perf_counter() - t0) * 1000
            )

            retrieved_child_chunks = [
                r.child_chunk.chunk_id
                for r in results
            ]

            retrieved_section_paths = {
                r.parent_chunk.path
                for r in results
            }

            child_hit = (
                gt_chunk
                in retrieved_child_chunks
            )

            section_hit = (
                gt_path
                in retrieved_section_paths
            )

            if child_hit:
                child_hits += 1

            if section_hit:
                section_hits += 1

            if (
                child_hit
                and
                section_hit
            ):
                section_and_child += 1

            elif (
                section_hit
                and
                not child_hit
            ):
                section_only += 1

            elif (
                not section_hit
                and
                not child_hit
            ):
                complete_miss += 1

            rr = 0.0
            for rank, child_id in enumerate(
                retrieved_child_chunks,
                start=1
            ):
                if child_id == gt_chunk:
                    rr = 1.0 / rank
                    break

            rr_scores.append(rr)

        return {

            "questions":
                total,

            "child_hit_at_k":
                child_hits / total,

            "section_hit_at_k":
                section_hits / total,

            "section_and_child_hit":
                section_and_child / total,

            "section_only_hit":
                section_only / total,

            "complete_miss":
                complete_miss / total,

            "mrr":
                sum(rr_scores) / total,

            "avg_latency_ms":
                sum(latencies_ms) / total
        }
