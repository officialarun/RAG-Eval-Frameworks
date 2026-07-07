from collections import defaultdict


class NeighborHierarchicalRetriever:

    def __init__(
        self,
        hierarchical_retriever,
        all_chunks,
        window=1
    ):
        self.hierarchical_retriever = (
            hierarchical_retriever
        )

        self.window = window

        self.section_chunks = (
            defaultdict(list)
        )

        for chunk in all_chunks:

            if (
                chunk.fragment_index
                < 0
            ):
                continue

            self.section_chunks[
                chunk.parent_section_id
            ].append(chunk)

        for section_id in (
            self.section_chunks
        ):

            self.section_chunks[
                section_id
            ].sort(
                key=lambda c:
                c.fragment_index
            )

    def retrieve(
        self,
        query,
        k=5
    ):

        base_results = (
            self.hierarchical_retriever
            .retrieve(
                query,
                k=k
            )
        )

        expanded = {}
        
        for result in base_results:

            child = (
                result.child_chunk
            )

            section_chunks = (
                self.section_chunks[
                    child.parent_section_id
                ]
            )

            current_idx = (
                child.fragment_index
            )

            for neighbor in (
                section_chunks
            ):

                if abs(
                    neighbor.fragment_index
                    - current_idx
                ) <= self.window:

                    expanded[
                        neighbor.chunk_id
                    ] = neighbor

        return list(
            expanded.values()
        )