from collections import Counter

class ChunkAuditor:
    def audit(self, chunks):
        print("=" * 80)
        print("CHUNK AUDIT REPORT")
        print("=" * 80)

        self._chunk_type_stats(chunks)
        self._size_stats(chunks)
        self._empty_chunks(chunks)
        self._table_chunks(chunks)
        self._parent_links(chunks)

        print()
        print("=" * 80)
        print("AUDIT COMPLETE")
        print("=" * 80)

    def _chunk_type_stats(self, chunks):
        print()
        print("Chunk Types")
        print("-" * 40)

        stats = Counter(chunk.chunk_type for chunk in chunks)

        for k, v in stats.items():
            print(f"{k}: {v}")

    def _size_stats(self, chunks):
        sizes = [len(chunk.content) for chunk in chunks]

        print()
        print("Size Statistics")
        print("-" * 40)

        print("Min:", min(sizes))
        print("Max:", max(sizes))
        print(
            "Average:",
            round(sum(sizes) / len(sizes), 2),
        )

        giant = [c for c in chunks if len(c.content) > 5000]

        print("Chunks >5000 chars:", len(giant))

        giant_non_parent = [
            c for c in chunks if (c.chunk_type != "parent_section" and len(c.content) > 5000)
        ]

        print("Non-parent chunks >5000 chars:", len(giant_non_parent))

        print()
        print("Largest Chunks")
        print("-" * 40)

        for c in giant[:20]:
            print("=" * 80)
            print("Type:", c.chunk_type)
            print("Size:", len(c.content))
            print("Section:", c.section_title)
        
        for c in giant_non_parent:
            print("=" * 80)
            print("Type:", c.chunk_type)
            print("Size:", len(c.content))
            print("Section:", c.section_title)
            print("Path:", c.path)
            print()
            print(c.content[:1000])

        oversized = [c for c in chunks if c.chunk_type == "section_fragment" and len(c.content) > 3000]
        print(
            "Oversized section fragments:",
            len(oversized)
        )
        for c in oversized:
            print("=" * 80)
            print(len(c.content), c.section_title, c.path, c.content[:100])
            print()
        
        print("Checking new metrics for embedding suitability...")
        embedding_candidates = [c for c in chunks if c.chunk_type != "parent_section" and len(c.content) > 3000]
        max_chunk = max(embedding_candidates, key=lambda c: len(c.content))
        print(len(max_chunk.content), max_chunk.section_title, max_chunk.chunk_type)

    def _empty_chunks(self, chunks):
        bad = [c for c in chunks if len(c.content.strip()) == 0]

        print()
        print("Empty Chunks")
        print("-" * 40)
        print(len(bad))

    def _table_chunks(self, chunks):
        table_chunks = [c for c in chunks if c.chunk_type == "table_fragment"]

        print()
        print("Table Fragments")
        print("-" * 40)
        print("Count:", len(table_chunks))

        missing_metadata = []
        for chunk in table_chunks:
            if "table_id" not in chunk.metadata:
                missing_metadata.append(chunk)

        print("Missing Metadata:", len(missing_metadata))

    def _parent_links(self, chunks):
        bad = []
        for chunk in chunks:
            if chunk.parent_section_id is None and chunk.chunk_type != "parent_section":
                bad.append(chunk)

        print()
        print("Parent Link Validation")
        print("-" * 40)
        print("Broken Links:", len(bad))
