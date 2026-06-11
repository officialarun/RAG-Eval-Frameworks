## Chunking Analysis  : Initial Report Stats and findings

The previous pipeline (EXP01) used document-level recursive splitting,
producing approximately 600 chunks.

EXP02 introduced structure-aware hierarchical chunking.

Results:

- Total chunks: 146
- Parent sections: 14
- Standalone semantic sections: 26
- Section fragments: 106

Average chunk sizes:

| Type | Mean Length |
|--------|--------|
| Parent Section | 5611 |
| Section | 1244 |
| Section Fragment | 837 |

Key Findings:

- Document hierarchy was preserved.
- Semantic boundaries were respected.
- Parent-child relationships were retained.
- Chunk count reduced significantly compared to EXP01.
- Fragment lengths remained near the target chunk size.

EXP02-A Findings

Structure-aware chunking successfully preserved hierarchy
and produced semantically coherent chunks.

Retrieval audit showed strong performance for:
- Positional Encoding
- Multihead Attention
- Tokenization
- Speculative Decoding

Failures primarily stemmed from insufficient contextual
representation during embedding rather than chunking issues.

Next Experiment:
Embed hierarchical paths together with chunk content.


Finding:

Structure-aware chunking successfully preserved document hierarchy and semantic boundaries.

Retrieval analysis revealed that failures are primarily due to ranking ambiguity between semantically related sections (e.g., Decoder vs Decoder-only Terminology), rather than chunk quality.

Future improvements should focus on retrieval enhancements such as reranking, hybrid search, or parent-child retrieval rather than further chunking modifications.

✓ Wiki Parser
✓ Hierarchical Sections
✓ Structure-Aware Chunking
✓ Parent Chunks
✓ Path Metadata
✓ Vector Retrieval Sanity Check

## At the end of EXP02-PHASE-A
Chunking quality is no longer the bottleneck.
Retrieval ranking is the bottleneck. 