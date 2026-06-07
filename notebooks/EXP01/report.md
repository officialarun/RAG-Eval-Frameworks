# EXP01 - Baseline RAG Retrieval Evaluation

## Experiment Overview

This experiment establishes the baseline performance of the current Retrieval-Augmented Generation (RAG) pipeline before introducing advanced chunking strategies, hybrid retrieval, reranking, or other retrieval optimizations.

The objective of this experiment was to evaluate how effectively the retriever can recover the source chunk used to generate evaluation questions.

---

# Experimental Setup

## Data Source

Wikipedia articles related to:

* Transformers
* Attention Mechanisms
* Large Language Models
* Deep Learning
* Machine Learning
* Embeddings
* Vector Databases
* Information Retrieval
* LangChain
* GPT Models
* BERT
* Llama
* Knowledge Graphs

A total of approximately 20 Wikipedia articles were collected.

---

## Chunking Strategy

### Chunker

RecursiveCharacterTextSplitter

### Parameters

| Parameter     | Value |
| ------------- | ----- |
| Chunk Size    | 1000  |
| Chunk Overlap | 200   |

---

## Corpus Statistics

| Metric               | Value           |
| -------------------- | --------------- |
| Total Chunks         | 862             |
| Minimum Chunk Length | 9 characters    |
| Maximum Chunk Length | 1000 characters |
| Average Chunk Length | 701 characters  |

---

## Embedding Model

nomic-embed-text

Generated using Ollama.

---

## Vector Database

ChromaDB

Persistence enabled.

---

## Retrieval Methods Evaluated

### 1. Vector Similarity Search

Configuration:

```python
search_type="similarity"
k=10
```

### 2. BM25 Retrieval

Configuration:

```python
BM25Retriever
k=10
```

---

# Evaluation Dataset

Synthetic evaluation questions were generated using Qwen.

Dataset structure:

```json
{
    "question": "...",
    "ground_truth": "...",
    "chunk_id": 0,
    "chunk_text": "..."
}
```

Question generation was performed chunk-wise.

---

## Evaluation Methodology

For every evaluation question:

1. Retrieve ranked chunks.
2. Locate the original chunk used to generate the question.
3. Record its position in the ranked retrieval results.
4. Compute retrieval metrics.

The experiment measures:

* Recall@1
* Recall@5
* Recall@10
* Mean Reciprocal Rank (MRR)
* Rank of Expected Chunk

---

# Results

## Vector Retrieval

| Metric       | Value    |
| ------------ | -------- |
| Recall@1     | 0.000000 |
| Recall@5     | 0.003378 |
| Recall@10    | 0.006757 |
| MRR          | 0.009345 |
| Average Rank | 40.83    |

---

## BM25 Retrieval

| Metric       | Value    |
| ------------ | -------- |
| Recall@1     | 0.003378 |
| Recall@5     | 0.013514 |
| Recall@10    | 0.023649 |
| MRR          | 0.014185 |
| Average Rank | 401.44   |

---

# Rank Distribution Analysis

Vector retrieval statistics:

| Metric             | Value |
| ------------------ | ----- |
| Count              | 64    |
| Mean Rank          | 40.83 |
| Standard Deviation | 24.73 |
| Minimum Rank       | 2     |
| 25th Percentile    | 20.25 |
| Median Rank        | 37    |
| 75th Percentile    | 55    |
| Maximum Rank       | 96    |

---

# Key Observations

## Observation 1

Vector retrieval significantly outperformed BM25.

Average rank:

```text
Vector : 40.83
BM25   : 401.44
```

Vector retrieval ranked relevant chunks approximately 10× higher than BM25.

---

## Observation 2

Despite outperforming BM25, vector retrieval did not consistently retrieve expected chunks within the top-k retrieval window.

Most expected chunks were ranked between:

```text
20 - 55
```

positions.

Since the production retriever currently uses:

```python
k = 5
```

many expected chunks never reach the LLM context window.

---

## Observation 3

The Recall@K metrics appear significantly lower than expected.

This may indicate limitations in the evaluation methodology rather than solely retrieval quality.

Example:

Question:

```text
What is a transformer?
```

Expected chunk:

```text
Chunk 0
```

Retrieved top chunk:

```text
Chunk 42
```

Although Chunk 42 contains relevant transformer-related information, the evaluation records the retrieval as a failure because it does not exactly match the source chunk.

---

## Observation 4

The evaluation currently assumes:

```text
One Question
        ↓
One Correct Chunk
```

However, many questions can be correctly answered by multiple semantically similar chunks.

Therefore the current evaluation likely underestimates actual retrieval usefulness.

---

# Limitations Identified

## Chunk Quality

Several chunks contain incomplete semantic information such as:

```text
Architecture
All transformers have the same primary components:
```

These chunks may not contain sufficient context to generate highly discriminative questions.

---

## Question Generation

Questions were generated directly from chunks.

This can produce:

* Generic questions
* Ambiguous questions
* Questions answerable by multiple chunks

which weakens exact-chunk retrieval evaluation.

---

## Evaluation Assumption

The current evaluation uses:

```text
Source Chunk Recovery
```

instead of:

```text
Answer Retrieval Quality
```

This distinction becomes important when multiple chunks contain overlapping information.

---

# Conclusions

The baseline experiment demonstrates that:

1. Vector retrieval substantially outperforms BM25 retrieval.
2. Retrieval is significantly better than random ranking.
3. Average expected chunk rank of approximately 41 suggests semantic information is being captured.
4. Current retrieval quality remains insufficient for top-5 production retrieval.
5. Chunking strategy and question-generation methodology appear to be the primary bottlenecks.

---

# Recommended Next Experiments

## EXP02

Smaller Chunk Size

```text
Chunk Size = 500
Overlap = 100
```

Objective:

Improve semantic coherence of chunks.

---

## EXP03

Semantic Chunking

Objective:

Split documents based on semantic boundaries rather than character counts.

---

## EXP04

Hybrid Retrieval

Combine:

* Vector Similarity
* BM25

Objective:

Leverage both lexical and semantic matching.

---

## EXP05

Question Quality Filtering

Generate questions only from chunks containing sufficient semantic information.

Potential approaches:

* Minimum chunk length threshold
* LLM-based chunk quality scoring
* Context-aware question generation

---

# Final Takeaway

The primary insight from EXP01 is that retrieval quality is not the sole limitation. Chunk granularity and question-generation methodology have a significant impact on evaluation outcomes and should be investigated before introducing more advanced retrieval architectures.
