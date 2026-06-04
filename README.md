# RAG Evaluation Framework – Project Report

## 1. Project Overview

### Project Name

RAG Evaluation Framework

### Objective

The objective of this project was not merely to build a Retrieval-Augmented Generation (RAG) chatbot, but to develop a framework capable of evaluating the quality of a RAG pipeline using both traditional semantic similarity metrics and LLM-as-a-Judge evaluation techniques.

The framework allows:

* Automatic dataset generation
* Retrieval evaluation
* Answer quality evaluation
* Benchmarking of retrieval strategies
* Generation of evaluation reports

---

## 2. Motivation

Modern RAG systems often produce answers that appear correct, but there is no easy way to quantify:

* Retrieval quality
* Context relevance
* Answer correctness
* Hallucination rate

Most tutorials focus on building a chatbot but ignore evaluation.

This project addresses that gap by creating a complete evaluation workflow.

---

## 3. System Architecture

```
Dataset Sources
↓
Document Ingestion
↓
Chunking
↓
Embedding Generation
↓
Chroma Vector Database
↓
Retriever
↓
Qwen LLM
↓
Answer Generation
↓
Evaluation Metrics
↓
Report Generation
```

---

## 4. Technologies Used

### LLM

**Qwen3:8B**

Purpose:

* Dataset generation
* Answer generation
* LLM-as-a-Judge evaluation

---

### Embedding Model

**nomic-embed-text**

Purpose:

* Document embeddings
* Semantic similarity scoring

---

### Vector Database

**ChromaDB**

Purpose:

* Storage of document embeddings
* Semantic retrieval

---

### Framework

**LangChain**

Purpose:

* Document handling
* Retriever abstraction
* Model integration

---

### Language

**Python**

---

## 5. Data Collection

### Wikipedia Integration

Data was automatically fetched using Wikipedia APIs.

Topics included:

* Transformer Architecture
* Attention Mechanisms
* Large Language Models
* Retrieval Augmented Generation
* Vector Databases
* Prompt Engineering
* BERT

Advantages:

* Publicly available
* Large volume of content
* Continuously updated

Challenges:

* Contains equations
* Contains references
* Contains noisy formatting

---

## 6. Document Chunking

### Initial Configuration

```
chunk_size = 500
chunk_overlap = 100
```

Observed Problems:

* Definitions split across chunks
* Partial sentences retrieved
* Reduced retrieval quality

Example:

Retrieved chunk:

```
"Parallelizing attention"
```

instead of a complete explanation.

---

### Improved Configuration

```
chunk_size = 1000
chunk_overlap = 200
```

Benefits:

* More complete context
* Better retrieval quality
* Improved answer generation

---

## 7. Vector Store Construction

Process:

```
Documents
↓
Chunking
↓
Embeddings
↓
ChromaDB
```

Each chunk stored:

```json
{
  "chunk_id": "...",
  "title": "...",
  "source": "..."
}
```

This enabled traceability between generated questions and source content.

---

## 8. Synthetic Dataset Generation

Purpose:

Create evaluation benchmarks automatically.

Process:

```
Random Chunk
↓
Qwen
↓
Generate QA Pairs
↓
Store Dataset
```

Generated Dataset Structure:

```json
{
  "question": "...",
  "ground_truth": "...",
  "chunk_id": "...",
  "chunk_text": "..."
}
```

Result:

~300 synthetic QA pairs

---

## 9. Retrieval Strategies Evaluated

### Similarity Search

Returns:

Top-K nearest chunks

Advantages:

* Fast
* Simple

Disadvantages:

* Often returns highly similar chunks
* Lower diversity

---

### MMR Retrieval

Maximum Marginal Relevance

Advantages:

* More diverse context
* Reduced redundancy

Disadvantages:

* Sometimes retrieves less precise chunks

Observation:

MMR produced better context diversity than standard similarity search.

---

## 10. Evaluation Metrics

### Exact Chunk Recall

Definition:

Expected Chunk ID ∈ Retrieved Chunk IDs

Advantages:

* Easy to compute

Disadvantages:

* Extremely strict
* Does not measure semantic relevance

Observation:

Frequently returned 0 despite correct answers.

Conclusion:

Not suitable as a primary metric.

---

### Semantic Similarity

Method:

Ground Truth vs Generated Answer using embeddings.

Metric:

Cosine Similarity

Advantages:

* Fast
* Deterministic

Observed Average:

~0.63

Interpretation:

Answers were often semantically related but expressed differently.

---

### LLM-as-a-Judge

Method:

```
Question
Ground Truth
Generated Answer
↓
Qwen Evaluation
```

Score Range:

0 – 1

Observed Average:

~0.74

Advantages:

* Captures meaning
* More aligned with human judgment

Disadvantages:

* Slower
* Potential evaluation bias

---

## 11. Key Findings

### Finding 1

Exact chunk matching significantly underestimated retrieval quality.

**Reason:**

Relevant information often existed in multiple chunks.

---

### Finding 2

Good answers do not necessarily require retrieval of the original source chunk.

**Example:**

Expected Chunk = 0

Retrieved Chunk = 217

Answer still correct.

---

### Finding 3

Wikipedia chunk quality strongly influences retrieval performance.

Many chunks contained:

* Equations
* Formatting artifacts
* Partial sentences

This negatively impacted retrieval.

---

### Finding 4

Embedding similarity scores were consistently lower than LLM Judge scores.

**Interpretation:**

The model often generated correct answers using different wording.

---

## 12. Current Limitations

### Dataset Bias

Most generated questions originate from a limited number of documents.

Impact:

Reduced diversity.

---

### Retrieval Evaluation

Current evaluation does not fully measure contextual relevance.

A semantic context evaluation metric is needed.

---

### No Hybrid Search

Current system uses:

Vector Search Only

Missing:

* BM25
* Hybrid Retrieval

---

### No Hallucination Detection

System does not explicitly identify hallucinated answers.

---

## 13. Future Improvements

### Hybrid Retrieval

```
Vector Search
+
BM25
```

Expected Benefit:

Better keyword matching.

---

### Reranking

```
Retriever
↓
Cross Encoder Reranker
↓
Top Context
```

Expected Benefit:

Higher retrieval precision.

---

### Better Embedding Models

Evaluate:

* BGE Large
* BGE-M3
* MXBAI

Expected Benefit:

Improved semantic retrieval.

---

### Context Relevance Metric

```
Original Context
vs
Retrieved Context
```

Measure semantic overlap.

---

### Hallucination Detection

Evaluate whether generated content is supported by retrieved context.

---

### Interactive Dashboard

Potential stack:

* Streamlit
* Plotly

Features:

* Score visualization
* Retrieval comparison
* Benchmark reporting

---

## 14. Project Strengths

✔ End-to-End RAG Pipeline

✔ Automated Dataset Generation

✔ Chroma Vector Database Integration

✔ Semantic Evaluation

✔ LLM-as-a-Judge Evaluation

✔ Retrieval Benchmarking

✔ Report Generation

✔ Modular Architecture

---

## 15. Impact

This project demonstrates:

* Generative AI
* Retrieval-Augmented Generation
* Vector Databases
* Embeddings
* Evaluation Framework Design
* LangChain
* ChromaDB
* LLM Evaluation
* Synthetic Dataset Generation

Unlike a standard chatbot project, this project focuses on measuring and improving RAG quality, which is significantly closer to real-world GenAI engineering workflows.

---

## 16. Final Conclusion

The project successfully evolved from a simple RAG chatbot into a RAG Evaluation Framework capable of generating datasets, evaluating retrieval quality, scoring answer correctness, and benchmarking retrieval strategies.

The most important lesson learned was that retrieval quality cannot be reliably measured using exact chunk matching alone. Semantic evaluation and LLM-based judging provide a far more realistic assessment of RAG performance.

Future work should focus on hybrid retrieval, reranking, hallucination detection, and richer benchmarking datasets.
