from ingestion import load_chunks
from langchain_ollama import ChatOllama
import textwrap
import json

# Initialize LLM
llm = ChatOllama(
    model="qwen3:8b",
    temperature=0
)

# Load chunks
chunks = load_chunks()

print(f"Loaded {len(chunks)} chunks")

# Store all QA pairs
dataset = []

# Start small for testing
for i, chunk in enumerate(chunks[:100]):

    chunk_text = chunk.page_content

    prompt = textwrap.dedent(f"""
    You are creating evaluation data for a RAG system.

    Generate EXACTLY 3 question-answer pairs from the text below.

    Text:
    {chunk_text}

    Return ONLY valid JSON.

    Example:

    [
      {{
        "question": "What is a transformer?",
        "answer": "A transformer is a neural network architecture based on attention."
      }},
      {{
        "question": "Why are transformers efficient?",
        "answer": "They allow parallel processing of tokens."
      }},
      {{
        "question": "What mechanism powers transformers?",
        "answer": "The attention mechanism."
      }}
    ]
    """)

    print(f"\nProcessing Chunk {i + 1}...")

    try:

        response = llm.invoke(prompt)

        content = response.content.strip()

        # Remove markdown json wrappers if present
        if content.startswith("```json"):
            content = content.replace("```json", "")
            content = content.replace("```", "")
            content = content.strip()

        qa_pairs = json.loads(content)

        for qa in qa_pairs:

            dataset.append({
                "question": qa["question"],
                "ground_truth": qa["answer"],
                "source": chunk.metadata.get("title"),
                "source_type": chunk.metadata.get("source"),
                "chunk_id": chunk.metadata["chunk_id"],
                "chunk_text": chunk.page_content
            })

        print(f"Added {len(qa_pairs)} QA pairs")

    except Exception as e:

        print(f"Failed on chunk {i + 1}")
        print(e)

print(f"\nTotal QA Pairs Generated: {len(dataset)}")

# Save dataset
with open(
    "../data/dataset/eval_dataset.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        dataset,
        f,
        indent=4,
        ensure_ascii=False
    )

print("\nDataset saved successfully!")
print("Location: data/dataset/eval_dataset.json")