import json
import os

from rag_pipeline import ask

from metrics.semantic_similarity import SemanticSimilarity
from metrics.llm_judge import LLMJudge


def load_dataset():
    dataset_path = "../data/dataset/eval_dataset.json"

    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        print(f"Loaded {len(dataset)} questions")
        return dataset

    except Exception as e:
        print(f"Dataset Load Error: {e}")
        return []


def evaluate_dataset(dataset, limit=20):
    semantic_evaluator = SemanticSimilarity()
    llm_judge = LLMJudge()
    results = []

    for idx, sample in enumerate(dataset[:limit]):
        print("\n" + "=" * 60)
        print(f"Question {idx + 1}/{limit}")
        print("=" * 60)

        try:
            question = sample["question"]
            ground_truth = sample["ground_truth"]
            expected_chunk_id = sample["chunk_id"]

            print(f"Question: {question}")

            rag_result = ask(question)
            generated_answer = rag_result["answer"]
            retrieved_chunk_ids = rag_result["retrieved_chunk_ids"]
            retrieved_context = "\n\n".join(
                doc.page_content for doc in rag_result["retrieved_docs"]
            )

            # Exact Chunk Recall
            exact_chunk_recall = int(expected_chunk_id in retrieved_chunk_ids)

            # Embedding Similarity
            answer_similarity_score = semantic_evaluator.score(
                ground_truth, generated_answer
            )

            # LLM Judge Score
            llm_answer_score = llm_judge.score_answer(
                question, ground_truth, generated_answer
            )

            result = {
                "question": question,
                "ground_truth": ground_truth,
                "generated_answer": generated_answer,
                "expected_chunk_id": expected_chunk_id,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "exact_chunk_recall": exact_chunk_recall,
                "answer_similarity_score": answer_similarity_score,
                "llm_answer_score": llm_answer_score,
                "retrieved_context": retrieved_context,
            }

            results.append(result)

            print(f"Embedding Score: {answer_similarity_score}")
            print(f"LLM Judge Score: {llm_answer_score}")

        except Exception as e:
            print(f"Failed Question {idx + 1}")
            print(e)

    return results


def save_results(results):
    os.makedirs("../data/results", exist_ok=True)
    output_file = "../data/results/evaluation_results.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\nSaved {len(results)} results")
    print(f"Location: {output_file}")


def main():
    dataset = load_dataset()
    if not dataset:
        return

    results = evaluate_dataset(dataset, limit=20)
    save_results(results)

    if results:
        avg_embedding = sum(r["answer_similarity_score"] for r in results) / len(results)
        avg_llm = sum(r["llm_answer_score"] for r in results) / len(results)
        print(f"\nAverage Embedding Score: {avg_embedding:.4f}")
        print(f"Average LLM Judge Score: {avg_llm:.4f}")
    else:
        print("\nNo results to compute averages.")


if __name__ == "__main__":
    main()
