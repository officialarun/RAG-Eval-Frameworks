import sys
from unittest.mock import MagicMock

sys.modules['langchain_community.chat_models.vertexai'] = MagicMock()

import os
import json
import pandas as pd

from langchain_ollama import ChatOllama
from ragas.llms import LangchainLLMWrapper
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

from rag_pipeline import ask


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

dataset_path = os.path.join(
    BASE_DIR,
    "data",
    "dataset",
    "eval_dataset.json"
)

with open(dataset_path, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

# reducing eval size for test purposes
eval_data = eval_data[:2]

questions = []
answers = []
contexts = []
ground_truths = []

evaluator_llm = LangchainLLMWrapper(
    ChatOllama(
        model="qwen3:8b"
    )
)


for item in eval_data:

    question = item["question"]

    print(f"\nEvaluating: {question}")

    result = ask(question)

    questions.append(question)

    answers.append(result["answer"])

    contexts.append(
        [
            doc.page_content
            for doc in result["retrieved_docs"]
        ]
    )

    ground_truths.append(
        item["ground_truth"]
    )

ragas_dataset = Dataset.from_dict(
    {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
)

answers_df = pd.DataFrame({
    "question": questions,
    "answer": answers,
    "ground_truth": ground_truths
})

answers_df.to_csv(
    os.path.join(
        BASE_DIR,
        "data",
        "dataset",
        "generated_answers.csv"
    ),
    index=False
)

print("Answers saved")

faithfulness.llm = evaluator_llm
answer_relevancy.llm = evaluator_llm

print("Starting RAGAS evaluation")

print("Questions:", len(questions))
print("Answers:", len(answers))
print("Contexts:", len(contexts))
print("Ground Truths:", len(ground_truths))

result = evaluate(
    ragas_dataset,
    metrics=[
        faithfulness,
    ]
)
print("Evaluation finished")
print(type(result))
print(repr(result))

print(result)