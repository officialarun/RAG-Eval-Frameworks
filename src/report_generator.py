import json


def load_results():

    with open(
        "../data/results/evaluation_results.json",
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def generate_report(results):

    total_questions = len(results)

    avg_embedding_score = sum(
        r["answer_similarity_score"]
        for r in results
    ) / total_questions

    avg_llm_score = sum(
        r["llm_answer_score"]
        for r in results
    ) / total_questions

    retrieval_success_rate = sum(
        r["exact_chunk_recall"]
        for r in results
    ) / total_questions

    best_questions = sorted(
        results,
        key=lambda x:
        x["llm_answer_score"],
        reverse=True
    )[:5]

    worst_questions = sorted(
        results,
        key=lambda x:
        x["llm_answer_score"]
    )[:5]

    report = {

        "total_questions":
            total_questions,

        "avg_embedding_score":
            round(
                avg_embedding_score,
                4
            ),

        "avg_llm_score":
            round(
                avg_llm_score,
                4
            ),

        "retrieval_success_rate":
            round(
                retrieval_success_rate,
                4
            ),

        "best_questions": [

            {
                "question":
                    q["question"],

                "llm_score":
                    q["llm_answer_score"]

            }

            for q in best_questions
        ],

        "worst_questions": [

            {
                "question":
                    q["question"],

                "llm_score":
                    q["llm_answer_score"]

            }

            for q in worst_questions
        ]
    }

    return report


def save_report(report):

    with open(
        "../data/results/final_report.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=4,
            ensure_ascii=False
        )


def main():

    results = load_results()

    report = generate_report(
        results
    )

    save_report(
        report
    )

    print("\nREPORT GENERATED\n")

    print(
        json.dumps(
            report,
            indent=4
        )
    )


if __name__ == "__main__":
    main()