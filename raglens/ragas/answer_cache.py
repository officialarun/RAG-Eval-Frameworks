import json
import os

from pathlib import Path


_DEFAULT_CACHE_DIR = "data/cache/generated_answers"


def _cache_dir() -> Path:
    path = Path(
        os.getenv("RAGLENS_ANSWER_CACHE_DIR", _DEFAULT_CACHE_DIR)
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_cache_file() -> Path:

    return _cache_dir() / "generated_answers.jsonl"


def load_completed_keys():

    cache_file = get_cache_file()

    if not cache_file.exists():

        return set()

    completed = set()

    with open(
        cache_file,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            record = json.loads(
                line
            )

            if (
                record["status"]
                == "success"
            ):

                completed.add(

                    (
                        record["chunk_id"],
                        record["retriever"]
                    )
                )

    return completed

def append_record(
    record
):

    with open(
        get_cache_file(),
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
        )

        file.write(
            "\n"
        )

def generation_summary():

    completed = 0

    cache_file = get_cache_file()

    if not cache_file.exists():

        return {
            "success": 0
        }

    with open(
        cache_file,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            record = json.loads(
                line
            )

            if (
                record["status"]
                == "success"
            ):
                completed += 1

    return {

        "success":
            completed
    }