import json
import os
from pathlib import Path


_DEFAULT_CACHE_DIR = "data/cache/question_generation"


def _cache_dir() -> Path:
    path = Path(
        os.getenv("RAGLENS_QUESTION_CACHE_DIR", _DEFAULT_CACHE_DIR)
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def append_record(
    record: dict
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

        file.write("\n")


def load_records():

    if not get_cache_file().exists():

        return []

    records = []

    with open(
        get_cache_file(),
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            records.append(
                json.loads(line)
            )

    return records


def get_completed_chunk_ids():

    return {
        record["chunk_id"]
        for record in load_records()
        if record.get("status")
        in {"success", "skip"}
    }


def cache_exists():

    return get_cache_file().exists()

def generation_summary():

    records = (
        load_records()
    )

    success = 0
    skip = 0
    failed = 0

    for record in records:

        status = (
            record.get(
                "status"
            )
        )

        if status == "success":

            success += 1

        elif status == "skip":

            skip += 1

        elif status == "failed":

            failed += 1

    total = (
        success
        + skip
        + failed
    )

    print()

    print("=" * 50)

    print(
        f"Success : {success}"
    )

    print(
        f"Skip    : {skip}"
    )

    print(
        f"Failed  : {failed}"
    )

    print(
        f"Total   : {total}"
    )

    print("=" * 50)

    print()

def clear_cache():

    cache_file = get_cache_file()

    if cache_file.exists():

        cache_file.unlink()

        print(
            "Cache cleared."
        )

    else:

        print(
            "Cache already empty."
        )

def get_cache_file() -> Path:

    return _cache_dir() / "generated_questions.jsonl"