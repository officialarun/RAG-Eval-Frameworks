import json
from pathlib import Path


CACHE_DIR = Path(
    "data/cache/question_generation"
)

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CACHE_FILE = (
    CACHE_DIR
    / "generated_questions.jsonl"
)


def append_record(
    record: dict
):

    with open(
        CACHE_FILE,
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

    if not CACHE_FILE.exists():

        return []

    records = []

    with open(
        CACHE_FILE,
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

    return CACHE_FILE.exists()

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

    if CACHE_FILE.exists():

        CACHE_FILE.unlink()

        print(
            "Cache cleared."
        )

    else:

        print(
            "Cache already empty."
        )

def get_cache_file():

    return CACHE_FILE