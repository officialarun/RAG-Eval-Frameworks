import pickle
from pathlib import Path


CACHE_DIR = Path(
    "data/cache"
)

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def save_parsed_documents(
    parsed_documents
):

    path = (
        CACHE_DIR
        / "parsed_documents.pkl"
    )

    with open(
        path,
        "wb"
    ) as file:

        pickle.dump(
            parsed_documents,
            file
        )

    print(
        f"Saved -> {path}"
    )


def load_parsed_documents():

    path = (
        CACHE_DIR
        / "parsed_documents.pkl"
    )

    with open(
        path,
        "rb"
    ) as file:

        parsed_documents = (
            pickle.load(
                file
            )
        )

    print(
        f"Loaded -> {path}"
    )

    return parsed_documents


def parsed_documents_exist():

    return (
        CACHE_DIR
        / "parsed_documents.pkl"
    ).exists()