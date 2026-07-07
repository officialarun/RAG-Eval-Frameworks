import os
import pickle
from pathlib import Path


_DEFAULT_CACHE_DIR = "data/cache"


def _cache_dir() -> Path:
    path = Path(
        os.getenv("RAGLENS_PARSER_CACHE_DIR", _DEFAULT_CACHE_DIR)
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_parsed_documents(
    parsed_documents
):

    path = (
        _cache_dir()
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
        _cache_dir()
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
        _cache_dir()
        / "parsed_documents.pkl"
    ).exists()