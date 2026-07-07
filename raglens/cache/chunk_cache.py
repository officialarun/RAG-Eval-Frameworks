import os
import pickle
from pathlib import Path

_DEFAULT_CACHE_DIR = "data/cache"


def _cache_dir() -> Path:
    path = Path(
        os.getenv("RAGLENS_CHUNK_CACHE_DIR", _DEFAULT_CACHE_DIR)
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_chunks(chunks):
    path = _cache_dir() / "all_chunks.pkl"
    with open(path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved -> {path}")


def load_chunks():
    path = _cache_dir() / "all_chunks.pkl"
    with open(path, "rb") as f:
        chunks = pickle.load(f)
    print(f"Loaded -> {path}")
    return chunks


def chunks_exist():
    return (_cache_dir() / "all_chunks.pkl").exists()
