"""
Phase 1 - Step 1: vectorize messy listing titles into dense semantic embeddings.

Uses sentence-transformers (default: all-MiniLM-L6-v2). Embeddings are
L2-normalized so downstream Euclidean k-means behaves like cosine similarity.

Cache is **per title**, keyed by sha1(title), in one pickle. A run that adds a
handful of new listings only encodes those few - it doesn't reload torch and
re-embed the whole corpus (which made every scan take ~a minute). Only a genuine
cache miss pays the model-load cost.
"""
from __future__ import annotations

import hashlib
import os
import pickle

import numpy as np

from src.ml.config import ml_config

CACHE_DIR = "data/processed"


def _cache_path(model_name: str) -> str:
    return os.path.join(CACHE_DIR, f"emb_cache_{model_name.replace('/', '_')}.pkl")


def _load_cache(path: str) -> dict[str, np.ndarray]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, EOFError, OSError):
        return {}


def _save_cache(path: str, cache: dict[str, np.ndarray]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def embed_titles(
    titles: list[str], model_name: str | None = None, use_cache: bool = True
) -> np.ndarray:
    cfg = ml_config()["embeddings"]
    model_name = model_name or cfg["model_name"]
    batch_size = int(cfg.get("batch_size", 32))
    os.makedirs(CACHE_DIR, exist_ok=True)

    hashes = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in titles]
    cache = _load_cache(_cache_path(model_name)) if use_cache else {}

    missing_idx = [i for i, h in enumerate(hashes) if h not in cache]
    if missing_idx:
        # de-dupe: several listings can share a title
        to_encode = sorted({titles[i] for i in missing_idx})
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415  (lazy: torch import is slow)

        print(f"[*] Encoding {len(to_encode)} new title(s) with {model_name} "
              f"({len(titles) - len(missing_idx)} cache hits)...")
        model = SentenceTransformer(model_name)
        vecs = model.encode(
            to_encode, batch_size=batch_size, show_progress_bar=len(to_encode) > 200,
            normalize_embeddings=True,
        ).astype(np.float32)
        for t, v in zip(to_encode, vecs):
            cache[hashlib.sha1(t.encode("utf-8")).hexdigest()] = v
        if use_cache:
            _save_cache(_cache_path(model_name), cache)
    else:
        print(f"[*] All {len(titles)} title embeddings served from cache.")

    return np.vstack([cache[h] for h in hashes])


if __name__ == "__main__":
    from src.ml.load_data import load_raw_listings

    df = load_raw_listings()
    X = embed_titles(df["title_clean"].tolist())
    print("embeddings shape:", X.shape)
