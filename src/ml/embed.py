"""
Phase 1 - Step 1: vectorize messy listing titles into dense semantic embeddings.

Uses sentence-transformers (default: all-MiniLM-L6-v2). Embeddings are L2-normalized
so downstream Euclidean k-means behaves like cosine similarity. Results are cached
to disk keyed by (model, title-set hash) so re-runs are instant.
"""
import hashlib
import os

import numpy as np

from src.ml.config import ml_config

CACHE_DIR = "data/processed"


def _cache_path(model_name: str, titles: list[str]) -> str:
    digest = hashlib.sha1(("\n".join(titles)).encode("utf-8")).hexdigest()[:16]
    safe_model = model_name.replace("/", "_")
    return os.path.join(CACHE_DIR, f"emb_{safe_model}_{digest}.npy")


def embed_titles(titles: list[str], model_name: str | None = None, use_cache: bool = True) -> np.ndarray:
    cfg = ml_config()["embeddings"]
    model_name = model_name or cfg["model_name"]
    batch_size = int(cfg.get("batch_size", 32))

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = _cache_path(model_name, titles)
    if use_cache and os.path.exists(cache_path):
        print(f"[*] Loading cached embeddings: {cache_path}")
        return np.load(cache_path)

    # Imported lazily - loading torch is slow and not needed for cache hits.
    from sentence_transformers import SentenceTransformer

    print(f"[*] Encoding {len(titles)} titles with {model_name}...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        titles,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    if use_cache:
        np.save(cache_path, embeddings)
        print(f"[+] Cached embeddings -> {cache_path}")
    return embeddings


if __name__ == "__main__":
    from src.ml.load_data import load_raw_listings

    df = load_raw_listings()
    X = embed_titles(df["title_clean"].tolist())
    print("embeddings shape:", X.shape)
