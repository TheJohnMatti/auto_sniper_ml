"""
Phase 1 - Step 2: group near-identical listings without hardcoding K.

Bisecting K-Means repeatedly splits the highest-inertia cluster, which mirrors
the real structure of the data (a few dominant models, a long tail of rare ones)
better than flat k-means with a guessed K. The configured ``initial_k`` is treated
as an upper bound and clamped down for small datasets.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import BisectingKMeans

from src.ml.config import ml_config


@dataclass
class ClusterResult:
    labels: np.ndarray          # (n_samples,) cluster id per listing
    centroids: np.ndarray       # (k, dim)
    k: int
    model: BisectingKMeans


def choose_k(n_samples: int, initial_k: int, min_per_cluster: int = 5) -> int:
    """Clamp the configured K so clusters are not trivially tiny."""
    upper = max(2, n_samples // min_per_cluster)
    return max(2, min(initial_k, upper))


def cluster_embeddings(embeddings: np.ndarray, initial_k: int | None = None) -> ClusterResult:
    cfg = ml_config()["clustering"]
    initial_k = initial_k if initial_k is not None else int(cfg.get("initial_k", 100))
    k = choose_k(len(embeddings), initial_k)

    strategy = cfg.get("bisecting_strategy", "biggest_inertia")
    random_state = int(cfg.get("random_state", 42))

    print(f"[*] Bisecting K-Means: n={len(embeddings)}, k={k}, strategy={strategy}")
    model = BisectingKMeans(
        n_clusters=k,
        bisecting_strategy=strategy,
        random_state=random_state,
        n_init=1,
    )
    labels = model.fit_predict(embeddings)
    return ClusterResult(labels=labels, centroids=model.cluster_centers_, k=k, model=model)


def nearest_to_centroid(embeddings: np.ndarray, labels: np.ndarray, centroids: np.ndarray,
                        cluster_id: int, n: int) -> np.ndarray:
    """Return indices of the ``n`` listings closest to a cluster's centroid."""
    member_idx = np.where(labels == cluster_id)[0]
    if len(member_idx) <= n:
        return member_idx
    dists = np.linalg.norm(embeddings[member_idx] - centroids[cluster_id], axis=1)
    return member_idx[np.argsort(dists)[:n]]


if __name__ == "__main__":
    from src.ml.load_data import load_raw_listings
    from src.ml.embed import embed_titles

    df = load_raw_listings()
    X = embed_titles(df["title_clean"].tolist())
    result = cluster_embeddings(X)

    df = df.assign(cluster=result.labels)
    sizes = df["cluster"].value_counts()
    print(f"\n[+] {result.k} clusters; largest {sizes.iloc[0]}, median {int(sizes.median())}")
    for cid in sizes.head(8).index:
        sample = df.loc[df["cluster"] == cid, "raw_title"].head(4).tolist()
        print(f"  cluster {cid} (n={sizes[cid]}): {sample}")
