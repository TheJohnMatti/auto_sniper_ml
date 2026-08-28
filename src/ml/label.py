"""
Phase 1 - Step 3: resolve each cluster to a canonical "Make Model" label.

There is **no external LLM API** in this pipeline. Labeling is an agent-in-the-loop
step:

    1. run_pipeline writes the centroid samples for every cluster to
       data/clusters/label_requests.json
    2. you invoke the `label-clusters` skill, which has you (Claude) read those
       samples and write canonical labels to data/clusters/label_map.json
    3. the next run_pipeline picks the labels up automatically

Any cluster without a curated label falls back to a deterministic token-frequency
heuristic, so the pipeline is never blocked on the labeling step.
"""
import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd

from src.ml.cluster import nearest_to_centroid
from src.ml.config import ml_config

LABEL_REQUESTS_PATH = "data/clusters/label_requests.json"
LABEL_MAP_PATH = "data/clusters/label_map.json"

_REQUEST_INSTRUCTIONS = (
    "Agent-in-the-loop labeling for auto_sniper_ml Phase 1. For each cluster, read "
    "sample_titles and write the canonical vehicle as 'Make Model' (title case, no "
    "year, no trim) into data/clusters/label_map.json keyed by cluster id. See the "
    "`label-clusters` skill for the full procedure."
)

_STOPWORDS = {
    "the", "a", "for", "sale", "clean", "low", "kms", "km", "miles", "obo",
    "auto", "automatic", "manual", "loaded", "financing", "available", "no",
    "accidents", "certified", "safety", "safetied", "mint", "condition",
    "awd", "fwd", "rwd", "4wd", "quattro", "sedan", "coupe", "hatchback",
    "wagon", "turbo", "sport", "premium", "limited", "touring",
}


def heuristic_label(titles: list[str]) -> str:
    """Deterministic model-level label from token frequency. Fallback only."""
    tokens: list[str] = []
    for title in titles:
        # letter-initial tokens, keeping trailing digits/hyphens so "mazda3",
        # "cx-5", "f-150", "3-series" survive as single model tokens
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9-]*", str(title).lower()):
            if tok not in _STOPWORDS and len(tok) > 1:
                tokens.append(tok)
    common = [tok for tok, _ in Counter(tokens).most_common(3)]

    # collapse containment pairs to the more specific token ("mazda" + "mazda3" -> "mazda3")
    picked: list[str] = []
    for tok in common:
        replaced = False
        for i, p in enumerate(picked):
            if p in tok or tok in p:
                picked[i] = tok if len(tok) > len(p) else p
                replaced = True
                break
        if not replaced:
            picked.append(tok)
        if len(picked) == 2:
            break
    return " ".join(t.capitalize() for t in picked) or "UNLABELED"


def write_label_requests(df: pd.DataFrame, embeddings: np.ndarray, labels: np.ndarray,
                         centroids: np.ndarray, path: str = LABEL_REQUESTS_PATH) -> str:
    """Dump per-cluster centroid samples for the `label-clusters` skill to classify."""
    n_samples = int(ml_config()["labeling"].get("samples_per_cluster", 10))

    clusters = []
    for cid in sorted(set(int(c) for c in labels)):
        idx = nearest_to_centroid(embeddings, labels, centroids, cid, n_samples)
        member_titles = df.loc[df["cluster"] == cid, "raw_title"].tolist()
        clusters.append({
            "cluster": cid,
            "n_listings": int((labels == cid).sum()),
            "sample_titles": df.iloc[idx]["raw_title"].tolist(),
            "heuristic_guess": heuristic_label(member_titles),
        })

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"instructions": _REQUEST_INSTRUCTIONS, "clusters": clusters}, f, indent=2)
    return path


def load_label_map(path: str = LABEL_MAP_PATH) -> dict[int, str]:
    """Read curated {cluster_id: 'Make Model'} labels if the skill has produced them."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "labels" in raw:
        raw = raw["labels"]
    return {int(k): str(v).strip() for k, v in raw.items() if str(v).strip()}


def resolve_labels(df: pd.DataFrame, embeddings: np.ndarray, labels: np.ndarray,
                   centroids: np.ndarray) -> pd.DataFrame:
    """Curated label per cluster where available, deterministic heuristic otherwise."""
    label_map = load_label_map()

    rows = []
    for cid in sorted(set(int(c) for c in labels)):
        member_titles = df.loc[df["cluster"] == cid, "raw_title"].tolist()
        idx = nearest_to_centroid(embeddings, labels, centroids, cid, 8)

        if cid in label_map:
            label, method = label_map[cid], "curated"
        else:
            label, method = heuristic_label(member_titles), "heuristic"

        rows.append({
            "cluster": cid,
            "canonical_label": label,
            "n_listings": int((labels == cid).sum()),
            "label_method": method,
            "sample_titles": " | ".join(df.iloc[idx]["raw_title"].tolist()),
        })

    out = pd.DataFrame(rows)
    n_curated = int((out["label_method"] == "curated").sum())
    print(f"[+] Resolved {len(out)} clusters ({n_curated} curated, {len(out) - n_curated} heuristic)")
    if n_curated < len(out):
        print(f"[i] {len(out) - n_curated} clusters still heuristic-labeled. Invoke the "
              f"`label-clusters` skill on {LABEL_REQUESTS_PATH}, then re-run this pipeline.")
    return out


if __name__ == "__main__":
    from src.ml.cluster import cluster_embeddings
    from src.ml.embed import embed_titles
    from src.ml.load_data import load_raw_listings

    df = load_raw_listings()
    X = embed_titles(df["title_clean"].tolist())
    res = cluster_embeddings(X)
    df = df.assign(cluster=res.labels)
    write_label_requests(df, X, res.labels, res.centroids)
    print(resolve_labels(df, X, res.labels, res.centroids).head(20).to_string())
