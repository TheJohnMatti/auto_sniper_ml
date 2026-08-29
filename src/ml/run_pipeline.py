"""
Phase 1 orchestrator: raw scrape CSVs -> entity-resolved, labeled listings.

    python -m src.ml.run_pipeline                # full: cluster from scratch
    python -m src.ml.run_pipeline --incremental  # freeze clusters, just assign new listings

**Full** run (the weekly retrain) clusters every title, writes the label
requests for the `label-clusters` skill, and snapshots the model
(`data/processed/model/centroids.npy` + `data/clusters/cluster_labels.csv`) so
later incremental runs stay consistent.

**Incremental** run (the hourly job) skips clustering entirely: it embeds the
current listings and assigns each to its nearest frozen centroid, reusing the
last full run's labels. Fast, and - crucially - stable, so "what counts as a
deal" doesn't reshuffle every hour. Falls back to a full run if no snapshot
exists yet.

Cluster labeling has no external API: the full run writes centroid samples to
data/clusters/label_requests.json; invoke the `label-clusters` skill to turn
them into data/clusters/label_map.json, then the next full run folds them in.

Outputs:
    data/processed/listings_labeled.{csv,pkl}   one row per unique listing + entity_id/entity_label
    data/clusters/cluster_labels.csv            one row per model cluster (full run only)
    data/clusters/label_requests.json           centroid samples for the labeling skill (full run only)
    data/processed/model/centroids.npy          frozen cluster centroids (full run only)
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin_min

from src.ml.cluster import cluster_embeddings
from src.ml.embed import embed_titles
from src.ml.label import LABEL_MAP_PATH, resolve_labels, write_label_requests
from src.ml.load_data import load_raw_listings

PROCESSED_DIR = "data/processed"
CLUSTERS_DIR = "data/clusters"
MODEL_DIR = "data/processed/model"
CENTROIDS_PATH = os.path.join(MODEL_DIR, "centroids.npy")
CLUSTER_LABELS_PATH = os.path.join(CLUSTERS_DIR, "cluster_labels.csv")
LISTINGS_CSV = os.path.join(PROCESSED_DIR, "listings_labeled.csv")
LISTINGS_PKL = os.path.join(PROCESSED_DIR, "listings_labeled.pkl")


def _assign_entities(df: pd.DataFrame) -> pd.DataFrame:
    """cluster label -> canonical_label -> (year + model) entity id/label."""
    model_only = df["canonical_label"].str.replace(r"^\s*(?:19|20)\d{2}\s+", "", regex=True)
    df["entity_label"] = [
        f"{int(y)} {m}" if pd.notna(y) else m
        for y, m in zip(df["year"], model_only)
    ]
    df["entity_id"] = df["entity_label"].astype("category").cat.codes
    return df


def _write_listings(df: pd.DataFrame) -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(LISTINGS_CSV, index=False)
    df.to_pickle(LISTINGS_PKL)


def _top_entities(df: pd.DataFrame) -> None:
    top = (
        df.groupby("entity_label")
        .agg(n=("entity_label", "size"), median_price=("price", "median"))
        .sort_values("n", ascending=False)
        .head(15)
    )
    print("\nTop entities by listing count:")
    print(top.to_string())


def full_run() -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(CLUSTERS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=== Phase 1: Entity Resolution (full) ===")
    df = load_raw_listings()
    print(f"[+] {len(df)} unique listings across {df['location'].nunique()} locations")

    embeddings = embed_titles(df["title_clean"].tolist())
    result = cluster_embeddings(embeddings)
    df = df.assign(cluster=result.labels)

    requests_path = write_label_requests(df, embeddings, result.labels, result.centroids)
    print(f"[+] Wrote {requests_path} ({result.k} clusters)")

    cluster_df = resolve_labels(df, embeddings, result.labels, result.centroids)
    label_map = dict(zip(cluster_df["cluster"], cluster_df["canonical_label"]))
    df["canonical_label"] = df["cluster"].map(label_map)
    df = _assign_entities(df)

    _write_listings(df)
    cluster_df.to_csv(CLUSTER_LABELS_PATH, index=False)
    np.save(CENTROIDS_PATH, result.centroids)

    print(f"\n[+] Wrote {LISTINGS_PKL}, {CLUSTER_LABELS_PATH}, {CENTROIDS_PATH}")
    print(f"[+] {df['entity_id'].nunique()} distinct entities "
          f"({result.k} model clusters x parsed year)")
    if not os.path.exists(LABEL_MAP_PATH):
        print(f"[i] No {LABEL_MAP_PATH} yet - labels are all heuristic. "
              f"Invoke the `label-clusters` skill, then re-run.")
    _top_entities(df)


def incremental_run() -> None:
    if not (os.path.exists(CENTROIDS_PATH) and os.path.exists(CLUSTER_LABELS_PATH)):
        print("[i] no frozen model snapshot - doing a full run instead")
        return full_run()

    print("=== Phase 1: Entity Resolution (incremental) ===")
    df = load_raw_listings()
    centroids = np.load(CENTROIDS_PATH)
    cl = pd.read_csv(CLUSTER_LABELS_PATH)
    label_map = dict(zip(cl["cluster"].astype(int), cl["canonical_label"].astype(str)))

    embeddings = embed_titles(df["title_clean"].tolist())
    labels, _ = pairwise_distances_argmin_min(embeddings, centroids)
    df["cluster"] = labels.astype(int)
    df["canonical_label"] = df["cluster"].map(label_map).fillna("UNKNOWN")
    df = _assign_entities(df)

    _write_listings(df)
    n_unknown = int((df["canonical_label"] == "UNKNOWN").sum())
    print(f"[+] {len(df)} listings assigned to {df['entity_id'].nunique()} entities "
          f"against {len(centroids)} frozen clusters ({n_unknown} unlabeled)")
    _top_entities(df)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--incremental", action="store_true",
                    help="assign to frozen clusters instead of re-clustering (hourly job)")
    args = ap.parse_args()
    incremental_run() if args.incremental else full_run()


if __name__ == "__main__":
    main()
