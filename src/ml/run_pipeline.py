"""
Phase 1 orchestrator: raw scrape CSVs -> entity-resolved, labeled listings.

    python src/ml/run_pipeline.py

Outputs:
    data/processed/listings_labeled.csv   one row per unique listing + cluster + entity_id/entity_label
    data/processed/listings_labeled.pkl   same, with pandas dtypes preserved for Phase 2
    data/clusters/cluster_labels.csv      one row per model cluster (label, size, method, samples)
"""
import os

import pandas as pd

from src.ml.cluster import cluster_embeddings
from src.ml.embed import embed_titles
from src.ml.label import label_clusters
from src.ml.load_data import load_raw_listings

PROCESSED_DIR = "data/processed"
CLUSTERS_DIR = "data/clusters"


def main() -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(CLUSTERS_DIR, exist_ok=True)

    print("=== Phase 1: Entity Resolution ===")
    df = load_raw_listings()
    print(f"[+] {len(df)} unique listings across {df['location'].nunique()} cities")

    embeddings = embed_titles(df["title_clean"].tolist())

    result = cluster_embeddings(embeddings)
    df = df.assign(cluster=result.labels)

    cluster_df = label_clusters(df, embeddings, result.labels, result.centroids)
    label_map = dict(zip(cluster_df["cluster"], cluster_df["canonical_label"]))
    df["canonical_label"] = df["cluster"].map(label_map)

    # Entity = model (from the cluster) + this listing's own parsed year, so a
    # single "Honda Civic" cluster resolves to 2012/2014/... entities for Phase 2.
    model_only = df["canonical_label"].str.replace(r"^\s*(?:19|20)\d{2}\s+", "", regex=True)
    df["entity_label"] = [
        f"{int(y)} {m}" if pd.notna(y) else m
        for y, m in zip(df["year"], model_only)
    ]
    df["entity_id"] = df["entity_label"].astype("category").cat.codes

    listings_csv = os.path.join(PROCESSED_DIR, "listings_labeled.csv")
    listings_pkl = os.path.join(PROCESSED_DIR, "listings_labeled.pkl")
    clusters_path = os.path.join(CLUSTERS_DIR, "cluster_labels.csv")
    df.to_csv(listings_csv, index=False)
    df.to_pickle(listings_pkl)
    cluster_df.to_csv(clusters_path, index=False)

    print(f"\n[+] Wrote {listings_csv}")
    print(f"[+] Wrote {listings_pkl}")
    print(f"[+] Wrote {clusters_path}")
    print(f"\n[+] {df['entity_id'].nunique()} distinct entities "
          f"({result.k} model clusters x parsed year)")
    print("\nTop entities by listing count:")
    top = (
        df.groupby("entity_label")
        .agg(n=("entity_label", "size"), median_price=("price", "median"))
        .sort_values("n", ascending=False)
        .head(15)
    )
    print(top.to_string())


if __name__ == "__main__":
    main()
