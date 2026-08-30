"""
Phase 1 - Step 0: consolidate raw scraper output into a single clean frame.

Reads every ``data/raw/facebook_*_raw_*.csv`` dump, deduplicates listings that
were seen in multiple scrape runs, and derives the numeric / structured fields
the downstream ML steps need (price, year, cleaned title).
"""
import glob
import os
import re

import numpy as np
import pandas as pd

RAW_DIR = "data/raw"
RAW_GLOB = os.path.join(RAW_DIR, "facebook_*_raw_*.csv")

_ITEM_ID_RE = re.compile(r"/marketplace/item/(\d+)")
_YEAR_RE = re.compile(r"\b(19[8-9]\d|20[0-4]\d)\b")
_PRICE_RE = re.compile(r"[\d.]+")

# Expected columns across scraper schema versions; older dumps lack the last two.
_EXPECTED_COLS = [
    "scraped_at", "platform", "location", "raw_title", "raw_price",
    "raw_price_original", "raw_listing_location", "raw_description", "url",
]


def _to_price(value: str) -> float:
    """'CA$3,200' -> 3200.0 ; 'Free' -> 0.0 ; junk -> NaN."""
    if not isinstance(value, str) or not value.strip():
        return np.nan
    text = value.strip()
    if text.lower() == "free":
        return 0.0
    cleaned = text.replace(",", "")
    match = _PRICE_RE.search(cleaned)
    return float(match.group()) if match else np.nan


def _extract_item_id(url: str) -> str | None:
    if not isinstance(url, str):
        return None
    match = _ITEM_ID_RE.search(url)
    return match.group(1) if match else None


def _extract_year(title: str):
    """Return the 4-digit model year found in a title, or pd.NA."""
    if not isinstance(title, str):
        return pd.NA
    match = _YEAR_RE.search(title)
    return int(match.group()) if match else pd.NA


def load_raw_listings(raw_glob: str = RAW_GLOB) -> pd.DataFrame:
    paths = sorted(glob.glob(raw_glob))
    if not paths:
        raise FileNotFoundError(f"No raw scrape files match {raw_glob!r}. Run src/scraper/run.py first.")

    frames = []
    for path in paths:
        df = pd.read_csv(path, dtype=str)
        for col in _EXPECTED_COLS:
            if col not in df.columns:
                df[col] = pd.NA
        frames.append(df[_EXPECTED_COLS])

    listings = pd.concat(frames, ignore_index=True)

    # --- derive structured fields -------------------------------------------------
    listings["item_id"] = listings["url"].map(_extract_item_id)
    listings["scraped_at"] = pd.to_datetime(listings["scraped_at"], errors="coerce")
    listings["price"] = listings["raw_price"].map(_to_price)
    listings["price_original"] = listings["raw_price_original"].map(_to_price)
    listings["year"] = listings["raw_title"].map(_extract_year).astype("Int64")

    # --- drop unusable rows -----------------------------------------------------
    listings = listings[listings["raw_title"].notna()]
    listings = listings[listings["raw_title"] != "UNKNOWN_TITLE"]
    # titles that are really just a stray price token (obscured/ad cards)
    listings = listings[~listings["raw_title"].str.match(r"^\s*(?:CA)?\$?[\d,.]+\s*$", na=False)]
    listings = listings[listings["item_id"].notna()]
    listings = listings[listings["price"].notna()]
    # Drop obvious price junk (phone numbers / VINs picked up as prices). A used
    # car worth sniping sits well inside this band; "Free" (0.0) is kept.
    listings = listings[(listings["price"] == 0) | listings["price"].between(200, 400_000)]

    # --- deduplicate: keep the freshest sighting of each listing ---------------
    listings = (
        listings.sort_values("scraped_at")
        .drop_duplicates(subset="item_id", keep="last")
        .reset_index(drop=True)
    )

    # --- drop non-cars ---------------------------------------------------------
    # The vehicles feed is full of motorcycles / boats / trailers / equipment.
    # Pricing those against cars invents fake deals (see src/ml/vehicle_type.py).
    from src.ml.vehicle_type import _load_curated, is_priceable

    curated = _load_curated()
    keep = [
        is_priceable(iid, title, curated)
        for iid, title in zip(listings["item_id"], listings["raw_title"])
    ]
    dropped = len(listings) - sum(keep)
    listings = listings[keep].reset_index(drop=True)
    if dropped:
        print(f"[i] dropped {dropped} non-car listings (motorcycle/boat/trailer/equipment)")

    # --- normalized text for embedding ----------------------------------------
    listings["title_clean"] = (
        listings["raw_title"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    )

    return listings


if __name__ == "__main__":
    df = load_raw_listings()
    print(f"[+] {len(df)} unique listings from {df['location'].nunique()} cities")
    print(df[["year", "raw_title", "price", "price_original", "raw_listing_location"]].head(15).to_string())
    print("\nprice describe:\n", df["price"].describe())
    print("\nmarked-down listings:", int(df["price_original"].notna().sum()))
