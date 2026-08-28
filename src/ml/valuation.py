"""
Phase 2: valuation & anomaly detection.

Given the entity-resolved listings from Phase 1 (`data/processed/listings_labeled.pkl`),
compute a robust price distribution per **year + model entity** and flag listings
that sit far below it.

Robust stats (median + MAD) are used instead of mean/std because marketplace
prices are heavy-tailed and salted with scams, parts listings and typos - a
single "$1" listing would wreck a mean/std z-score. Each listing is scored
leave-one-out so it never inflates its own baseline.

    python -m src.ml.valuation

Outputs:
    data/processed/valuation.csv   every scored listing + entity stats + robust_z
    data/processed/deals.csv       just the flagged underpriced listings, best first
"""
import os

import numpy as np
import pandas as pd

from src.ml.config import ml_config

LABELED_PATH = "data/processed/listings_labeled.pkl"
PROCESSED_DIR = "data/processed"

_MAD_TO_SIGMA = 1.4826  # makes MAD a consistent estimator of sigma for normal data


def _robust_z_leave_one_out(prices: np.ndarray) -> np.ndarray:
    """Robust z-score of each price against the median/MAD of the *others*."""
    n = len(prices)
    out = np.full(n, np.nan)
    if n < 2:
        return out
    for i in range(n):
        others = np.delete(prices, i)
        med = np.median(others)
        mad = np.median(np.abs(others - med))
        scale = _MAD_TO_SIGMA * mad
        if scale == 0:
            # all comps identical - fall back to a small relative tolerance
            scale = max(1.0, 0.02 * med)
        out[i] = (prices[i] - med) / scale
    return out


def score_listings(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or ml_config()["valuation"]
    min_comps = int(cfg.get("min_comps", 5))
    z_threshold = float(cfg.get("z_threshold", -2.0))
    min_discount_pct = float(cfg.get("min_discount_pct", 0.15))
    suspect_ratio = float(cfg.get("suspect_ratio", 0.35))

    # Only listings we can actually value: real canonical model, a parsed year,
    # a positive price.
    scored = df[
        (df["canonical_label"].notna())
        & (df["canonical_label"] != "UNKNOWN")
        & (df["year"].notna())
        & (df["price"] > 0)
    ].copy()

    scored["entity_median"] = scored.groupby("entity_id")["price"].transform("median")
    scored["entity_comps"] = scored.groupby("entity_id")["price"].transform("size")
    scored["robust_z"] = (
        scored.groupby("entity_id")["price"]
        .transform(lambda s: _robust_z_leave_one_out(s.to_numpy(dtype=float)))
    )
    scored["discount_pct"] = 1.0 - scored["price"] / scored["entity_median"]

    enough = scored["entity_comps"] >= min_comps
    scored["is_suspect"] = enough & (scored["price"] < suspect_ratio * scored["entity_median"])
    scored["is_deal"] = (
        enough
        & (scored["robust_z"] <= z_threshold)
        & (scored["discount_pct"] >= min_discount_pct)
        & ~scored["is_suspect"]
    )
    # Corroborating signal: the seller already marked it down.
    scored["seller_marked_down"] = scored["price_original"].notna() & (
        scored["price"] < scored["price_original"]
    )

    return scored.sort_values("robust_z")


def main() -> None:
    if not os.path.exists(LABELED_PATH):
        raise FileNotFoundError(f"{LABELED_PATH} not found. Run `python -m src.ml.run_pipeline` first.")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df = pd.read_pickle(LABELED_PATH)
    scored = score_listings(df)

    cols = [
        "entity_label", "price", "entity_median", "discount_pct", "robust_z",
        "entity_comps", "is_deal", "is_suspect", "seller_marked_down",
        "raw_price_original", "location", "raw_listing_location", "url",
    ]
    valuation_path = os.path.join(PROCESSED_DIR, "valuation.csv")
    deals_path = os.path.join(PROCESSED_DIR, "deals.csv")
    scored[cols].to_csv(valuation_path, index=False)

    deals = scored[scored["is_deal"]].copy()
    deals[cols].to_csv(deals_path, index=False)

    n_scored = int((scored["entity_comps"] >= ml_config()["valuation"]["min_comps"]).sum())
    print(f"[+] Scored {n_scored} listings across "
          f"{scored.loc[scored['entity_comps'] >= 5, 'entity_id'].nunique()} entities with enough comps")
    print(f"[+] {len(deals)} deals, {int(scored['is_suspect'].sum())} suspect (likely junk/scam)")
    print(f"[+] Wrote {valuation_path} and {deals_path}")

    if len(deals):
        show = deals.head(20).assign(
            price=lambda d: d["price"].map("${:,.0f}".format),
            median=lambda d: d["entity_median"].map("${:,.0f}".format),
            off=lambda d: (d["discount_pct"] * 100).map("{:.0f}%".format),
            z=lambda d: d["robust_z"].map("{:.2f}".format),
        )
        print("\nTop deals:")
        print(show[["entity_label", "price", "median", "off", "z", "entity_comps",
                    "seller_marked_down", "location"]].to_string(index=False))


if __name__ == "__main__":
    main()
