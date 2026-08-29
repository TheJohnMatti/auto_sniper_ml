"""
Phase 2: valuation & anomaly detection.

Given the entity-resolved listings from Phase 1 (`data/processed/listings_labeled.pkl`),
compute a robust price distribution per **year + model entity** and flag listings
that sit far below it.

Robust stats (median + MAD) are used instead of mean/std because marketplace
prices are heavy-tailed and salted with scams, parts listings and typos - a
single "$1" listing would wreck a mean/std z-score. Each listing is scored
leave-one-out so it never inflates its own baseline.

Prices are **mileage-adjusted before scoring** (see src.ml.mileage): a 300 000 km
car restated to the reference odometer stops looking underpriced just because
it's worn out. Odometer comes from the Phase 2b description scrape.

    python -m src.ml.valuation

If `data/processed/listing_signals.csv` exists (from src.ml.sentiment), the
description-derived condition / urgency / dealer-ad signals are folded into a
combined `deal_score` and the deal list is ranked and filtered by it.

**Outliers** ("too good to be true") are kept in the scored table - the robust
leave-one-out baseline already shrugs them off, and we want the description
scrape to learn their tells - but flagged `is_outlier` and dropped from the deal
list at inference time. A listing is an outlier when the price field clearly
isn't the sale price (finance payment / lease buy-in / rental deposit), when the
price is a keyboard-mash placeholder ($1, $1234, ...), or when the discount is
too deep to be real *and* nothing in the description (salvage, blown engine,
...) explains it.

Outputs:
    data/processed/valuation.csv   every scored listing + entity stats + robust_z
    data/processed/deals.csv       just the flagged underpriced listings, best first
"""
import os

import numpy as np
import pandas as pd

from src.ml.config import ml_config
from src.ml.mileage import adjust_prices, fit_km_slope

LABELED_PATH = "data/processed/listings_labeled.pkl"
SIGNALS_PATH = "data/processed/listing_signals.csv"
PROCESSED_DIR = "data/processed"

_MAD_TO_SIGMA = 1.4826  # makes MAD a consistent estimator of sigma for normal data

# Prices that are almost always a placeholder rather than a real ask - repdigits,
# keyboard walks and "just put something" round numbers. $1234 alone shows up ~46x.
_PLACEHOLDER_PRICES = {
    1, 10, 11, 12, 99, 100, 111, 123, 123456, 321, 999,
    1000, 1111, 1234, 4321, 9999, 11111, 12321, 12345, 54321, 99999, 111111,
}
# description red flags severe enough to genuinely justify a rock-bottom price
_SEVERE_FLAGS = {
    "parts_only", "wont_run", "salvage_rebuilt", "flood_fire",
    "engine_trans", "accident_damage", "as_is", "needs_work",
}


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


def _odometer_by_item(path: str = SIGNALS_PATH) -> pd.Series | None:
    if not os.path.exists(path):
        return None
    sig = pd.read_csv(path, dtype={"item_id": str})
    if "odometer_km" not in sig.columns:
        return None
    sig = sig.dropna(subset=["item_id"]).drop_duplicates("item_id", keep="last")
    return pd.to_numeric(sig.set_index("item_id")["odometer_km"], errors="coerce")


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

    # --- mileage adjustment: restate every price to a reference odometer -------
    odo = _odometer_by_item()
    scored["odometer_km"] = (
        scored["item_id"].map(odo) if odo is not None else np.nan
    )
    # drop odometers that can't be right: absurd values, or far too low for the
    # car's age (a "7 000 km" 2013 is a dropped digit) - a bogus low reading
    # would otherwise make a worn car look like a steal.
    age = (2026 - scored["year"]).clip(lower=0)
    bad_odo = scored["odometer_km"].notna() & (
        (scored["odometer_km"] < 1_000)
        | (scored["odometer_km"] > 500_000)
        | ((age >= 4) & (scored["odometer_km"] < age * 1_500))
    )
    scored.loc[bad_odo, "odometer_km"] = np.nan

    km_slope = fit_km_slope(scored, min_comps=min_comps)
    scored = adjust_prices(scored, km_slope)
    scored["km_slope"] = km_slope

    # --- robust distribution, computed on the mileage-adjusted price ----------
    grp = scored.groupby("entity_id")
    scored["entity_median"] = grp["price"].transform("median")            # raw, for display
    scored["entity_adj_median"] = grp["mileage_adj_price"].transform("median")
    scored["entity_comps"] = grp["price"].transform("size")
    scored["robust_z"] = grp["mileage_adj_price"].transform(
        lambda s: _robust_z_leave_one_out(s.to_numpy(dtype=float))
    )
    scored["discount_pct"] = 1.0 - scored["mileage_adj_price"] / scored["entity_adj_median"]
    scored["raw_discount_pct"] = 1.0 - scored["price"] / scored["entity_median"]

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


_SIGNAL_COLS = [
    "has_description", "condition_score", "urgency_score", "red_flag_count",
    "red_flags", "is_dealer_or_ad", "price_not_sale_price", "price_not_sale_cues",
    "listing_age_days",
]


def attach_signals(scored: pd.DataFrame, path: str = SIGNALS_PATH) -> pd.DataFrame:
    """Left-join description signals and compute a combined deal_score."""
    if os.path.exists(path):
        sig = pd.read_csv(path, dtype={"item_id": str})
        keep = ["item_id"] + [c for c in _SIGNAL_COLS if c in sig.columns]
        scored = scored.merge(sig[keep], on="item_id", how="left")

    for col, default in (("has_description", False), ("is_dealer_or_ad", False),
                         ("price_not_sale_price", False), ("price_not_sale_cues", ""),
                         ("condition_score", 0.0), ("urgency_score", 0.0),
                         ("red_flag_count", 0), ("red_flags", "")):
        if col not in scored.columns:
            scored[col] = default
        scored[col] = scored[col].fillna(default)
    if "listing_age_days" not in scored.columns:
        scored["listing_age_days"] = np.nan
    scored["listing_age_days"] = pd.to_numeric(scored["listing_age_days"], errors="coerce")

    # how far below market, capped so a 3-comp blowup doesn't dominate
    underpricing = (scored["discount_pct"] / 0.5).clip(0.0, 1.0)
    # recency: a real underpriced car sells fast, so fresh listings are worth more.
    # +0.15 at <=2d fading to 0 by ~3 weeks; unknown age is neutral.
    recency = (1.0 - scored["listing_age_days"].fillna(7.0) / 21.0).clip(-0.5, 1.0) * 0.15
    scored["deal_score"] = (
        underpricing
        + 0.30 * scored["condition_score"]
        + 0.10 * scored["urgency_score"]
        + recency
        - 0.15 * scored["red_flag_count"].clip(upper=4)
        - 0.50 * scored["is_dealer_or_ad"].astype(float)
    ).round(3)

    # a dealer ad / "we buy cars" post is never a deal, however cheap it looks
    scored.loc[scored["is_dealer_or_ad"], "is_deal"] = False
    return flag_region(flag_stale(flag_outliers(scored)))


def flag_outliers(scored: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Flag "too good to be true" listings and drop them from the deal list.

    Kept in the scored table (the robust leave-one-out baseline already ignores
    them, and the description scrape learns their tells); only `is_deal` and the
    downstream notifications are suppressed.
    """
    cfg = cfg or ml_config()["valuation"]
    too_good_ratio = float(cfg.get("outlier_ratio", 0.25))
    too_good_z = float(cfg.get("outlier_z", -4.0))
    project_floor = float(cfg.get("project_car_floor", 400))

    ratio = scored["price"] / scored["entity_median"]
    enough = scored["entity_comps"] >= 3

    # 1. the price field isn't the sale price (finance payment / lease / rental)
    not_sale_price = scored["price_not_sale_price"].astype(bool)

    # 2. keyboard-mash / round-number placeholder instead of a real ask
    placeholder = (
        scored["price"].isin(_PLACEHOLDER_PRICES) | (scored["price"] <= 100)
    ) & (ratio < 0.6)

    # 3. discount too deep to be real, with nothing in the description to explain it
    severe = scored["red_flags"].fillna("").apply(
        lambda s: bool(_SEVERE_FLAGS.intersection(s.split(";"))) if s else False
    )
    damage_explains = severe & (scored["price"] >= project_floor)
    implausible = enough & ((ratio < too_good_ratio) | (scored["robust_z"] <= too_good_z))

    scored["is_outlier"] = (
        not_sale_price | placeholder | (implausible & ~damage_explains)
    )
    scored.loc[scored["is_outlier"], "is_deal"] = False
    return scored


def flag_stale(scored: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Drop deals that have sat on the market too long to still be real.

    A genuinely underpriced car is gone in days. Anything still listed after
    weeks - especially something that *looks* like a steal - has a reason it
    hasn't sold (bad title, hidden damage, phantom listing, wrong price). We
    only have an age for listings whose description page was scraped; unknown
    age is treated as fresh (other signals still apply).
    """
    cfg = cfg or ml_config()["valuation"]
    max_age = float(cfg.get("max_listing_age_days", 45))
    steal_age = float(cfg.get("stale_steal_days", 14))
    steal_discount = float(cfg.get("stale_steal_discount", 0.45))

    age = scored["listing_age_days"]
    scored["is_stale"] = (age > max_age) | (
        (age > steal_age) & (scored["discount_pct"] >= steal_discount)
    )
    scored["is_stale"] = scored["is_stale"].fillna(False)
    scored.loc[scored["is_stale"], "is_deal"] = False
    return scored


def flag_region(scored: pd.DataFrame) -> pd.DataFrame:
    """Production only surfaces deals in the configured region (SW Ontario).

    Comps / baselines still use every scraped listing - only `is_deal` (and
    therefore notifications) is gated. A listing with no parsed "Town, PROV"
    line falls back to its scrape `location` tag.
    """
    from src.ml.geo import in_region

    loc = scored.get("raw_listing_location")
    if loc is None:
        loc = pd.Series("", index=scored.index)
    loc = loc.astype(str).str.strip()
    loc = loc.where(loc.ne(""), scored.get("location", pd.Series("", index=scored.index)))
    scored["in_region"] = loc.map(in_region)
    scored.loc[~scored["in_region"], "is_deal"] = False
    return scored


def main() -> None:
    if not os.path.exists(LABELED_PATH):
        raise FileNotFoundError(f"{LABELED_PATH} not found. Run `python -m src.ml.run_pipeline` first.")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df = pd.read_pickle(LABELED_PATH)
    scored = attach_signals(score_listings(df))

    cols = [
        "entity_label", "price", "entity_median", "mileage_adj_price",
        "odometer_km", "mileage_known", "listing_age_days", "discount_pct",
        "raw_discount_pct", "robust_z", "entity_comps", "deal_score", "is_deal",
        "is_suspect", "is_outlier", "is_stale", "in_region", "seller_marked_down",
        "has_description", "condition_score", "urgency_score", "red_flags",
        "is_dealer_or_ad", "price_not_sale_cues", "raw_price_original",
        "location", "raw_listing_location", "url",
    ]
    cols = [c for c in cols if c in scored.columns]
    valuation_path = os.path.join(PROCESSED_DIR, "valuation.csv")
    deals_path = os.path.join(PROCESSED_DIR, "deals.csv")
    scored[cols].sort_values("deal_score", ascending=False).to_csv(valuation_path, index=False)

    deals = scored[scored["is_deal"]].sort_values("deal_score", ascending=False).copy()
    deals[cols].to_csv(deals_path, index=False)

    cfg = ml_config()["valuation"]
    n_scored = int((scored["entity_comps"] >= cfg["min_comps"]).sum())
    n_desc = int(scored["has_description"].sum())
    slope = float(scored["km_slope"].iloc[0]) if len(scored) else float("nan")
    print(f"[+] Scored {n_scored} listings across "
          f"{scored.loc[scored['entity_comps'] >= 5, 'entity_id'].nunique()} entities with enough comps")
    print(f"[+] Mileage slope {slope:.2e} log$/km "
          f"(~{(1 - np.exp(slope * 1e5)) * 100:.0f}% per 100k km); "
          f"{int(scored['mileage_known'].sum())} listings have an odometer")
    print(f"[+] {n_desc} listings have a scraped description; "
          f"{int(scored['is_dealer_or_ad'].sum())} flagged dealer/ad")
    print(f"[+] {len(deals)} deals, {int(scored['is_suspect'].sum())} suspect, "
          f"{int(scored['is_outlier'].sum())} outlier (too good to be true), "
          f"{int(scored['is_stale'].sum())} stale (on market too long)")
    print(f"[+] {int(scored['in_region'].sum())}/{len(scored)} listings in region; "
          f"deals outside the region are dropped ({int((~scored['in_region']).sum())} excluded)")
    print(f"[+] Wrote {valuation_path} and {deals_path}")

    if len(deals):
        show = deals.head(20).assign(
            price=lambda d: d["price"].map("${:,.0f}".format),
            median=lambda d: d["entity_median"].map("${:,.0f}".format),
            km=lambda d: d["odometer_km"].map(lambda v: "-" if pd.isna(v) else f"{v/1000:.0f}k"),
            age=lambda d: d["listing_age_days"].map(lambda v: "-" if pd.isna(v) else f"{v:.0f}d"),
            off=lambda d: (d["discount_pct"] * 100).map("{:.0f}%".format),
            cond=lambda d: d["condition_score"].map("{:+.2f}".format),
        )
        print("\nTop deals (by deal_score; discount mileage-adjusted):")
        print(show[["entity_label", "price", "median", "km", "age", "off", "deal_score",
                    "cond", "red_flags", "entity_comps", "location"]].to_string(index=False))

    outliers = scored[scored["is_outlier"]].sort_values("price")
    if len(outliers):
        why = np.where(
            outliers["price_not_sale_price"].astype(bool), "price!=sale",
            np.where(
                outliers["price"].isin(_PLACEHOLDER_PRICES) | (outliers["price"] <= 100),
                "placeholder", "discount unexplained",
            ),
        )
        oshow = outliers.head(12).assign(
            why=why[:12],
            price=lambda d: d["price"].map("${:,.0f}".format),
            median=lambda d: d["entity_median"].map("${:,.0f}".format),
        )
        print("\nOutliers dropped from deals (too good to be true):")
        print(oshow[["entity_label", "price", "median", "why", "red_flags",
                     "entity_comps", "location"]].to_string(index=False))


if __name__ == "__main__":
    main()
