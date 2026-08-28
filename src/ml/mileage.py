"""
Mileage-adjusted pricing for Phase 2 valuation.

A 2010 Civic with 95 000 km and one with 310 000 km are not the same car, yet
they land in the same ``year + model`` entity, so the high-km one looks like a
huge "deal" against the entity median when it is really just worn out. This
module estimates how much value a kilometre costs - pooled across every entity,
robustly - and restates each listing's price to a common reference odometer so
the downstream z-score compares like with like.

Model
-----
    log(price) = entity_effect + beta * clip(odometer_km, 0, KM_CAP)

``beta`` (log-price lost per km, negative) is fit once over every listing that
has a plausible odometer and at least ``min_comps`` entity siblings. We use the
median of pairwise slopes (Theil-Sen style) rather than least squares so a
handful of scam / typo odometers can't tilt the line. The estimate is clamped to
a sane band and falls back to ``DEFAULT_SLOPE`` when there isn't enough data.

    mileage_factor(km) = exp(beta * (clip(km, 0, KM_CAP) - REF_KM))
    mileage_adj_price  = price / mileage_factor(km_used)

so a 260 000 km car is marked *up* toward what it would fetch at the reference
odometer and stops masquerading as underpriced. Listings with no / implausible
odometer are restated at their entity's median known odometer (global ``REF_KM``
if the entity has none), i.e. left essentially unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PLAUSIBLE_KM = (1_000.0, 500_000.0)  # outside this an odometer is a typo/scam -> treated as unknown
KM_CAP = 300_000.0                   # depreciation vs km flattens out; don't extrapolate past this
REF_KM = 120_000.0                   # odometer every price is restated to
DEFAULT_SLOPE = -3.0e-6              # log-price per km, used when the fit is underpowered
SLOPE_BOUNDS = (-8.0e-6, -0.8e-6)    # clamp the fitted slope to a defensible range
_MIN_PAIR_KM = 20_000.0             # only use listing pairs at least this far apart in km
_MAX_PAIRS = 40_000                 # cap the pairwise-slope sample for speed
_MIN_FIT_POINTS = 40


def _plausible_km(km: pd.Series) -> pd.Series:
    lo, hi = PLAUSIBLE_KM
    return km.between(lo, hi)


def fit_km_slope(df: pd.DataFrame, min_comps: int = 5, seed: int = 42) -> float:
    """Estimate log-price lost per km from listings with an odometer.

    ``df`` needs ``price``, ``odometer_km`` and ``entity_id`` columns.
    """
    need = {"price", "odometer_km", "entity_id"}
    if not need.issubset(df.columns):
        return DEFAULT_SLOPE

    d = df[["price", "odometer_km", "entity_id"]].copy()
    d = d[(d["price"] > 0) & _plausible_km(d["odometer_km"])]
    d["entity_comps"] = d.groupby("entity_id")["price"].transform("size")
    d = d[d["entity_comps"] >= min_comps]
    if len(d) < _MIN_FIT_POINTS:
        return DEFAULT_SLOPE

    # residual after removing the entity effect (approximated by the log median)
    ent_med = d.groupby("entity_id")["price"].transform("median")
    resid = np.log(d["price"].to_numpy()) - np.log(ent_med.to_numpy())
    km = d["odometer_km"].clip(upper=KM_CAP).to_numpy()

    rng = np.random.default_rng(seed)
    n = len(d)
    i = rng.integers(0, n, _MAX_PAIRS)
    j = rng.integers(0, n, _MAX_PAIRS)
    far = np.abs(km[i] - km[j]) >= _MIN_PAIR_KM
    if far.sum() < _MIN_FIT_POINTS:
        return DEFAULT_SLOPE
    slopes = (resid[i][far] - resid[j][far]) / (km[i][far] - km[j][far])
    slope = float(np.median(slopes))

    lo, hi = SLOPE_BOUNDS
    if not np.isfinite(slope):
        return DEFAULT_SLOPE
    return float(np.clip(slope, lo, hi))


def mileage_factor(km: np.ndarray, slope: float) -> np.ndarray:
    """Price multiplier of a car at ``km`` relative to one at ``REF_KM``."""
    km = np.clip(np.asarray(km, dtype=float), 0.0, KM_CAP)
    return np.exp(slope * (km - REF_KM))


def adjust_prices(df: pd.DataFrame, slope: float) -> pd.DataFrame:
    """Add mileage columns to ``df`` (needs ``price``, ``odometer_km``, ``entity_id``).

    Adds:
        mileage_known      odometer present and plausible
        odometer_km_used   odometer actually used (filled for unknowns)
        mileage_factor     multiplier vs a REF_KM car
        mileage_adj_price  price restated to REF_KM
    """
    out = df.copy()
    known = _plausible_km(out["odometer_km"]) if "odometer_km" in out.columns else pd.Series(
        False, index=out.index
    )
    out["mileage_known"] = known.fillna(False)

    km = out["odometer_km"] if "odometer_km" in out.columns else pd.Series(np.nan, index=out.index)
    km = km.where(out["mileage_known"])
    # unknowns -> entity's median known odometer, else the global reference
    ent_known_med = km.groupby(out["entity_id"]).transform("median")
    km_used = km.fillna(ent_known_med).fillna(REF_KM).clip(lower=0.0, upper=KM_CAP)
    out["odometer_km_used"] = km_used

    out["mileage_factor"] = mileage_factor(km_used.to_numpy(), slope)
    out["mileage_adj_price"] = out["price"] / out["mileage_factor"]
    return out


def main() -> None:
    """Quick look at the fitted depreciation curve (needs the Phase 1 + 2b outputs)."""
    lab = pd.read_pickle("data/processed/listings_labeled.pkl")
    sig = pd.read_csv("data/processed/listing_signals.csv", dtype={"item_id": str})
    df = lab.merge(sig[["item_id", "odometer_km"]], on="item_id", how="left")
    df["odometer_km"] = pd.to_numeric(df["odometer_km"], errors="coerce")

    slope = fit_km_slope(df)
    print(f"[+] slope = {slope:.3e} log$/km")
    for km in (60_000, 120_000, 180_000, 240_000, 300_000):
        pct = (mileage_factor(np.array([km]), slope)[0] - 1) * 100
        print(f"    {km:>7,} km  vs {int(REF_KM):,} km ref : {pct:+.0f}% value")
    known = _plausible_km(df["odometer_km"]).sum()
    print(f"[+] {known} / {len(df)} listings have a usable odometer")


if __name__ == "__main__":
    main()
