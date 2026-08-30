"""
Write data/clusters/vehicle_type_requests.json - the listings the `classify-vehicle`
skill should hand-tag.

Targets, kept deliberately small:
  * everything the keyword rule (src/ml/vehicle_type.py) currently drops as a
    non-car - so a human can catch a false positive (a real car wrongly dropped);
  * kept listings whose title names no recognizable car make, or is too terse to
    tell - the residue the rule can't call.

Classic / vintage cars that the frozen cluster model just mislabels are NOT
surfaced here - the label-mismatch guard in valuation already keeps those out of
alerts, and they're still cars.

    python scripts/vehicle_type_requests.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml.load_data import _to_price, load_raw_listings  # noqa: E402
from src.ml.vehicle_type import _load_curated, looks_non_car  # noqa: E402

OUT = "data/clusters/vehicle_type_requests.json"

_CAR_MAKES = re.compile(
    r"\b(honda|toyota|ford|chev|chevy|chevrolet|gmc|nissan|mazda|hyundai|kia|"
    r"volkswagen|vw|bmw|mercedes|benz|audi|subaru|lexus|acura|infiniti|dodge|"
    r"jeep|ram|chrysler|buick|cadillac|tesla|volvo|mini|mitsubishi|fiat|porsche|"
    r"jaguar|land\s?rover|genesis|lincoln|pontiac|saturn|scion|suzuki|"
    r"plymouth|mercury|oldsmobile|hummer|saab|mg|datsun|austin)\b",
    re.IGNORECASE,
)


def _raw_rows() -> list[dict]:
    import pandas as pd

    frames = []
    for p in Path("data/raw").glob("facebook_*_raw_*.csv"):
        frames.append(pd.read_csv(p, dtype=str))
    raw = pd.concat(frames, ignore_index=True)
    raw["price"] = raw["raw_price"].map(_to_price)
    raw = raw.dropna(subset=["url"]).drop_duplicates("url")
    out = []
    for r in raw.itertuples(index=False):
        m = re.search(r"/marketplace/item/(\d+)", str(r.url))
        if m:
            out.append({"item_id": m.group(1), "title": str(r.raw_title or ""),
                        "price": None if pd.isna(r.price) else float(r.price)})
    return out


def main() -> None:
    curated = _load_curated()
    seen: set[str] = set()
    rows = []
    for rec in _raw_rows():
        iid, title = rec["item_id"], rec["title"]
        if iid in curated or iid in seen or not title:
            continue
        dropped = looks_non_car(title)
        has_make = bool(_CAR_MAKES.search(title))
        terse = len(title.split()) <= 2
        if not (dropped or not has_make or terse):
            continue
        seen.add(iid)
        rows.append({**rec, "rule_drops_as_non_car": dropped})

    # everything the load step already used
    load_raw_listings()  # sanity: config + data present

    payload = {
        "instructions": (
            "Tag each listing's vehicle type: car | truck | van | suv | motorcycle "
            "| atv | boat | trailer | equipment | rv | other. Only car/truck/van/suv "
            "get priced. `rule_drops_as_non_car` is the current keyword verdict - "
            "confirm or correct it. Merge into data/clusters/vehicle_type.json."
        ),
        "listings": sorted(rows, key=lambda x: x["title"].lower()),
    }
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    n_drop = sum(r["rule_drops_as_non_car"] for r in rows)
    print(f"[+] {len(rows)} listings to review ({n_drop} the rule drops, "
          f"{len(rows) - n_drop} kept-but-unclear) -> {OUT}")


if __name__ == "__main__":
    main()
