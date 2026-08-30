"""
Is this listing actually a car/truck we can price?

The Facebook `/marketplace/category/vehicles` feed is *vehicles*, not *cars*:
motorcycles, ATVs, snowmobiles, jet skis, boats, trailers, campers, RVs, and
heavy/farm equipment all show up. Pricing a $7,500 CBR against $16k Accords
produces a fake "53% under market" deal (this happened). Everything non-car has
to be dropped before clustering.

Two layers:
  1. `looks_non_car(title)` - a high-precision keyword rule. Deliberately narrow:
     it must never drop a real car (a "Ram Rebel", a "Mazda MX-5"), so it only
     fires on tokens that are unambiguous off-road / marine / towed / moto.
  2. `data/clusters/vehicle_type.json` - a curated {item_id: type} map the
     `classify-vehicle` skill fills for the ambiguous cases the rule can't call.

`type` is one of: car, truck, van, suv, motorcycle, atv, boat, trailer,
equipment, rv, other. Only the first four are priced.
"""
from __future__ import annotations

import json
import os
import re

CURATED_PATH = "data/clusters/vehicle_type.json"
PRICED_TYPES = {"car", "truck", "van", "suv"}

# Unambiguous non-car tokens. Every entry here would be bizarre in a real car
# listing title. Keep it that way - when in doubt, leave it out and let the
# curated map / clustering handle it.
_NON_CAR = re.compile(
    r"""\b(
        # --- marine ---
        boat|pontoon|sailboat|kayak|canoe|dinghy|outboard|trolling\ motor|
        jet\s?ski|sea-?doo|waverunner|personal\ watercraft|\bpwc\b|
        # --- off-road / powersports ---
        atv|quad(?:\ ?bike)?|four\s?wheeler|\butv\b|\bsxs\b|side[-\ ]by[-\ ]side|
        dirt\s?bike|dirtbike|pit\s?bike|mini\s?bike|pocket\s?bike|
        snowmobile|snow\s?mobile|ski-?doo|sled\b|
        moped|\bscooter\b|vespa|\bmo-?ped\b|
        # --- motorcycle model/brand tells (car makes excluded) ---
        motorcycle|motorbike|\bcbr\b|\bgsx-?r\b|gsxr|\byzf\b|\bninja\b|\bgrom\b|
        \brebel\ (?:250|300|500|1100)\b|\bshadow\ (?:vt|aero|spirit)\b|
        harley|harley-davidson|sportster|softail|street\ glide|road\ (?:king|glide)|
        \bdyna\b|fat\ boy|iron\ 883|\bducati\b|panigale|multistrada|
        \bktm\b|\brc\ ?390\b|husqvarna|husaberg|\baprilia\b|\bmv\ agusta\b|
        triumph\ (?:bonneville|street\ triple|tiger|rocket)|
        # --- towed ---
        \btrailer\b|camper|\brv\b|motorhome|travel\ trailer|fifth\ wheel|
        5th\ wheel|toy\ hauler|\btoyhauler\b|pop-?up\ camper|teardrop\ camper|
        # --- equipment / farm ---
        excavator|skid\ ?steer|\bbobcat\b|backhoe|\bloader\b|bulldozer|
        forklift|telehandler|\bmini\ ex\b|
        tractor|kubota|john\ deere|\bmassey\b|new\ holland|case\ ih|
        combine|\bswather\b|\bbaler\b|\bplow\b|\btiller\b|
        zero\s?turn|riding\ mower|lawn\ (?:mower|tractor)|
        golf\ cart|\bgator\b|side\ dump|\bskidsteer\b
    )\b""",
    re.IGNORECASE | re.VERBOSE,
)

# Positive car/truck signals that veto a weak non-car match (e.g. "Ram Rebel").
_CAR_VETO = re.compile(
    r"\b(sedan|hatchback|coupe|wagon|minivan|crossover|"
    r"ram\ (?:1500|2500|3500|rebel|laramie|bighorn)|"
    r"f-?150|f-?250|f-?350|silverado|sierra|tacoma|tundra|ridgeline|colorado|"
    r"mx-?5|miata|4matic|xdrive|quattro|awd\ sedan)\b",
    re.IGNORECASE,
)


def _load_curated(path: str = CURATED_PATH) -> dict[str, str]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    if isinstance(raw, dict) and "types" in raw:
        raw = raw["types"]
    return {str(k): str(v).strip().lower() for k, v in raw.items()}


def looks_non_car(title: str) -> bool:
    """Keyword rule. True only when the title is clearly not a car/truck."""
    if not isinstance(title, str) or not title.strip():
        return False
    if _CAR_VETO.search(title):
        return False
    return bool(_NON_CAR.search(title))


def is_priceable(item_id: str | None, title: str, curated: dict[str, str] | None = None) -> bool:
    """Curated verdict wins; otherwise fall back to the keyword rule."""
    curated = _load_curated() if curated is None else curated
    if item_id and item_id in curated:
        return curated[item_id] in PRICED_TYPES
    return not looks_non_car(title)
