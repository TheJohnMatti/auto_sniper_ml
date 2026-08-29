"""
Geographic gate for production inference.

We only want to be pinged about deals we could actually go buy - southwestern
Ontario, centred on London. Facebook shows each listing's location as a
"Town, PROV" line; this module maps that town to coordinates and keeps the
listing if it's within `radius_km` of a centre point (both from
`ml_pipeline.inference.region` in config.yaml).

Town -> coordinate is a static table (no geocoder, no network). A town we don't
recognise is treated as out-of-region, except that anything mentioning "London"
(and not London, UK) always passes - that's the bullseye.
"""
from __future__ import annotations

import math
import re

from src.ml.config import ml_config

# London, ON and the SW Ontario belt around it (lat, lon). Rough municipal
# centroids - precise enough for a ~100 km gate.
_TOWN_COORDS: dict[str, tuple[float, float]] = {
    "london": (42.9849, -81.2453),
    "st thomas": (42.7792, -81.1810),
    "woodstock": (43.1306, -80.7467),
    "ingersoll": (43.0392, -80.8836),
    "tillsonburg": (42.8622, -80.7269),
    "aylmer": (42.7686, -80.9828),
    "strathroy": (42.9575, -81.6169),
    "strathroy-caradoc": (42.9575, -81.6169),
    "mount brydges": (42.9019, -81.4838),
    "komoka": (42.9500, -81.4167),
    "dorchester": (43.0000, -81.0667),
    "lucan": (43.1806, -81.4000),
    "lucan biddulph": (43.1806, -81.4000),
    "exeter": (43.3500, -81.4833),
    "parkhill": (43.1594, -81.6869),
    "grand bend": (43.3167, -81.7500),
    "ilderton": (43.0500, -81.4167),
    "thorndale": (43.1167, -81.1667),
    "belmont": (42.8833, -81.0833),
    "port stanley": (42.6667, -81.2167),
    "sarnia": (42.9994, -82.3089),
    "point edward": (43.0000, -82.4000),
    "corunna": (42.8833, -82.4333),
    "petrolia": (42.8792, -82.1461),
    "wyoming": (42.9500, -82.1167),
    "watford": (42.9500, -81.8833),
    "forest": (43.1000, -82.0000),
    "chatham": (42.4048, -82.1910),
    "chatham-kent": (42.4048, -82.1910),
    "wallaceburg": (42.5942, -82.3906),
    "blenheim": (42.3333, -81.9833),
    "ridgetown": (42.4394, -81.8878),
    "dresden": (42.5872, -82.1836),
    "tilbury": (42.2583, -82.4361),
    "glencoe": (42.7500, -81.7167),
    "wardsville": (42.6500, -81.7500),
    "west lorne": (42.6000, -81.5833),
    "rodney": (42.5667, -81.6833),
    "stratford": (43.3701, -80.9821),
    "st marys": (43.2586, -81.1414),
    "mitchell": (43.4667, -81.2000),
    "seaforth": (43.5500, -81.4000),
    "clinton": (43.6333, -81.5333),
    "goderich": (43.7501, -81.7165),
    "bayfield": (43.5620, -81.6989),
    "listowel": (43.7369, -80.9531),
    "norwich": (42.9847, -80.5967),
    "simcoe": (42.8375, -80.3036),
    "delhi": (42.8536, -80.5011),
    "port dover": (42.7833, -80.2000),
    "kitchener": (43.4516, -80.4925),
    "waterloo": (43.4643, -80.5204),
    "cambridge": (43.3616, -80.3144),
    "brantford": (43.1394, -80.2644),
    "paris": (43.1934, -80.3841),
    "woodstock ontario": (43.1306, -80.7467),
    "windsor": (42.3149, -83.0364),
    "lasalle": (42.2333, -83.0667),
    "tecumseh": (42.3100, -82.9250),
    "amherstburg": (42.1000, -83.1000),
    "leamington": (42.0536, -82.5994),
    "kingsville": (42.0392, -82.7397),
    "essex": (42.1750, -82.8250),
}

_UK_HINT = re.compile(r"\b(uk|united kingdom|england|gb)\b", re.I)
_LOC_RE = re.compile(r"^\s*(.*?)\s*,\s*[A-Za-z]{2,3}\.?\s*$")


def _norm_town(text: str) -> str | None:
    """'St. Thomas, ON' -> 'st thomas'. Returns None if it isn't a Town, PROV line."""
    if not isinstance(text, str) or not text.strip():
        return None
    m = _LOC_RE.match(text)
    town = (m.group(1) if m else text).lower()
    town = town.replace(".", "").replace("'", "")
    town = re.sub(r"[^a-z\s-]", " ", town)
    town = re.sub(r"\s+", " ", town).strip()
    return town or None


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371.0088 * math.asin(math.sqrt(h))


def _region_cfg() -> tuple[tuple[float, float], float]:
    reg = (ml_config().get("inference", {}) or {}).get("region", {}) or {}
    center = tuple(reg.get("center", (42.9849, -81.2453)))
    return center, float(reg.get("radius_km", 115))


def in_region(location: str, center: tuple[float, float] | None = None,
              radius_km: float | None = None) -> bool:
    """True if the listing's location is inside the inference region."""
    if center is None or radius_km is None:
        cfg_center, cfg_radius = _region_cfg()
        center = center or cfg_center
        radius_km = cfg_radius if radius_km is None else radius_km

    if not isinstance(location, str) or not location.strip():
        return False
    low = location.lower()
    if _UK_HINT.search(low):
        return False  # "London, UK" and friends - hard no
    if "london" in low:
        return True   # the bullseye - always in region
    town = _norm_town(location)
    if town is None:
        return False
    coord = _TOWN_COORDS.get(town) or _TOWN_COORDS.get(town.replace("-", " "))
    if coord is None:
        return False
    return haversine_km(coord, center) <= radius_km


if __name__ == "__main__":
    center, radius = _region_cfg()
    print(f"region: within {radius} km of {center}")
    for s in ["London, ON", "St. Thomas, ON", "Sarnia, ON", "Stratford, ON",
              "Windsor, ON", "Toronto, ON", "Chatham-Kent, ON", "London, UK",
              "Kitchener, ON", "Goderich, ON", "Barrie, ON", ""]:
        print(f"  {'IN ' if in_region(s, center, radius) else 'out'}  {s!r}")
