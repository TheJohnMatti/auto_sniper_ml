"""
Surface entity clusters the pipeline priced but should not have trusted.

Clustering groups listings that are "the same car" so their prices can be
compared. When a cluster is wrong, every listing in it is mispriced against a
meaningless median — and the failure is silent, because a wrong median still
produces a confident-looking number. Two real examples from one run:

    2023 Dodge Caliber   median $75,858   -> actually five Dodge Chargers, one
                                            of them a $215 parts listing
    2026 Porsche Macan   median  $1,200   -> dirt bikes, a gyrocopter, a Kia EV9
                                            and a $169,997 Corvette

The pipeline can *detect* that a cluster is incoherent: a 500x spread between
its cheapest and dearest listing is not a price distribution. What it cannot do
is say what the cluster actually is. That judgment is irreducible, it takes a
person about three seconds, and until someone makes it the pricing is wrong.

So this asks. It never blocks: each run merges answers that have arrived and
registers up to `--limit` new questions, which makes it safe to call from the
scheduled pipeline while answers accumulate at human pace.

    python scripts/interject_watch.py --limit 5

Answers are merged into data/clusters/label_map.json, which is what
src/ml/label.py already reads, so nothing downstream changes.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import interject
except ImportError:  # pragma: no cover - a clearer message than a traceback
    sys.exit("the interject client is not installed:\n    uv pip install interject")

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABELLED = ROOT / "data/processed/listings_labeled.csv"
LABEL_MAP = ROOT / "data/clusters/label_map.json"

#: Below this many listings a cluster has no meaningful price distribution, so
#: its spread says nothing and there is nothing to be suspicious about.
MIN_LISTINGS = 3

#: A real model's listings do not span a 40x price range. Anything wider is a
#: cluster containing more than one thing.
SUSPECT_SPREAD = 40.0

#: The answer meaning "this is not one vehicle; do not price it as one".
MIXED = "MIXED"

#: Deals outside the configured region are dropped before anyone is notified, so
#: a junk cluster made entirely of out-of-region listings cannot change an
#: outcome. Asking about it would be noise, and the whole point is not to spend
#: someone's attention on what does not matter.
REGION = "london_swon"

_MAKES = (
    "honda toyota ford chevrolet chevy gmc nissan mazda hyundai kia volkswagen vw bmw "
    "mercedes audi subaru lexus acura infiniti dodge jeep ram chrysler buick cadillac "
    "tesla volvo mini mitsubishi fiat porsche jaguar genesis lincoln pontiac saturn "
    "scion suzuki plymouth mercury oldsmobile hummer saab datsun"
).split()


def _candidate_labels(titles: list[str], limit: int = 3) -> list[str]:
    """The most common 'Make Model' pairs actually present in the titles.

    Deliberately derived from the listings rather than from the existing label:
    the existing label is the thing under suspicion.
    """
    pairs: Counter[str] = Counter()
    for title in titles:
        tokens = re.findall(r"[a-z0-9\-]+", str(title).lower())
        for i, token in enumerate(tokens[:-1]):
            if token in _MAKES:
                model = tokens[i + 1]
                if model.isdigit() and len(model) == 4:  # a year, not a model
                    continue
                pairs[f"{token.title()} {model.title()}"] += 1
    return [label for label, _ in pairs.most_common(limit)]


def _suspicion(group: pd.DataFrame) -> tuple[float, str] | None:
    """Why this cluster should not be trusted, or None if it looks fine."""
    prices = group["price"].dropna()
    prices = prices[prices > 0]
    if len(prices) < MIN_LISTINGS:
        return None

    spread = float(prices.max() / max(prices.min(), 1.0))
    if spread >= SUSPECT_SPREAD:
        return spread, f"prices span {spread:.0f}x, from ${prices.min():,.0f} to ${prices.max():,.0f}"

    # The other failure mode: a coherent price range under the wrong name, which
    # is worse because nothing about the numbers looks odd.
    label = str(group["entity_label"].iloc[0]).lower()
    model = label.split()[-1] if label.split() else ""
    if model and len(model) > 2:
        titles = " ".join(str(t).lower() for t in group["raw_title"])
        if model not in titles:
            return 1.0, f"no listing's title mentions {model!r}"
    return None


def _load_map() -> dict[str, str]:
    if not LABEL_MAP.exists():
        return {}
    raw = json.loads(LABEL_MAP.read_text())
    return raw.get("labels", raw) if isinstance(raw, dict) else {}


def _save_map(labels: dict[str, str]) -> None:
    LABEL_MAP.parent.mkdir(parents=True, exist_ok=True)
    tmp = LABEL_MAP.with_suffix(".json.tmp")
    # Numeric key order, matching how the file is already written. Sorting these
    # as strings would put "10" after "1" and rewrite all 250 lines, burying a
    # three-line change in a 200-line diff.
    ordered = dict(sorted(labels.items(), key=lambda kv: int(kv[0])))
    tmp.write_text(json.dumps(ordered, indent=2) + "\n")
    tmp.replace(LABEL_MAP)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=5,
                        help="how many new questions to register per run (default 5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be asked, without asking")
    args = parser.parse_args()

    if not LABELLED.exists():
        print(f"no {LABELLED.relative_to(ROOT)} — run the pipeline first")
        return 1

    listings = pd.read_csv(LABELLED)
    labels = _load_map()

    suspects = []
    for cluster, group in listings.groupby("cluster"):
        verdict = _suspicion(group)
        if verdict is None:
            continue
        severity, why = verdict
        in_region = int((group["location"] == REGION).sum())
        if in_region == 0:
            continue
        # Rank by how much a wrong answer here actually costs: how many listings
        # it misprices where anyone would ever see them.
        suspects.append((in_region * severity, int(cluster), group, why, in_region))
    suspects.sort(reverse=True, key=lambda s: s[0])

    total_bad = sum(1 for _, g in listings.groupby("cluster") if _suspicion(g))
    print(f"{total_bad} incoherent cluster(s); {len(suspects)} of them price "
          f"listings in {REGION}, which is where an outcome can change")
    if args.dry_run:
        for _, cluster, group, why, in_region in suspects[: args.limit]:
            print(f"\n  cluster {cluster}: priced as {group['entity_label'].iloc[0]!r}"
                  f"  ({in_region} listing(s) in {REGION})")
            print(f"    {why}")
            for title in list(group["raw_title"])[:4]:
                print(f"      {str(title)[:64]}")
        return 0

    answered = registered = 0
    for _, cluster, group, why, in_region in suspects:
        titles = [str(t) for t in group["raw_title"]]
        current = str(group["entity_label"].iloc[0])
        prices = group["price"].dropna()
        context = {
            "cluster": cluster,
            "priced_as": current,
            "listings": len(group),
            "in_region": in_region,
            "median": float(prices.median()) if len(prices) else None,
            "problem": why,
            "sample_titles": titles[:6],
        }

        # peek, not ask: `ask` registers as it checks, so scanning the whole
        # backlog with it would register every suspect regardless of --limit.
        key = interject.key_for("entity_label", context)
        snapshot = interject.peek(key)
        if snapshot and snapshot.get("state") == "answered":
            value = str((snapshot.get("answer") or {}).get("value") or "").strip()
            if value:
                labels[str(cluster)] = value
                answered += 1
            continue
        if registered >= args.limit:
            continue

        candidates = _candidate_labels(titles)
        options = [c for c in candidates if c.lower() != current.lower()][:2]
        options = [*options, current, MIXED]
        try:
            interject.ask(
                "This cluster is being priced as one vehicle. What is it really?",
                options=options,
                id="entity_label",
                context=context,
                # The commonest make/model in the titles is a decent guess, and
                # giving it lets the triage layer learn when to stop asking.
                suggest={"value": candidates[0], "confidence": 0.6} if candidates else None,
                batch_key="auto_sniper.entity_label",
                priority=7,
                wait=0,
            )
            registered += 1
        except interject.Suspended:
            registered += 1
        except interject.Expired:
            pass

    if answered:
        _save_map(labels)
    print(f"merged {answered} correction(s); {registered} question(s) waiting on a human")
    if answered:
        print("re-run the pipeline to re-price with the corrected labels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
