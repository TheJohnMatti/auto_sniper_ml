"""
Curate the ambiguous cases through `interject` instead of a bespoke JSON handoff.

This replaces the request/response file dance that the `label-clusters` and
`classify-vehicle` skills implement by hand. Instead of writing a request file,
waiting for an agent to read it and produce a response file, each open question
is registered with an interject daemon, answered from wherever you happen to be
(phone, CLI, web), and merged back into the same curated maps the pipeline
already reads. Nothing downstream changes: `src/ml/vehicle_type.py` still reads
`data/clusters/vehicle_type.json`, and the pipeline still reads
`data/clusters/label_map.json`.

It never blocks. Every run does two things and exits:

  1. merges any question that has been answered since last time into the map;
  2. registers up to `--limit` new questions for items still unknown.

So it is safe to call from a scheduled job: answers accumulate at human pace
while the pipeline keeps running on what it already knows.

    python scripts/interject_curate.py vehicle-type --limit 25
    python scripts/interject_curate.py cluster-label --limit 25

Requires an interject daemon and the client:

    uv pip install -e ~/code/interject/python
    export INTERJECT_URL=http://127.0.0.1:8787
    export INTERJECT_PROJECT=auto_sniper
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

try:
    import interject
except ImportError:  # pragma: no cover - a clearer message than a traceback
    sys.exit(
        "the interject client is not installed:\n"
        "    uv pip install -e ~/code/interject/python"
    )

ROOT = Path(__file__).resolve().parents[1]

VEHICLE_TYPES = [
    "car", "truck", "van", "suv", "motorcycle", "atv",
    "boat", "trailer", "equipment", "rv", "other",
]


def _load(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open() as handle:
        return json.load(handle)


def _save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(path)


_UNANSWERED = object()


def _resolve(question_id: str, context: dict[str, Any]) -> Any:
    """The stored answer for this question, or `_UNANSWERED`.

    Uses `peek` rather than `ask` on purpose: `ask` registers as it checks, so
    using it to scan a backlog of 719 items would register all 719 regardless of
    any limit. `peek` looks without creating.
    """
    snapshot = interject.peek(interject.key_for(question_id, context))
    if snapshot is None or snapshot.get("state") != "answered":
        return _UNANSWERED
    return (snapshot.get("answer") or {}).get("value")


def _register(prompt: str, **kwargs: Any) -> None:
    """Register a question without waiting for it.

    `wait=0` plus catching Suspended is what makes this non-blocking, and
    registration is idempotent on the question key, so re-running is free.
    """
    try:
        interject.ask(prompt, wait=0, **kwargs)
    except (interject.Suspended, interject.Expired):
        pass


def curate_vehicle_types(limit: int) -> int:
    requests = _load(ROOT / "data/clusters/vehicle_type_requests.json")
    if not requests:
        print("no vehicle_type_requests.json — run scripts/vehicle_type_requests.py first")
        return 1
    curated: dict[str, str] = _load(ROOT / "data/clusters/vehicle_type.json") or {}

    pending = [item for item in requests["listings"] if item["item_id"] not in curated]
    print(f"{len(curated)} already curated, {len(pending)} unknown")

    answered = 0
    registered = 0
    for item in pending:
        # The rule's verdict is shown rather than suggested: it knows a listing
        # is *not* a car, but not which kind of not-a-car, so it cannot propose
        # one of the eleven answers.
        context = {
            "item_id": item["item_id"],
            "title": item["title"],
            "price": item.get("price"),
            "keyword_rule_says_non_car": item.get("rule_drops_as_non_car", False),
        }
        answer = _resolve("vehicle_type", context)
        if answer is not _UNANSWERED:
            curated[item["item_id"]] = str(answer)
            answered += 1
            continue
        if registered >= limit:
            continue
        _register(
            "What kind of vehicle is this?",
            id="vehicle_type",
            options=VEHICLE_TYPES,
            context=context,
            batch_key="auto_sniper.vehicle_type",
            priority=4,
        )
        registered += 1

    if answered:
        _save(ROOT / "data/clusters/vehicle_type.json", curated)
    print(f"merged {answered} new answer(s); {registered} question(s) waiting on a human")
    return 0


def curate_cluster_labels(limit: int) -> int:
    requests = _load(ROOT / "data/clusters/label_requests.json")
    if not requests:
        print("no label_requests.json — run `python -m src.ml.run_pipeline` first")
        return 1
    label_map: dict[str, str] = _load(ROOT / "data/clusters/label_map.json") or {}

    pending = [c for c in requests["clusters"] if str(c["cluster"]) not in label_map]
    print(f"{len(label_map)} already labelled, {len(pending)} unknown")

    answered = 0
    registered = 0
    for cluster in pending:
        samples = cluster["sample_titles"][:6]
        guess = cluster.get("heuristic_guess")
        # Sample titles are part of the context and therefore part of the
        # question key: a re-cluster that changes a cluster's members is a
        # genuinely different question, and gets asked again.
        context = {
            "cluster": cluster["cluster"],
            "n_listings": cluster["n_listings"],
            "sample_titles": samples,
        }
        answer = _resolve("cluster_label", context)
        if answer is not _UNANSWERED:
            label_map[str(cluster["cluster"])] = str(answer)
            answered += 1
            continue
        if registered >= limit:
            continue
        _register(
            "What vehicle is this cluster? Answer as 'Make Model'.",
            id="cluster_label",
            kind="text",
            context=context,
            # The heuristic already proposes an answer, which is exactly what the
            # triage layer needs in order to learn when not to ask at all.
            suggest={"value": guess, "confidence": 0.7} if guess else None,
            batch_key="auto_sniper.cluster_label",
            priority=6,
        )
        registered += 1

    if answered:
        _save(ROOT / "data/clusters/label_map.json", label_map)
    print(f"merged {answered} new answer(s); {registered} question(s) waiting on a human")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("what", choices=["vehicle-type", "cluster-label"])
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="how many new questions to register in one run (default 25)",
    )
    args = parser.parse_args()

    if not os.environ.get("INTERJECT_URL"):
        print("INTERJECT_URL is not set; using the default http://127.0.0.1:8787")
    os.environ.setdefault("INTERJECT_PROJECT", "auto_sniper")

    if args.what == "vehicle-type":
        return curate_vehicle_types(args.limit)
    return curate_cluster_labels(args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
