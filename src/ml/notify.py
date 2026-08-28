"""
Push new high-`deal_score` listings to your phone via ntfy (https://ntfy.sh).

ntfy is the right fit here: no account, no API key, free apps for iOS / Android,
and a message is just an HTTP POST to ``<server>/<topic>``. Subscribe the phone
app to the same topic and every deal shows up as a tappable notification that
opens the listing.

    python -m src.ml.notify                 # send everything new since last run
    python -m src.ml.notify --dry-run       # print what would be sent, send nothing
    python -m src.ml.notify --min-score 0.9 --limit 5
    python -m src.ml.notify --all           # ignore the "already sent" state

Reads:  data/processed/deals.csv         (from src.ml.valuation - already filtered
                                          of suspects / outliers / dealer ads)
State:  data/processed/notified.json     item_ids already pushed, so re-runs and
                                          cron don't spam you
Config: ml_pipeline.notifications in config.yaml
Auth:   env NTFY_TOKEN  (only for protected or self-hosted topics)

Topics on the public ntfy.sh server are unauthenticated - anyone who guesses the
name can read it. Use a long random topic (see config.yaml) or self-host.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.ml.config import ml_config

DEALS_PATH = "data/processed/deals.csv"
STATE_PATH = "data/processed/notified.json"
_ITEM_ID_RE = re.compile(r"/marketplace/item/(\d+)")
_STATE_TTL_DAYS = 60  # forget sent item_ids older than this so the file stays small


# --- config -----------------------------------------------------------------

_DEFAULTS = {
    "enabled": True,
    "server": "https://ntfy.sh",
    "topic": "",
    "min_deal_score": 0.6,
    "priority_threshold": 0.9,  # deal_score >= this -> high-priority ping
    "max_per_run": 10,
}


def _cfg() -> dict:
    raw = ml_config().get("notifications", {}) or {}
    ntfy = raw.get("ntfy", {}) or {}
    return {
        "enabled": raw.get("enabled", _DEFAULTS["enabled"]),
        "server": (ntfy.get("server") or _DEFAULTS["server"]).rstrip("/"),
        "topic": ntfy.get("topic") or _DEFAULTS["topic"],
        "min_deal_score": float(raw.get("min_deal_score", _DEFAULTS["min_deal_score"])),
        "priority_threshold": float(
            ntfy.get("priority_threshold", _DEFAULTS["priority_threshold"])
        ),
        "max_per_run": int(raw.get("max_per_run", _DEFAULTS["max_per_run"])),
    }


# --- state ------------------------------------------------------------------

def _load_state(path: str = STATE_PATH) -> dict:
    if not os.path.exists(path):
        return {"sent": {}}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            state = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {"sent": {}}
    state.setdefault("sent", {})
    cutoff = datetime.now(timezone.utc) - timedelta(days=_STATE_TTL_DAYS)
    state["sent"] = {
        iid: meta
        for iid, meta in state["sent"].items()
        if _parse_iso(meta.get("at")) is None or _parse_iso(meta["at"]) >= cutoff
    }
    return state


def _save_state(state: dict, path: str = STATE_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


# --- formatting -------------------------------------------------------------

def _item_id(url: str) -> str | None:
    m = _ITEM_ID_RE.search(url or "")
    return m.group(1) if m else None


def _money(value) -> str:
    try:
        return f"${float(value):,.0f}"
    except (TypeError, ValueError):
        return "?"


def format_deal(row: pd.Series) -> dict:
    """-> {title, body, tags, priority_hint, url} for one deal."""
    off = ""
    if pd.notna(row.get("discount_pct")):
        off = f" · {row['discount_pct'] * 100:.0f}% under comps"

    title = f"{row.get('entity_label', 'car')} — {_money(row.get('price'))}{off}"

    bits = [f"comps median {_money(row.get('entity_median'))}"]
    km = row.get("odometer_km")
    if pd.notna(km):
        bits.append(f"{float(km) / 1000:.0f}k km")
    if pd.notna(row.get("condition_score")) and abs(row["condition_score"]) >= 0.2:
        bits.append(f"condition {row['condition_score']:+.2f}")
    flags = str(row.get("red_flags") or "").strip()
    if flags and flags.lower() != "nan":
        bits.append(f"flags: {flags.replace(';', ', ')}")
    if row.get("seller_marked_down"):
        bits.append("seller already dropped price")
    loc = str(row.get("location") or "").strip()
    if loc:
        bits.append(loc)
    bits.append(f"deal_score {row.get('deal_score', float('nan')):.2f}")

    tags = ["dart"]
    if flags and flags.lower() != "nan":
        tags.append("warning")

    return {
        "title": title,
        "body": " · ".join(bits),
        "tags": tags,
        "score": float(row.get("deal_score", 0.0) or 0.0),
        "url": str(row.get("url") or ""),
    }


# --- sending --------------------------------------------------------------

def _send_ntfy(msg: dict, cfg: dict, high_priority: bool) -> None:
    endpoint = f"{cfg['server']}/{cfg['topic']}"
    headers = {
        "Title": msg["title"].encode("utf-8"),
        "Tags": ",".join(msg["tags"]),
        "Priority": "high" if high_priority else "default",
    }
    if msg["url"]:
        headers["Click"] = msg["url"]
        headers["Actions"] = f"view, Open listing, {msg['url']}"
    token = os.environ.get("NTFY_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(
        endpoint, data=msg["body"].encode("utf-8"), headers=headers, method="POST"
    )
    with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310 - fixed https endpoint
        if resp.status >= 300:
            raise RuntimeError(f"ntfy returned HTTP {resp.status}")


# --- orchestration -------------------------------------------------------

def pick_new_deals(
    deals: pd.DataFrame, state: dict, min_score: float, limit: int, ignore_state: bool
) -> tuple[list[pd.Series], int]:
    df = deals.copy()
    df["item_id"] = df["url"].map(_item_id)
    df = df[df["item_id"].notna() & (df["deal_score"] >= min_score)]
    if not ignore_state:
        df = df[~df["item_id"].isin(state["sent"])]
    df = df.sort_values("deal_score", ascending=False)
    total = len(df)
    return [row for _, row in df.head(limit).iterrows()], total


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print, don't send, don't record")
    ap.add_argument("--all", dest="ignore_state", action="store_true",
                    help="ignore the already-sent state (still won't double-send within this run)")
    ap.add_argument("--min-score", type=float, default=None, help="override notifications.min_deal_score")
    ap.add_argument("--limit", type=int, default=None, help="override notifications.max_per_run")
    args = ap.parse_args(argv)

    cfg = _cfg()
    min_score = args.min_score if args.min_score is not None else cfg["min_deal_score"]
    limit = args.limit if args.limit is not None else cfg["max_per_run"]

    if not os.path.exists(DEALS_PATH):
        print(f"[!] {DEALS_PATH} not found - run `python -m src.ml.valuation` first.", file=sys.stderr)
        return 1
    if not args.dry_run:
        if not cfg["enabled"]:
            print("[i] notifications.enabled is false - nothing sent. Use --dry-run to preview.")
            return 0
        if not cfg["topic"] or "changeme" in cfg["topic"]:
            print("[!] Set a real ml_pipeline.notifications.ntfy.topic in config.yaml first "
                  "(a long random string), then subscribe the ntfy app to it.", file=sys.stderr)
            return 1

    deals = pd.read_csv(DEALS_PATH)
    state = _load_state()
    new_deals, total = pick_new_deals(deals, state, min_score, limit, args.ignore_state)

    if not new_deals:
        print(f"[i] No new deals at deal_score >= {min_score:.2f} "
              f"({len(deals)} in deals.csv, {len(state['sent'])} already notified).")
        return 0

    dest = "" if args.dry_run else f" -> {cfg['server']}/{cfg['topic']}"
    verb = "previewing" if args.dry_run else "sending"
    print(f"[+] {total} new deal(s) >= {min_score:.2f}; {verb} top {len(new_deals)}{dest}")

    sent = 0
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for row in new_deals:
        msg = format_deal(row)
        high = msg["score"] >= cfg["priority_threshold"]
        marker = "!!" if high else " ·"
        print(f"  {marker} {msg['title']}\n       {msg['body']}\n       {msg['url']}")
        if args.dry_run:
            continue
        try:
            _send_ntfy(msg, cfg, high)
        except (urllib.error.URLError, RuntimeError, TimeoutError) as e:
            print(f"       [!] send failed: {e}", file=sys.stderr)
            continue
        state["sent"][str(row["item_id"])] = {"at": now, "deal_score": round(msg["score"], 3)}
        sent += 1

    if not args.dry_run:
        _save_state(state)
        more = total - len(new_deals)
        print(f"[+] Sent {sent}/{len(new_deals)}"
              + (f"; {more} more over the --limit, will go next run" if more > 0 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
