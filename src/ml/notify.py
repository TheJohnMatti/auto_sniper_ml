"""
Push new high-`deal_score` listings to your phone via ntfy (https://ntfy.sh).

ntfy is the right fit here: no account, no API key, free apps for iOS / Android,
and a message is just an HTTP POST to ``<server>/<topic>``. Subscribe the phone
app to the same topic and every deal shows up as a tappable notification that
opens the listing.

    python -m src.ml.notify                 # send everything new since last run
    python -m src.ml.notify --dry-run       # print what would be sent, send nothing
    python -m src.ml.notify --min-score 0.9 --limit 5
    python -m src.ml.notify --all           # re-send even things already notified

Reads:  data/processed/deals.csv     (from src.ml.valuation - already filtered of
                                      suspects / outliers / dealer ads / stale)
State:  SQLite at $NOTIFY_DB (default data/processed/notified.sqlite3). A row
        with `sent_at` set is NEVER sent again - the DB's PRIMARY KEY + a
        claim-before-send handshake make that hold even with overlapping cron
        runs or a crash mid-send. In a container, point NOTIFY_DB at a mounted
        volume so the guarantee survives restarts.
Config: ml_pipeline.notifications in config.yaml (thresholds, defaults)
Env:    NTFY_TOPIC   your topic - REQUIRED; overrides config.yaml. This repo is
                     public, so the real topic goes in a local .env, not a file.
        NTFY_SERVER  default https://ntfy.sh; set for a self-hosted server
        NTFY_TOKEN   only for protected / self-hosted topics
        NOTIFY_DB    path to the state DB (see above)
A local .env is auto-loaded if python-dotenv is installed.

Topics on the public ntfy.sh server are unauthenticated - anyone who knows the
name can read it (and post to it). Use a long random topic or self-host.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

import pandas as pd

try:
    from dotenv import load_dotenv

    load_dotenv()  # pick up NTFY_TOPIC / NTFY_SERVER / NTFY_TOKEN from a local .env
except ImportError:
    pass

from src.ml.config import ml_config

DEALS_PATH = "data/processed/deals.csv"
DEFAULT_DB_PATH = "data/processed/notified.sqlite3"
LEGACY_JSON_PATH = "data/processed/notified.json"
_ITEM_ID_RE = re.compile(r"/marketplace/item/(\d+)")
_PRUNE_AFTER_DAYS = 180        # drop delivered rows older than this (relists get new ids)
_STALE_CLAIM_SECONDS = 3600    # an unsent claim older than this = a crashed run; reclaim it


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _age_seconds(iso: str | None) -> float:
    if not iso:
        return float("inf")
    try:
        return (datetime.now(timezone.utc) - datetime.fromisoformat(iso)).total_seconds()
    except ValueError:
        return float("inf")


# --- config -----------------------------------------------------------------

_DEFAULTS = {
    "enabled": True,
    "server": "https://ntfy.sh",
    "topic": "",
    "min_deal_score": 0.6,
    "priority_threshold": 0.9,  # deal_score >= this -> high-priority ping
    "max_per_run": 10,
    "max_alert_age_days": 3,    # a deal older than this is probably gone - don't ping
}

_OBSERVE_DB = "data/processed/observations.sqlite3"


def _cfg() -> dict:
    raw = ml_config().get("notifications", {}) or {}
    ntfy = raw.get("ntfy", {}) or {}
    # env wins over config.yaml - the repo is public, so the real topic lives in
    # a local .env / the environment, never a committed file.
    server = os.environ.get("NTFY_SERVER") or ntfy.get("server") or _DEFAULTS["server"]
    topic = os.environ.get("NTFY_TOPIC") or ntfy.get("topic") or _DEFAULTS["topic"]
    return {
        "enabled": raw.get("enabled", _DEFAULTS["enabled"]),
        "server": server.rstrip("/"),
        "topic": topic,
        "min_deal_score": float(raw.get("min_deal_score", _DEFAULTS["min_deal_score"])),
        "priority_threshold": float(
            ntfy.get("priority_threshold", _DEFAULTS["priority_threshold"])
        ),
        "max_per_run": int(raw.get("max_per_run", _DEFAULTS["max_per_run"])),
        "max_alert_age_days": float(
            raw.get("max_alert_age_days", _DEFAULTS["max_alert_age_days"])
        ),
    }


def _still_listed(db_path: str = _OBSERVE_DB) -> set[str] | None:
    """item_ids seen in the most recent scrape pass and not marked gone. A deal
    on a listing that's already dropped off the feed is noise - it sold."""
    if not os.path.exists(db_path):
        return None
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        latest = con.execute("SELECT MAX(seen_at) FROM observations").fetchone()[0]
        if not latest:
            return None
        # "current" = seen within 90 min of the newest observation (covers a
        # slow scan cadence) and no gone_at.
        cutoff = (pd.Timestamp(latest) - pd.Timedelta(minutes=90)).isoformat()
        rows = con.execute(
            "SELECT item_id FROM listing_history WHERE gone_at IS NULL AND last_seen >= ?",
            (cutoff,),
        ).fetchall()
        con.close()
        return {r[0] for r in rows}
    except sqlite3.Error:
        return None


# --- state store (SQLite) --------------------------------------------------
#
# One row per item_id. `sent_at IS NOT NULL` == delivered, and that is a
# permanent, idempotent record: claim() will never hand the same item out for
# sending twice. The flow per item is:
#
#   claim()      INSERT OR IGNORE; we own it iff we inserted the row (or the
#                existing row is an abandoned, unsent claim we then steal).
#   mark_sent()  stamp sent_at once ntfy accepted it.
#   release()    delete our unsent claim if the POST failed, so it retries next run.

_SCHEMA = """
CREATE TABLE IF NOT EXISTS notified (
    item_id       TEXT PRIMARY KEY,
    first_seen_at TEXT NOT NULL,
    sent_at       TEXT,
    deal_score    REAL,
    title         TEXT
);
CREATE INDEX IF NOT EXISTS notified_sent_at ON notified(sent_at);
"""


def db_path() -> str:
    return os.environ.get("NOTIFY_DB") or DEFAULT_DB_PATH


def connect(path: str | None = None) -> sqlite3.Connection:
    path = path or db_path()
    if path != ":memory:":
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path, timeout=30, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.executescript(_SCHEMA)
    _migrate_legacy_json(conn)
    return conn


def _migrate_legacy_json(conn: sqlite3.Connection) -> None:
    if not os.path.exists(LEGACY_JSON_PATH):
        return
    have = conn.execute("SELECT COUNT(*) FROM notified").fetchone()[0]
    if have:
        return
    try:
        with open(LEGACY_JSON_PATH, "r", encoding="utf-8") as fh:
            sent = json.load(fh).get("sent", {})
    except (json.JSONDecodeError, OSError):
        return
    rows = [
        (iid, meta.get("at") or _utcnow(), meta.get("at") or _utcnow(), meta.get("deal_score"))
        for iid, meta in sent.items()
    ]
    if rows:
        conn.executemany(
            "INSERT OR IGNORE INTO notified(item_id, first_seen_at, sent_at, deal_score) "
            "VALUES (?,?,?,?)",
            rows,
        )
        print(f"[i] migrated {len(rows)} item_ids from {LEGACY_JSON_PATH} into the state DB")


def already_sent_ids(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT item_id FROM notified WHERE sent_at IS NOT NULL")}


def claim(conn: sqlite3.Connection, item_id: str, score: float, title: str) -> bool:
    """Atomically take ownership of sending `item_id`. False => leave it alone."""
    now = _utcnow()
    cur = conn.execute(
        "INSERT OR IGNORE INTO notified(item_id, first_seen_at, deal_score, title) "
        "VALUES (?,?,?,?)",
        (item_id, now, score, title),
    )
    if cur.rowcount == 1:
        return True
    row = conn.execute(
        "SELECT sent_at, first_seen_at FROM notified WHERE item_id=?", (item_id,)
    ).fetchone()
    if row is None or row[0] is not None:
        return False  # already delivered - the guarantee
    # an unsent claim exists: a concurrent run has it, or a previous run crashed.
    if _age_seconds(row[1]) < _STALE_CLAIM_SECONDS:
        return False
    stolen = conn.execute(
        "UPDATE notified SET first_seen_at=? WHERE item_id=? AND sent_at IS NULL",
        (now, item_id),
    )
    return stolen.rowcount == 1


def mark_sent(conn: sqlite3.Connection, item_id: str) -> None:
    conn.execute("UPDATE notified SET sent_at=? WHERE item_id=?", (_utcnow(), item_id))


def release(conn: sqlite3.Connection, item_id: str) -> None:
    conn.execute("DELETE FROM notified WHERE item_id=? AND sent_at IS NULL", (item_id,))


def prune(conn: sqlite3.Connection, days: int = _PRUNE_AFTER_DAYS) -> None:
    cutoff = (datetime.now(timezone.utc) - pd.Timedelta(days=days)).isoformat(timespec="seconds")
    conn.execute("DELETE FROM notified WHERE sent_at IS NOT NULL AND sent_at < ?", (cutoff,))


# --- formatting -------------------------------------------------------------

def _item_id(url: str) -> str | None:
    m = _ITEM_ID_RE.search(url or "")
    return m.group(1) if m else None


def _money(value) -> str:
    try:
        return f"${float(value):,.0f}"
    except (TypeError, ValueError):
        return "?"


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


def format_deal(row: pd.Series) -> dict:
    """-> {title, body, tags, score, url} for one deal."""
    off = ""
    if pd.notna(row.get("discount_pct")):
        off = f" · {row['discount_pct'] * 100:.0f}% under comps"

    comps = row.get("entity_comps")
    thin = pd.notna(comps) and comps <= 5
    marker = "⚠ " if thin else ""
    title = f"{marker}{row.get('entity_label', 'car')} — {_money(row.get('price'))}{off}"

    bits = []
    # show the seller's own title when it doesn't obviously match the model we priced it as
    raw = _norm(row.get("raw_title"))
    ent = _norm(row.get("entity_label"))
    ent_model = " ".join(w for w in ent.split() if not w.isdigit())
    if raw and ent_model and ent_model.split()[-1] not in raw:
        bits.append(f'listed as "{str(row.get("raw_title")).strip()}"')

    bits.append(f"comps median {_money(row.get('entity_median'))} (n={int(comps) if pd.notna(comps) else '?'})")
    km = row.get("odometer_km")
    if pd.notna(km):
        bits.append(f"{float(km) / 1000:.0f}k km")
    age = row.get("listing_age_days")
    if pd.notna(age):
        bits.append("listed today" if age < 1 else f"listed {age:.0f}d ago")
    if pd.notna(row.get("condition_score")) and abs(row["condition_score"]) >= 0.2:
        bits.append(f"condition {row['condition_score']:+.2f}")
    flags = str(row.get("red_flags") or "").strip()
    has_flags = bool(flags) and flags.lower() != "nan"
    if has_flags:
        bits.append(f"flags: {flags.replace(';', ', ')}")
    if row.get("seller_marked_down"):
        bits.append("seller already dropped price")
    loc = str(row.get("location") or "").strip()
    if loc:
        bits.append(loc)
    bits.append(f"deal_score {row.get('deal_score', float('nan')):.2f}")

    return {
        "title": title,
        "body": " · ".join(bits),
        "tags": ["dart", "warning"] if (has_flags or thin) else ["dart"],
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
    with urllib.request.urlopen(req, timeout=15) as resp:
        if resp.status >= 300:
            raise RuntimeError(f"ntfy returned HTTP {resp.status}")


# --- orchestration -------------------------------------------------------

def candidate_deals(
    deals: pd.DataFrame, done: set[str], min_score: float, limit: int, ignore_state: bool,
    max_age_days: float = 3.0,
) -> tuple[list[pd.Series], int, int]:
    df = deals.copy()
    df["item_id"] = df["url"].map(_item_id)
    df = df[df["item_id"].notna() & (df["deal_score"] >= min_score)]
    if not ignore_state:
        df = df[~df["item_id"].isin(done)]

    # Freshness: a deal you can't act on is worse than no deal. Drop listings
    # that are too old, or that have already dropped off the feed (sold).
    before = len(df)
    if "listing_age_days" in df.columns:
        age = pd.to_numeric(df["listing_age_days"], errors="coerce")
        df = df[age.isna() | (age <= max_age_days)]
    live = _still_listed()
    if live is not None:
        df = df[df["item_id"].isin(live)]
    stale_dropped = before - len(df)

    df = df.drop_duplicates("item_id").sort_values("deal_score", ascending=False)
    total = len(df)
    return [row for _, row in df.head(limit).iterrows()], total, stale_dropped


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dry-run", action="store_true", help="print, don't send, don't touch state")
    ap.add_argument("--all", dest="ignore_state", action="store_true",
                    help="consider deals already notified too (claim() still blocks a true double-send)")
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
            print("[!] No ntfy topic set. Put `NTFY_TOPIC=<a-long-random-string>` in a "
                  "local .env (or your environment), then subscribe the ntfy app to it.",
                  file=sys.stderr)
            return 1

    deals = pd.read_csv(DEALS_PATH)

    if args.dry_run:
        conn = None
        done: set[str] = set()
    else:
        conn = connect()
        prune(conn)
        done = already_sent_ids(conn)

    candidates, total, stale_dropped = candidate_deals(
        deals, done, min_score, limit, args.ignore_state, cfg["max_alert_age_days"]
    )

    if not candidates:
        extra = f", {stale_dropped} dropped as stale/gone" if stale_dropped else ""
        print(f"[i] No fresh deals at deal_score >= {min_score:.2f} "
              f"({len(deals)} in deals.csv, {len(done)} already notified{extra}).")
        return 0

    dest = "" if args.dry_run else f" -> {cfg['server']}/{cfg['topic']}"
    verb = "previewing" if args.dry_run else "sending"
    print(f"[+] {total} candidate deal(s) >= {min_score:.2f}; {verb} up to {len(candidates)}{dest}")

    sent = skipped = errors = 0
    for row in candidates:
        msg = format_deal(row)
        item_id = str(row["item_id"])
        high = msg["score"] >= cfg["priority_threshold"]
        print(f"  {'!!' if high else ' ·'} {msg['title']}\n       {msg['body']}\n       {msg['url']}")

        if args.dry_run:
            continue
        if not claim(conn, item_id, round(msg["score"], 3), msg["title"]):
            print("       [i] already notified / claimed elsewhere - skipping")
            skipped += 1
            continue
        try:
            _send_ntfy(msg, cfg, high)
        except (urllib.error.URLError, RuntimeError, TimeoutError, OSError) as e:
            print(f"       [!] send failed, will retry next run: {e}", file=sys.stderr)
            release(conn, item_id)
            errors += 1
            continue
        mark_sent(conn, item_id)
        sent += 1

    if not args.dry_run:
        conn.close()
        more = total - len(candidates)
        tail = f"; {more} over --limit (next run)" if more > 0 else ""
        print(f"[notify] sent={sent} skipped={skipped} errors={errors}{tail}")
        if errors and not sent:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
