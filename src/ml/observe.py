"""
Append-only observation log - the substrate for continuous training.

Every scrape pass records what it saw and never overwrites, so over weeks this
accumulates the two things the hourly deal scorer can't see from a single
snapshot:

  * price **trajectories** per listing (did it drop? how fast?)
  * **time on market** - first_seen ... last_seen, and a `gone_at` once a listing
    stops showing up. Since we scrape newest-first and only ~90 listings deep,
    "gone" means *sold, delisted, or aged past our scrape depth* - a noisy but
    directional signal the weekly retrain can model.

    python -m src.ml.observe        # fold the latest data/raw/*.csv into the log

Store: data/processed/observations.sqlite3
    observations       one row per (item_id, seen_at)
    listing_history    one row per item_id: first/last seen, price path, gone_at
"""
from __future__ import annotations

import glob
import os
import sqlite3

import pandas as pd

from src.ml.load_data import RAW_GLOB, _extract_item_id, _to_price

DB_PATH = os.environ.get("OBSERVE_DB") or "data/processed/observations.sqlite3"
GONE_AFTER_DAYS = 3.0   # not seen for this long (with a scrape since) => mark gone_at

_SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    item_id          TEXT NOT NULL,
    seen_at          TEXT NOT NULL,
    price            REAL,
    price_original   REAL,
    title            TEXT,
    location         TEXT,
    listing_location TEXT,
    PRIMARY KEY (item_id, seen_at)
);
CREATE TABLE IF NOT EXISTS listing_history (
    item_id          TEXT PRIMARY KEY,
    first_seen       TEXT,
    last_seen        TEXT,
    n_sightings      INTEGER,
    first_price      REAL,
    last_price       REAL,
    min_price        REAL,
    max_price        REAL,
    title            TEXT,
    location         TEXT,
    listing_location TEXT,
    gone_at          TEXT
);
CREATE INDEX IF NOT EXISTS obs_item ON observations(item_id);
"""


def connect(path: str = DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path, timeout=30, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.executescript(_SCHEMA)
    return conn


def _raw_observations(raw_glob: str = RAW_GLOB) -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(raw_glob)):
        df = pd.read_csv(path, dtype=str)
        for col in ("raw_title", "raw_price", "raw_price_original", "location",
                    "raw_listing_location", "url", "scraped_at"):
            if col not in df.columns:
                df[col] = pd.NA
        frames.append(df)
    if not frames:
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)
    raw["item_id"] = raw["url"].map(_extract_item_id)
    raw["seen_at"] = pd.to_datetime(raw["scraped_at"], errors="coerce", utc=True)
    raw = raw[raw["item_id"].notna() & raw["seen_at"].notna()]
    return pd.DataFrame({
        "item_id": raw["item_id"],
        "seen_at": raw["seen_at"].dt.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "price": raw["raw_price"].map(_to_price),
        "price_original": raw["raw_price_original"].map(_to_price),
        "title": raw["raw_title"].fillna("").str.slice(0, 200),
        "location": raw["location"].fillna(""),
        "listing_location": raw["raw_listing_location"].fillna(""),
    })


def _rebuild_history(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM listing_history")
    conn.execute("""
        INSERT INTO listing_history
        SELECT
            item_id,
            MIN(seen_at), MAX(seen_at), COUNT(*),
            (SELECT price FROM observations o2 WHERE o2.item_id = o.item_id
                 AND price IS NOT NULL ORDER BY seen_at ASC  LIMIT 1),
            (SELECT price FROM observations o2 WHERE o2.item_id = o.item_id
                 AND price IS NOT NULL ORDER BY seen_at DESC LIMIT 1),
            MIN(price), MAX(price),
            (SELECT title FROM observations o2 WHERE o2.item_id = o.item_id
                 ORDER BY LENGTH(COALESCE(title,'')) DESC LIMIT 1),
            (SELECT location FROM observations o2 WHERE o2.item_id = o.item_id
                 ORDER BY seen_at DESC LIMIT 1),
            (SELECT listing_location FROM observations o2 WHERE o2.item_id = o.item_id
                 AND COALESCE(listing_location,'') <> '' ORDER BY seen_at DESC LIMIT 1),
            NULL
        FROM observations o
        GROUP BY item_id
    """)
    # gone_at: last_seen is old enough AND the corpus has moved on since then
    latest = conn.execute("SELECT MAX(seen_at) FROM observations").fetchone()[0]
    if latest:
        cutoff = (
            pd.Timestamp(latest) - pd.Timedelta(days=GONE_AFTER_DAYS)
        ).isoformat()
        conn.execute(
            "UPDATE listing_history SET gone_at = last_seen "
            "WHERE gone_at IS NULL AND last_seen < ?",
            (cutoff,),
        )


def main() -> None:
    obs = _raw_observations()
    if obs.empty:
        print("[i] no raw scrapes to fold in")
        return

    conn = connect()
    before = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    conn.executemany(
        "INSERT OR IGNORE INTO observations "
        "(item_id, seen_at, price, price_original, title, location, listing_location) "
        "VALUES (:item_id, :seen_at, :price, :price_original, :title, :location, :listing_location)",
        obs.to_dict("records"),
    )
    added = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0] - before
    _rebuild_history(conn)

    n_listings, n_gone = conn.execute(
        "SELECT COUNT(*), COUNT(gone_at) FROM listing_history"
    ).fetchone()
    dropped = conn.execute(
        "SELECT AVG(1.0 - last_price/first_price) FROM listing_history "
        "WHERE first_price > 0 AND last_price > 0 AND last_price < first_price"
    ).fetchone()[0]
    conn.close()

    print(f"[+] observations.sqlite3: +{added} sightings, {n_listings} listings tracked")
    print(f"    {n_gone} gone (sold / delisted / aged out); "
          f"avg markdown among those that dropped: "
          f"{(dropped or 0) * 100:.1f}%")


if __name__ == "__main__":
    main()
