"""
Incremental detail-page scrape: fetch the full seller description (and the
structured "About this vehicle" block) for each listing found by run.py.

The category feed only exposes title + price; the description lives on the
individual listing page. Visiting ~2,400 pages is slow and raises bot-detection
risk, so this runs in resumable batches:

    python -m src.scraper.fetch_descriptions                   # config batch size + concurrency
    python -m src.scraper.fetch_descriptions 40                # explicit batch size
    python -m src.scraper.fetch_descriptions --deals           # deal/suspect candidates first
    python -m src.scraper.fetch_descriptions --since=20260828  # only listings from recent scrapes
    python -m src.scraper.fetch_descriptions --concurrency=3   # N pages in flight at once

State lives in data/raw/descriptions.csv (append-only). Each run skips item_ids
already present and processes the next `description_batch_size` listings, newest
scrape first. Concurrency defaults to `scraping.concurrency_limit` in config;
keep it low (2-3) - these are unauthenticated requests from one IP. Downstream:
src/ml/sentiment.py turns raw_description into condition/urgency signals.
"""
import asyncio
import csv
import glob
import os
import random
import re
import sys
from datetime import datetime

from playwright.async_api import TimeoutError as PlaywrightTimeout
from playwright.async_api import async_playwright

RAW_DIR = "data/raw"
OUT_PATH = os.path.join(RAW_DIR, "descriptions.csv")
RAW_GLOB = os.path.join(RAW_DIR, "facebook_*_raw_*.csv")

FIELDS = [
    "item_id", "url", "fetched_at", "status", "listed_age",
    "odometer_km", "transmission", "owners", "raw_description",
]

_ITEM_ID_RE = re.compile(r"/marketplace/item/(\d+)")

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Grab the tightest element that still contains the whole listing card (the one
# holding "Seller's description"), after expanding any "See more" toggles.
_BIG_BLOCK_JS = r"""
() => {
  [...document.querySelectorAll('span,div[role="button"]')]
     .filter(n => (n.innerText || '').trim() === 'See more')
     .forEach(n => { try { n.click(); } catch (e) {} });
  const main = document.querySelector('div[role=main]') || document.body;
  let best = '';
  main.querySelectorAll('div,span').forEach(n => {
    const t = n.innerText || '';
    if (t.includes("Seller's description") && t.length > best.length
        && t.length < 8000 && n.children.length <= 4) best = t;
  });
  return best;
}
"""


def _item_id(url: str) -> str | None:
    m = _ITEM_ID_RE.search(url or "")
    return m.group(1) if m else None


def _file_ts(path: str) -> str:
    m = re.search(r"_(\d{8})_\d{6}\.csv$", path)
    return m.group(1) if m else ""


def _raw_files_newest_first(since: str | None = None) -> list[str]:
    """Raw scrape CSVs sorted newest first; optionally only those on/after `since` (YYYYMMDD)."""
    files = glob.glob(RAW_GLOB)
    if since:
        files = [f for f in files if _file_ts(f) >= since]
    return sorted(files, key=_file_ts, reverse=True)


def _priority_ids() -> list[str]:
    """item_ids flagged as deal/suspect by the valuation step, if it has run."""
    ids: list[str] = []
    for name in ("deals.csv", "valuation.csv"):
        path = os.path.join("data/processed", name)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if name == "valuation.csv" and not any(
                    row.get(col) in ("True", "true", "1") for col in ("is_suspect", "is_outlier")
                ):
                    continue
                iid = _item_id(row.get("url", ""))
                if iid:
                    ids.append(iid)
    return ids


def _pending(limit: int, deals_first: bool = False,
             since: str | None = None) -> tuple[list[tuple[str, str]], int, int]:
    """(batch, total_pending, total_known) — unique listings not yet fetched.

    Newest scrape files are read first so fresh listings (more likely still live,
    and the ones we actually want to act on) are fetched before stale ones. With
    deals_first, listings the valuation step flagged are moved to the front.
    `since` (YYYYMMDD) restricts to listings seen in scrapes on/after that date -
    older listings are mostly sold and come back empty.
    """
    known: dict[str, str] = {}
    for path in _raw_files_newest_first(since):
        with open(path, "r", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                iid = _item_id(row.get("url", ""))
                if iid:
                    known.setdefault(iid, row["url"])

    # "done" = we have text, or we confirmed there is none ("empty"). Transient
    # failures (error:*, "slow" with no text) stay in the queue for a retry.
    done: set[str] = set()
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, "r", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("raw_description", "").strip() or row.get("status") == "empty":
                    done.add(row["item_id"])

    todo = [(iid, url) for iid, url in known.items() if iid not in done]

    if deals_first:
        priority = set(_priority_ids()) & {iid for iid, _ in todo}
        todo.sort(key=lambda t: t[0] not in priority)

    return todo[:limit], len(todo), len(known)


def _parse_block(block: str) -> dict:
    out = {"about": "", "description": "", "listed_age": ""}
    if not block:
        return out

    m = re.search(r"Listed .*?ago", block)
    if m:
        out["listed_age"] = m.group(0)

    if "About this vehicle" in block:
        out["about"] = block.split("About this vehicle", 1)[1].split("Seller's description")[0].strip()

    if "Seller's description" in block:
        t = block.split("Seller's description", 1)[1]
        for stop in ("Location is approximate", "\nToday's picks", "\nSee less",
                     "\nSponsored", "\nRelated searches", "\nSeller information",
                     "\nSimilar listings"):
            i = t.find(stop)
            if i > 0:
                t = t[:i]
        t = re.split(r"\nMessage\nSave\nShare", t)[0]
        # trailing "City, PROV ·" left over from the location line, and See more/less toggles
        t = re.sub(r"\n[^\n]{0,40},\s*[A-Z]{2}\s*·?\s*$", "", t)
        t = re.sub(r"\s*\bSee (?:more|less)\s*$", "", t)
        out["description"] = t.strip(" \n·")

    return out


def _parse_about(about: str) -> dict:
    km = re.search(r"Driven ([\d,]+)\s*km", about or "")
    owners = re.search(r"(\d+)\s+owners?", about or "")
    if "Automatic transmission" in (about or ""):
        transmission = "automatic"
    elif "Manual transmission" in (about or ""):
        transmission = "manual"
    else:
        transmission = ""
    return {
        "odometer_km": int(km.group(1).replace(",", "")) if km else "",
        "transmission": transmission,
        "owners": int(owners.group(1)) if owners else "",
    }


async def _fetch_one(ctx, iid: str, url: str) -> dict:
    rec = {k: "" for k in FIELDS}
    rec.update(item_id=iid, url=url, fetched_at=datetime.now().isoformat(), status="ok")

    page = await ctx.new_page()
    try:
        # FB detail pages are heavy and often never fire "load"; a goto timeout is
        # not fatal - the description is usually in the DOM by then anyway.
        try:
            await page.goto(url, timeout=45000, wait_until="domcontentloaded")
        except PlaywrightTimeout:
            rec["status"] = "slow"

        await page.keyboard.press("Escape")
        try:
            await page.wait_for_selector("text=Seller's description", timeout=10000)
        except Exception:
            pass
        await asyncio.sleep(1.5)

        parsed = _parse_block(await page.evaluate(_BIG_BLOCK_JS))
        description = parsed["description"]
        if not description:
            try:
                og = await page.locator("meta[property='og:description']").first.get_attribute("content")
            except Exception:
                og = ""
            description = (og or "").strip()
            rec["status"] = "og_fallback" if description else "empty"
        elif rec["status"] == "slow":
            rec["status"] = "ok_slow"

        rec["raw_description"] = description.replace("\r", " ").strip()
        rec["listed_age"] = parsed["listed_age"]
        rec.update(_parse_about(parsed["about"]))
    except Exception as e:  # noqa: BLE001 - record and move on, one bad page shouldn't abort the batch
        rec["status"] = f"error:{type(e).__name__}"
    finally:
        await page.close()
    return rec


async def _run(limit: int, deals_first: bool = False, since: str | None = None,
               concurrency: int = 1) -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    batch, pending, known = _pending(limit, deals_first=deals_first, since=since)
    scope = f" (scrapes >= {since})" if since else ""
    print(f"[*] {known} listings known{scope}, {pending} without a description; "
          f"fetching {len(batch)} now (concurrency={concurrency}).")
    if not batch:
        print("[+] Nothing to do.")
        return

    new_file = not os.path.exists(OUT_PATH)
    done = 0
    sem = asyncio.Semaphore(max(1, concurrency))
    write_lock = asyncio.Lock()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(user_agent=_UA, viewport={"width": 1400, "height": 900})

        with open(OUT_PATH, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            if new_file:
                writer.writeheader()
                fh.flush()

            async def worker(iid: str, url: str) -> None:
                nonlocal done
                async with sem:
                    rec = await _fetch_one(ctx, iid, url)
                    # jitter stays *inside* the semaphore so it also paces throughput
                    await asyncio.sleep(2.0 + random.random() * 1.5)
                async with write_lock:
                    writer.writerow(rec)
                    fh.flush()
                    done += 1
                    print(f"  [{done}/{len(batch)}] {iid}  {rec['status']}  "
                          f"desc={len(rec['raw_description'])}c  km={rec['odometer_km'] or '-'}")

            await asyncio.gather(*(worker(iid, url) for iid, url in batch))

        await browser.close()

    _, still_pending, _ = _pending(0, since=since)
    print(f"[+] Done. {still_pending} listings still pending{scope} — re-run to continue.")


def main() -> None:
    from src.ml.config import load_config

    args = sys.argv[1:]
    deals_first = "--deals" in args
    since = next((a.split("=", 1)[1] for a in args if a.startswith("--since=")), None)
    conc = next((int(a.split("=", 1)[1]) for a in args if a.startswith("--concurrency=")), None)
    args = [a for a in args if a != "--deals" and not a.startswith(("--since=", "--concurrency="))]

    cfg = load_config()["scraping"]
    limit = int(args[0]) if args else int(cfg.get("description_batch_size", 150))
    concurrency = conc if conc is not None else int(cfg.get("concurrency_limit", 1))
    asyncio.run(_run(limit, deals_first=deals_first, since=since, concurrency=concurrency))


if __name__ == "__main__":
    main()
