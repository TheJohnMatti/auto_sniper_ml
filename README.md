# 🎯 Auto-Sniper ML

An automated marketplace arbitrage engine that scrapes, structures, and prices used vehicles in real-time to identify market inefficiencies.

## 🏗️ Architecture

This project abandons the anti-pattern of passing raw, chaotic marketplace data directly into LLMs or pricing models. Instead, it utilizes a decoupled, high-performance ML pipeline:

### Phase 1: Entity Resolution (Unsupervised + Agent Synthesis)

1. **Scraping Engine:** Uses `Playwright` to navigate the JavaScript-heavy DOM of Facebook Marketplace, extracting raw vehicle listing strings. Targets are `scraping.locations` in `config.yaml` — either a Facebook city slug or, for a point Facebook has no named route for (London, ON — `/marketplace/london/` is London **UK**), a `latitude`/`longitude`/`radius_km` on `/marketplace/category/vehicles`. Always sorted newest-first.

2. **Vectorization:** Converts messy, user-generated titles (e.g., `"2014 hnda civc manual"`) into dense semantic embeddings using `sentence-transformers`.

3. **Bisecting K-Means Clustering:** Employs an iterative local bisection strategy to organically group similar listings without hardcoding the exact number of unique vehicles ($K$).

4. **Cluster Taxonomy (agent-in-the-loop):** The pipeline writes each cluster's centroid samples to `data/clusters/label_requests.json`. There is **no external LLM API** — the classification is a one-off curation pass done by the coding agent via the `label-clusters` skill, which writes canonical `Make Model` labels to `data/clusters/label_map.json`. Clusters left unlabeled fall back to a deterministic token-frequency heuristic. The pipeline then combines the model label with each listing's own parsed year into an `entity_label` / `entity_id` (e.g. `2014 Honda Civic`).

### Phase 2: Valuation & Anomaly Detection

Once listings are mapped to entity IDs, `src/ml/valuation.py` builds a **robust
price distribution per year+model entity** (median + MAD, not mean/std — market
prices are heavy-tailed and full of scams/parts/typos). Each listing gets a
leave-one-out robust z-score against its entity's other comps. Listings are
flagged as **deals** when they clear the z-score, a minimum discount %, and are
not so cheap they're almost certainly junk (**suspect**).

**Mileage adjustment (`src/ml/mileage.py`):** a 300 000 km car and a 90 000 km
car of the same year+model aren't comparable, so the high-km one looks like a
steal. The module fits one pooled, robust (median-of-pairwise-slopes)
depreciation curve — `log(price) ≈ entity_effect + β·km`, currently ≈ **−3×10⁻⁶
log$/km (~27 % per 100 000 km)** — and restates every price to a 120 000 km
reference before scoring. Cars with no / implausible odometer are held at their
entity's median odometer (i.e. barely moved).

**Outlier filter ("too good to be true"):** free/$1 BMWs whose description is
really "make me an offer", finance-payment ads (`$281 bi-weekly`), lease
buy-ins and rental deposits all read as enormous discounts. `is_outlier` fires
when the price field clearly isn't the sale price, when it's a keyboard-mash
placeholder (`$1234` shows up ~46×), or when the discount is too deep to be real
*and* nothing in the description (salvage, blown engine, …) explains it.

**Staleness filter:** a genuinely underpriced car sells in days. `is_stale`
drops any deal that's been listed longer than `max_listing_age_days` (45), or a
big-discount "steal" still up after two weeks — age comes from the scraped
listing page (*"Listed 3 weeks ago"*) plus time elapsed since that scrape, so it
stays honest on re-runs. `deal_score` also carries a mild recency bonus so
fresh listings rank first.

All three flags stay in `valuation.csv` — the leave-one-out baseline already
ignores the junk and the description scrape learns its tells — but are dropped
from `deals.csv` and notifications.

### Phase 2b: Description Signals

The category feed only exposes title + price. `src/scraper/fetch_descriptions.py`
visits each listing page (resumable, in batches — it's slow and bot-sensitive) to
pull the **full seller description** plus the structured *About this vehicle*
block (odometer, transmission, owners). `src/ml/sentiment.py` then runs a
transparent lexicon + rule model over the description — **no API, no model
download** — producing a `condition_score` (assurances like *no rust / new
brakes / one owner* vs. red flags like *as-is / head gasket / needs work*), an
`urgency_score`, and a `is_dealer_or_ad` flag. `valuation.py` folds these into a
combined `deal_score` and drops dealer/ad posts from the deal list.

### Phase 3: Notifications

`src/ml/notify.py` pushes each fresh `deals.csv` row above a `deal_score`
threshold to your phone via [ntfy](https://ntfy.sh) (no account / API key — a
notification is just an HTTP POST to a topic). High-`deal_score` deals get an
elevated (louder) priority.

**Exactly-once, cloud-ready:** state is a SQLite DB (`$NOTIFY_DB`, default
`data/processed/notified.sqlite3`). Each listing goes through a
claim-before-send handshake against the DB's primary key, so a row once marked
`sent_at` is **never** sent again — even with overlapping cron runs or a crash
mid-send (an abandoned claim is reclaimed after an hour; a failed POST is
released for retry). Point `$NOTIFY_DB` at a mounted volume in a container.

## 🚀 Getting Started

1. **Install Dependencies:**
   Ensure you have [Poetry](https://python-poetry.org/) installed.

   ```bash
   poetry install
   poetry run playwright install chromium
   ```

3. **Run the Scraping Pipeline:**

   ```bash
   poetry run python src/scraper/run.py
   ```

   Writes one `data/raw/facebook_<city>_raw_<timestamp>.csv` per city. Columns:
   `scraped_at, platform, location, raw_title, raw_price, raw_price_original,
   raw_listing_location, raw_description, url`.

4. **Run the Phase 1 ML Pipeline (entity resolution):**

   ```bash
   poetry run python -m src.ml.run_pipeline
   ```

   Consolidates every raw CSV, deduplicates by listing id, embeds titles, and
   bisecting-k-means clusters them. Outputs:

   - `data/processed/listings_labeled.{csv,pkl}` — one row per unique listing
     with `cluster`, `canonical_label`, `entity_id`, `entity_label`
   - `data/clusters/cluster_labels.csv` — one row per model cluster
   - `data/clusters/label_requests.json` — centroid samples awaiting labels

   Individual steps are runnable on their own for debugging, e.g.
   `python -m src.ml.load_data` or `python -m src.ml.cluster`.

5. **Label the clusters** (agent step, no API): invoke the `label-clusters`
   skill. It reads `label_requests.json`, writes `data/clusters/label_map.json`,
   and you re-run step 4 to fold the curated labels in. Skip this and the
   pipeline still runs on heuristic labels.

6. **Run Phase 2 valuation:**

   ```bash
   poetry run python -m src.ml.valuation
   ```

   Reads `listings_labeled.pkl` and writes:

   - `data/processed/valuation.csv` — every scored listing + entity median /
     mileage-adjusted price / robust z-score / discount % / `is_deal` /
     `is_suspect` / `is_outlier`
   - `data/processed/deals.csv` — the flagged underpriced listings (outliers and
     suspects removed), best first

   Thresholds live under `ml_pipeline.valuation` in `config.yaml`. Odometer comes
   from Phase 2b, so run this again after scraping descriptions.

7. **Scrape descriptions & score them** (Phase 2b, optional but recommended):

   ```bash
   poetry run python -m src.scraper.fetch_descriptions --deals            # candidates first
   pwsh scripts/backfill_descriptions.ps1 -Since <YYYYMMDD> -RunSentiment  # backfill the rest
   ```

   `fetch_descriptions` is resumable — it appends to `data/raw/descriptions.csv`,
   skips listings it already has, and fetches `--concurrency` pages at once
   (default `scraping.concurrency_limit`; keep it 2–3, unauthenticated). `--since`
   limits it to recent scrapes (older listings are mostly sold). The backfill
   script loops it in spaced batches until the queue drains, then runs
   `sentiment` (→ `data/processed/listing_signals.csv`) and re-runs `valuation`
   to fold the signals into `deal_score`.

8. **Get pushed the deals** (Phase 3):

   ```bash
   poetry run python -m src.ml.notify --dry-run   # preview
   poetry run python -m src.ml.notify             # push new ones to your phone
   ```

   Sends each fresh `deals.csv` row with `deal_score ≥ min_deal_score` as an
   [ntfy](https://ntfy.sh) notification — no account, no API key. Because this
   repo is public, the topic lives in a local `.env` (gitignored), not
   `config.yaml`:

   ```bash
   echo "NTFY_TOPIC=$(python -c 'import secrets;print("auto-sniper-"+secrets.token_urlsafe(18))')" >> .env
   ```

   Install the ntfy app (iOS / Android / web), add a subscription to that exact
   topic name on server `ntfy.sh`, and you're done. `NTFY_SERVER` / `NTFY_TOKEN`
   env vars cover self-hosted or protected topics. A SQLite state DB
   (`$NOTIFY_DB`, default `data/processed/notified.sqlite3`) guarantees no
   listing is ever pushed twice — even under overlapping cron runs; on a
   container, put it on a persistent volume. `--min-score` / `--limit` override
   the config; `--dry-run` previews without sending or recording.

> **Note:** if `poetry` is not on your PATH, call the project virtualenv's
> interpreter directly (`poetry env info -p` prints its location).

## ☁️ Deployment (scheduled GitHub Action)

`.github/workflows/pipeline.yml` runs the whole chain
(`scripts/run_once.sh`: scrape → entity resolution → descriptions → sentiment →
valuation → notify) **hourly** on a GitHub-hosted runner. The repo is public so
Actions minutes are free. The scrape is one southwestern-Ontario pass
(`scraping.locations`), so a run is only a few minutes.

**Why not Vercel / a serverless cron:** the pipeline needs `torch` +
`sentence-transformers` (~1 GB, over Vercel's 250 MB function limit), a real
Chromium for Playwright, and a persistent disk for the notify state DB — none of
which serverless provides. A GitHub Action gets a full VM and is itself a cron.

**Cadence:** GitHub's scheduler is best-effort (a `* * * *` job often fires
5–15 min late, and can be skipped under load). For true ~5-minute *sniper*
latency, run just the lightweight scanner (feed scrape → match against the last
full model → notify, no `torch`) as an always-on loop on a small always-free VM
(Oracle Cloud / Fly.io), and keep this Action as the weekly heavy rebuild. That
split isn't built yet — the hourly Action is the current whole-pipeline
deployment.

**Secrets** are GitHub Actions repo secrets (encrypted, never in code / logs /
PRs). `NTFY_TOPIC` is already set; add `NTFY_TOKEN` too if you move to a
protected or self-hosted ntfy topic:

```bash
gh secret set NTFY_TOPIC        # your long random topic (also in local .env)
gh secret set NTFY_TOKEN        # optional
```

Then enable Actions for the repo. The schedule starts on its own; use **Run
workflow** (`workflow_dispatch`) for a manual run — it has `skip_scrape` /
`skip_descriptions` toggles.

**State between runs** lives in the `pipeline-data-*` Actions cache (`data/` —
raw scrapes, `descriptions.csv`, the embedding cache, and `notified.sqlite3`).
The cron never idles long enough for the 7-day cache eviction to bite; if it
does, the next run rebuilds from scratch and may re-notify current deals once.
`deals.csv` / `valuation.csv` are also uploaded as a run artifact (14 days).

**Caveats:**

- **Region gate:** production only surfaces deals whose town is within
  `ml_pipeline.inference.region.radius_km` of London, ON (`src/ml/geo.py`).
  Comps/baselines still use everything scraped. Widen `radius_km` (or the scrape
  radius) to include Windsor-Essex.
- **Facebook may block the runner's datacenter IP** for unauthenticated
  Marketplace requests. If scrapes come back empty, run the scraper from a
  residential IP (locally or a small VPS) and let the Action do everything with
  `skip_scrape`, or add a proxy.
- **Coordinate scrape is unverified against live Facebook** from this
  environment: `/marketplace/category/vehicles?latitude=…&longitude=…&radius=…`
  is confirmed to resolve to "London, Ontario", but the card selector
  (`[data-virtualized='false']`) and scroll behaviour on that route need a real
  smoke-test run. `_extract_listings` fails soft (logs, returns nothing) if the
  selector has rotted.
- **Label drift:** clustering is re-run each time, so as the listing population
  churns, cluster ids shift and the curated `data/clusters/label_map.json`
  gradually stops matching (clusters fall back to the heuristic label). Re-run
  the `label-clusters` skill and commit a fresh `label_map.json` every few weeks.