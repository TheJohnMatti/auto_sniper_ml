# 🎯 Auto-Sniper ML

An automated marketplace arbitrage engine that scrapes, structures, and prices used vehicles in real-time to identify market inefficiencies.

## 🏗️ Architecture

This project abandons the anti-pattern of passing raw, chaotic marketplace data directly into LLMs or pricing models. Instead, it utilizes a decoupled, high-performance ML pipeline:

### Phase 1: Entity Resolution (Unsupervised + Agent Synthesis)

1. **Scraping Engine:** Utilizes `Playwright` to navigate the JavaScript-heavy DOMs of Facebook Marketplace and Kijiji, extracting raw vehicle listing strings.

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
*and* nothing in the description (salvage, blown engine, …) explains it. Outliers
stay in `valuation.csv` — the leave-one-out baseline already ignores them and the
description scrape learns their tells — but are dropped from `deals.csv` and
notifications.

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
notification is just an HTTP POST to a topic). A local `notified.json` state file
dedupes across runs so it's cron-safe. High-`deal_score` deals get an elevated
(louder) priority.

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
   env vars cover self-hosted or protected topics. State in
   `data/processed/notified.json` means cron / re-runs never double-send;
   `--all` ignores it, `--min-score` / `--limit` override the config.

> **Note:** if `poetry` is not on your PATH, call the project virtualenv's
> interpreter directly (`poetry env info -p` prints its location).