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
not so cheap they're almost certainly junk (**suspect**). Mileage-based
depreciation and push notifications are still TODO — mileage needs the
description scrape first.

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
     robust z-score / discount %
   - `data/processed/deals.csv` — just the flagged underpriced listings, best first

   Thresholds live under `ml_pipeline.valuation` in `config.yaml`.

> **Note:** if `poetry` is not on your PATH, call the project virtualenv's
> interpreter directly (`poetry env info -p` prints its location).