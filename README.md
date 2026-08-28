# 🎯 Auto-Sniper ML

An automated marketplace arbitrage engine that scrapes, structures, and prices used vehicles in real-time to identify market inefficiencies.

## 🏗️ Architecture

This project abandons the anti-pattern of passing raw, chaotic marketplace data directly into LLMs or pricing models. Instead, it utilizes a decoupled, high-performance ML pipeline:

### Phase 1: Entity Resolution (Unsupervised + LLM Synthesis)

1. **Scraping Engine:** Utilizes `Playwright` to navigate the JavaScript-heavy DOMs of Facebook Marketplace and Kijiji, extracting raw vehicle listing strings.

2. **Vectorization:** Converts messy, user-generated titles (e.g., `"2014 hnda civc manual"`) into dense semantic embeddings using `sentence-transformers`.

3. **Bisecting K-Means Clustering:** Employs an iterative local bisection strategy to organically group similar listings without hardcoding the exact number of unique vehicles ($K$).

4. **LLM Taxonomy:** Samples the centroids of the resulting low-variance clusters and passes them to a generative LLM to extract a clean, canonical label (e.g., `2014 Honda Civic`).

### Phase 2: Valuation Regression (Supervised)

*(In Development)*
Once listings are mapped to discrete entity IDs, the system applies statistical pricing models (analyzing standard deviations, median historical prices, and mileage depreciation curves) to flag severely underpriced assets (Z-score anomaly detection) and fire push notifications.

## 🚀 Getting Started

1. **Install Dependencies:**
   Ensure you have [Poetry](https://python-poetry.org/) installed.

   ```bash
   poetry install
   poetry run playwright install chromium
   ```

2. **Environment Variables:**
   Copy the example environment file and add your API keys.

   ```bash
   cp .env.example .env
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

   Consolidates every raw CSV, deduplicates by listing id, embeds titles,
   bisecting-k-means clusters them, and labels each cluster (LLM if a key is set
   in `.env`, otherwise a deterministic heuristic). Outputs:

   - `data/processed/listings_labeled.{csv,pkl}` — one row per unique listing
     with `cluster`, `canonical_label`, `entity_id`, `entity_label`
   - `data/clusters/cluster_labels.csv` — one row per model cluster

   Individual steps are runnable on their own for debugging, e.g.
   `python -m src.ml.load_data` or `python -m src.ml.cluster`.

> **Note:** if `poetry` is not on your PATH, call the project virtualenv's
> interpreter directly (`poetry env info -p` prints its location).