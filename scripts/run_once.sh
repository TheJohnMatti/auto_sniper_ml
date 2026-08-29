#!/usr/bin/env bash
# One full pass of the pipeline: scrape -> entity resolution -> descriptions ->
# sentiment -> valuation -> notify. Used by .github/workflows/pipeline.yml and
# runnable locally (`bash scripts/run_once.sh`).
#
# Env:
#   PYTHON              interpreter to use (default: python)
#   SKIP_SCRAPE=1       reuse existing data/raw, don't hit Facebook
#   SKIP_DESCRIPTIONS=1 skip the detail-page description scrape
#   DESC_SINCE          YYYYMMDD floor for fetch_descriptions (default: 4 days ago)
#   NTFY_TOPIC          required for notify to actually send (see src/ml/notify.py)
set -euo pipefail

PY="${PYTHON:-python}"
cd "$(dirname "$0")/.."

log() { printf '\n=== %s ===\n' "$1"; }

if [ "${SKIP_SCRAPE:-0}" != "1" ]; then
  log "scrape"
  "$PY" -m src.scraper.run || echo "[!] scrape failed - continuing on cached data"
fi

log "entity resolution"
"$PY" -m src.ml.run_pipeline

if [ "${SKIP_DESCRIPTIONS:-0}" != "1" ]; then
  log "descriptions"
  since="${DESC_SINCE:-$(date -u -d '4 days ago' +%Y%m%d 2>/dev/null || date -u -v-4d +%Y%m%d)}"
  "$PY" -m src.scraper.fetch_descriptions --deals "--since=${since}" || \
    echo "[!] description scrape failed - continuing"
fi

log "sentiment"
"$PY" -m src.ml.sentiment

log "valuation"
"$PY" -m src.ml.valuation

log "notify"
"$PY" -m src.ml.notify
