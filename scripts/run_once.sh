#!/usr/bin/env bash
# One pass of the pipeline: scrape -> observation log -> entity resolution ->
# descriptions -> sentiment -> valuation -> notify. Used by the GitHub workflows
# and runnable locally (`bash scripts/run_once.sh`).
#
# Entity resolution is INCREMENTAL by default (assign to frozen clusters - the
# hourly job). The weekly retrain sets FULL_RETRAIN=1 to re-cluster from scratch
# over the whole accumulated corpus and refresh the model snapshot.
#
# Env:
#   PYTHON              interpreter to use (default: python)
#   SKIP_SCRAPE=1       reuse existing data/raw, don't hit Facebook
#   SKIP_DESCRIPTIONS=1 skip the detail-page description scrape
#   FULL_RETRAIN=1      re-cluster instead of incremental assign (weekly job)
#   DESC_SINCE          YYYYMMDD floor for fetch_descriptions (default: 4 days ago)
#   NTFY_TOPIC          required for notify to actually send (see src/ml/notify.py)
set -euo pipefail

PY="${PYTHON:-python}"
cd "$(dirname "$0")/.."

# Single-flight: a scheduled 5-10 min interval can outpace a slow description
# scrape, so never let two runs overlap (they share the cache + state DBs).
LOCK="${TMPDIR:-/tmp}/auto_sniper_run_once.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "[i] another run_once.sh is already running ($LOCK) - exiting"
  exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

log() { printf '\n=== %s ===\n' "$1"; }

if [ "${SKIP_SCRAPE:-0}" != "1" ]; then
  log "scrape"
  "$PY" -m src.scraper.run || echo "[!] scrape failed - continuing on cached data"
fi

if ! ls data/raw/facebook_*_raw_*.csv >/dev/null 2>&1; then
  echo "[!] no scraped data in data/raw/ - nothing to process."
  echo "    Facebook serves a login wall to datacenter IPs; the scrape must run"
  echo "    from a residential IP (your machine / a self-hosted runner). See README."
  exit 1
fi

log "observation log"
"$PY" -m src.ml.observe || echo "[!] observe failed - continuing"

if [ "${FULL_RETRAIN:-0}" = "1" ]; then
  log "entity resolution (full re-cluster)"
  "$PY" -m src.ml.run_pipeline
else
  log "entity resolution (incremental)"
  "$PY" -m src.ml.run_pipeline --incremental
fi

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
