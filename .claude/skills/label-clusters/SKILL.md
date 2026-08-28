---
name: label-clusters
description: >
  Phase 1 "LLM taxonomy" step for auto_sniper_ml, done by the agent directly with
  no external API. Invoke after `python -m src.ml.run_pipeline` has written
  data/clusters/label_requests.json. Reads each cluster's sample listing titles,
  classifies the cluster into a canonical "Make Model" label, and writes
  data/clusters/label_map.json for the next pipeline run. Use when the user asks
  to label clusters, run entity resolution labeling, or "do the LLM step".
---

# Label clusters

You are the taxonomist. `src/ml/cluster.py` has grouped messy marketplace car
titles into clusters; your job is to name each cluster with the one canonical
vehicle it represents.

## Inputs

`data/clusters/label_requests.json`:

```json
{
  "instructions": "...",
  "clusters": [
    {
      "cluster": 42,
      "n_listings": 31,
      "sample_titles": ["2014 Honda civic", "2013 honda civic ex", "..."],
      "heuristic_guess": "Honda Civic"
    }
  ]
}
```

`sample_titles` are the listings nearest the cluster centroid (most
representative). `heuristic_guess` is a token-frequency fallback — a weak hint,
frequently wrong on spelling, word order, or model number. Do not trust it.

## Procedure

1. Read `data/clusters/label_requests.json`.
2. For **every** cluster, decide the canonical vehicle from `sample_titles`:
   - Output `"Make Model"` in title case: `Honda Civic`, `Toyota Corolla`,
     `Mazda 3`, `Ford F-150`, `Volkswagen Golf`, `Mercedes-Benz C-Class`.
   - **No year** — the pipeline appends each listing's own parsed year.
   - **No trim** (LX, Sport, AWD, …) — entity resolution is model-level.
   - Normalize makes/models: `mazda3`→`Mazda 3`, `vw`→`Volkswagen`,
     `chev`/`chevy`→`Chevrolet`, `benz`/`mb`→`Mercedes-Benz`,
     `gm` truck→actual make, `f150`/`f-150`→`F-150`, `3 series`→`3 Series`.
   - Mixed cluster with a clear plurality → label it that plurality vehicle.
   - Genuinely incoherent cluster (no plurality) → `"UNKNOWN"`.
   - Keep spelling identical across clusters so `entity_label` groups correctly
     (e.g. always `Chevrolet Cruze`, never also `Chev Cruze`).
3. Write `data/clusters/label_map.json`:

   ```json
   { "0": "Ford Focus", "1": "Ford Focus", "2": "Honda Civic", ... }
   ```

   One entry for every cluster id in the request (also acceptable:
   `{"labels": { ... }}`).
4. Re-run `python -m src.ml.run_pipeline` (or tell the user to) so the curated
   labels are folded into `data/processed/listings_labeled.*`. On that run
   `cluster_labels.csv` should show `label_method=curated` for the clusters you
   labeled.

## Notes

- There are typically ~250 clusters. Work through all of them; read in batches if
  needed but the output must be complete.
- This is not a production classifier — it is a one-off curation pass the agent
  redoes whenever clustering is re-run with materially different data.
- If `label_requests.json` is missing, run `python -m src.ml.run_pipeline` first.
