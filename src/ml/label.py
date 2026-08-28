"""
Phase 1 - Step 3: turn each cluster into a clean canonical vehicle label.

We sample the listings nearest each cluster centroid and ask an LLM to collapse
them into a single "Year Make Model Trim" string. If no API key is configured the
pipeline still runs - it falls back to a deterministic heuristic (modal year +
first two title tokens) so downstream valuation work is never blocked.
"""
import os
import re
from collections import Counter

import numpy as np
import pandas as pd

from src.ml.cluster import nearest_to_centroid
from src.ml.config import ml_config

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

_STOPWORDS = {
    "the", "a", "for", "sale", "clean", "low", "kms", "km", "miles", "obo",
    "auto", "automatic", "manual", "loaded", "financing", "available", "no",
    "accidents", "certified", "safety", "safetied", "mint", "condition",
    "awd", "fwd", "rwd", "4wd", "quattro", "sedan", "coupe", "hatchback",
    "wagon", "turbo", "sport", "premium", "limited", "touring",
}


def _heuristic_label(titles: list[str], years: list) -> str:
    """Model-level label (no year); the pipeline appends each listing's own year."""
    tokens: list[str] = []
    for title in titles:
        # letter-initial tokens, keeping trailing digits/hyphens so "mazda3",
        # "cx-5", "f-150", "3-series" survive as single model tokens
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9-]*", title.lower()):
            if tok not in _STOPWORDS and len(tok) > 1:
                tokens.append(tok)
    common = [tok for tok, _ in Counter(tokens).most_common(3)]
    # collapse containment pairs to the more specific token ("mazda" + "mazda3" -> "mazda3")
    picked: list[str] = []
    for tok in common:
        replaced = False
        for i, p in enumerate(picked):
            if p in tok or tok in p:
                picked[i] = tok if len(tok) > len(p) else p
                replaced = True
                break
        if not replaced:
            picked.append(tok)
        if len(picked) == 2:
            break
    return " ".join(t.capitalize() for t in picked) or "UNLABELED"


def _llm_label(sample_titles: list[str], cfg: dict) -> str | None:
    provider = cfg.get("provider", "openai").lower()
    n = cfg.get("samples_per_cluster", 5)
    prompt = cfg["prompt_template"].format(samples_per_cluster=n).strip()
    prompt += "\n\nTitles:\n" + "\n".join(f"- {t}" for t in sample_titles)
    prompt += "\n\nRespond with only the canonical label."

    try:
        if provider == "openai":
            if not os.environ.get("OPENAI_API_KEY"):
                return None
            from openai import OpenAI

            client = OpenAI()
            resp = client.chat.completions.create(
                model=cfg.get("model", "gpt-4-turbo"),
                temperature=float(cfg.get("temperature", 0.1)),
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content.strip()

        if provider in ("gemini", "google"):
            if not os.environ.get("GOOGLE_API_KEY"):
                return None
            import google.generativeai as genai

            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
            model = genai.GenerativeModel(cfg.get("model", "gemini-1.5-flash"))
            return model.generate_content(prompt).text.strip()
    except Exception as e:
        print(f"[!] LLM labeling failed ({provider}): {e}")
        return None

    print(f"[!] Unknown labeling provider: {provider}")
    return None


def label_clusters(df: pd.DataFrame, embeddings: np.ndarray, labels: np.ndarray,
                   centroids: np.ndarray) -> pd.DataFrame:
    cfg = ml_config()["llm_labeling"]
    n_samples = int(cfg.get("samples_per_cluster", 5))

    rows = []
    llm_used = 0
    for cid in sorted(set(labels)):
        idx = nearest_to_centroid(embeddings, labels, centroids, cid, n_samples)
        sample_titles = df.iloc[idx]["raw_title"].tolist()

        label = _llm_label(sample_titles, cfg)
        method = "llm"
        if not label:
            label = _heuristic_label(
                df.loc[df["cluster"] == cid, "raw_title"].tolist(),
                df.loc[df["cluster"] == cid, "year"].tolist(),
            )
            method = "heuristic"
        else:
            llm_used += 1

        rows.append({
            "cluster": cid,
            "canonical_label": label,
            "n_listings": int((labels == cid).sum()),
            "label_method": method,
            "sample_titles": " | ".join(sample_titles),
        })

    print(f"[+] Labeled {len(rows)} clusters ({llm_used} via LLM, {len(rows) - llm_used} heuristic)")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from src.ml.load_data import load_raw_listings
    from src.ml.embed import embed_titles
    from src.ml.cluster import cluster_embeddings

    df = load_raw_listings()
    X = embed_titles(df["title_clean"].tolist())
    res = cluster_embeddings(X)
    df = df.assign(cluster=res.labels)
    out = label_clusters(df, X, res.labels, res.centroids)
    print(out.head(20).to_string())
