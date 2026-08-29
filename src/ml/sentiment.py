"""
Turn scraped seller descriptions into structured condition / urgency signals.

Lexicon + rule based on purpose: for used-car arbitrage the useful signal is not
literary sentiment but **condition risk** ("as-is", "head gasket", "needs work"),
**condition assurance** ("no rust", "new brakes", "one owner") and **seller
urgency** ("must sell", "moving", "obo"). A transparent keyword model is easy to
audit and tune; no model download, no API.

    python -m src.ml.sentiment

Input:  data/raw/descriptions.csv   (from src.scraper.fetch_descriptions)
Output: data/processed/listing_signals.csv   (one row per listing with a description)
"""
import os
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

DESCRIPTIONS_PATH = "data/raw/descriptions.csv"
SIGNALS_PATH = "data/processed/listing_signals.csv"

# Facebook renders "Listed 3 weeks ago" / "Listed a day ago" / "Just listed".
_AGE_RE = re.compile(
    r"(?:listed\s+)?(?:(\d+)|(a|an))\s*(hour|day|week|month|year)s?\s*ago", re.I
)
_AGE_UNIT_DAYS = {"hour": 1 / 24, "day": 1.0, "week": 7.0, "month": 30.0, "year": 365.0}


def parse_listed_age(text: str) -> float | None:
    """'Listed 3 weeks ago' -> 21.0 days. 'Just listed' -> 0.0. Unknown -> None."""
    if not isinstance(text, str) or not text.strip():
        return None
    if re.search(r"just listed|listed today|moments ago", text, re.I):
        return 0.0
    m = _AGE_RE.search(text)
    if not m:
        return None
    qty = 1.0 if m.group(2) else float(m.group(1))
    return round(qty * _AGE_UNIT_DAYS[m.group(3).lower()], 2)

# --- lexicons -----------------------------------------------------------------
# Each entry: label -> regex (matched case-insensitively against the description).

RED_FLAGS = {
    "as_is": r"\bas[\s-]?is\b|\bsold as is\b|\bas is where is\b",
    "salvage_rebuilt": r"\bsalvage\b|\brebuilt(?:\s+title)?\b|\bbranded\b|\bwrite[\s-]?off\b|\bwrote off\b|\breconstructed\b",
    "flood_fire": r"\bflood(?:ed)?\b|\bfire damage\b|\bwater damage\b",
    "needs_work": r"\bneeds?\s+(?:work|tlc|repair|fixing|attention|love)\b|\bhandyman special\b|\bmechanic'?s?\s+special\b|\bproject car\b|\bfixer[\s-]?upper\b",
    "parts_only": r"\bparts?\s+(?:car|only)\b|\bparting out\b|\bfor parts\b|\bpart out\b",
    "wont_run": r"\b(?:won'?t|will not|doesn'?t|does not|no)\s+(?:start|run|crank)\b|\bnot running\b|\bnon[\s-]?runner\b|\bdead\b",
    "engine_trans": r"\bcheck engine\b|\bengine light\b|\bcel\b|\bhead gasket\b|\bblown (?:engine|motor|head)\b|\bknock(?:ing)?\b|\btransmission (?:slip|issue|problem|going|gone)\b|\btranny (?:slip|issue|problem)\b|\bmisfire\b|\boil leak\b|\bburns oil\b|\bcoolant leak\b",
    "rust": r"\brust(?:ed|y| through| out| holes| issues| spots)?\b(?!\s*free)|\bsurface rust\b|\bframe rot\b",
    "accident_damage": r"\baccident\b(?!\s*free)|\bcollision\b|\bframe damage\b|\bstructural damage\b|\bhail damage\b|\bbody damage\b",
    "no_warranty": r"\bno warranty\b|\bno refunds?\b|\bno returns?\b",
}

POSITIVES = {
    "no_rust": r"\b(?:no|zero|0|minimal|little)\s+rust\b|\brust[\s-]?free\b|\bnever rusted\b",
    "well_maintained": r"\bwell[\s-]?maintained\b|\bmaintained\b|\bregularly serviced\b|\bservice (?:records?|history)\b|\balways serviced\b|\bdealer maintained\b",
    "one_owner": r"\b(?:one|1|single)[\s-]?owner\b|\bfirst owner\b",
    "non_smoker": r"\bnon[\s-]?smok(?:er|ing)\b|\bsmoke[\s-]?free\b|\bgarage[\s-]?kept\b|\bgaraged\b",
    "new_parts": r"\b(?:new|brand new|recently (?:replaced|changed|installed))\s+(?:brakes?|rotors?|tires?|tyres?|battery|alternator|starter|clutch|timing (?:belt|chain)|serpentine belt|exhaust|muffler|struts?|shocks?|suspension|windshield)\b",
    "runs_drives_well": r"\b(?:runs?|drives?)\s+(?:great|well|perfect(?:ly)?|smooth(?:ly)?|excellent|strong|like new|amazing|mint)\b|\bno (?:issues|problems)\b|\bmechanically (?:sound|solid|perfect)\b",
    "condition_praise": r"\bmint(?:\s+condition)?\b|\bpristine\b|\bimmaculate\b|\bexcellent condition\b|\bgreat condition\b|\bshowroom\b|\blike new\b",
    "no_accidents": r"\bno accidents?\b|\baccident[\s-]?free\b|\bclean (?:title|carfax|history)\b|\bnever been in an accident\b",
    "certified_safety": r"\bcertified\b|\bsafet(?:y|ied)\b|\bfresh (?:safety|mvi|inspection|cert)\b|\bnew mvi\b|\be[\s-]?tested\b|\bvalid (?:safety|mvi|inspection)\b",
}

URGENCY = {
    "must_sell": r"\bmust (?:sell|go)\b|\bneed(?:s)? (?:it )?(?:to )?(?:go|gone|sold)\b|\bneed(?:s)? sold\b|\bhave to sell\b",
    "life_event": r"\bmoving\b|\brelocat(?:ing|e)\b|\bout[\s-]?of[\s-]?province\b|\bout of country\b|\bleaving (?:the )?(?:country|province)\b|\bdownsizing\b",
    "hurry": r"\bquick sale\b|\basap\b|\btoday only\b|\bfirst come\b|\bgone by\b|\bthis week(?:end)?\b|\bmotivated (?:seller|to sell)\b|\bpriced to sell\b|\bneed gone\b",
    "price_moved": r"\bprice drop\b|\breduced\b|\blowered (?:the )?price\b|\bnegotiable\b|\bobo\b|\bo\.b\.o\b|\bor best offer\b|\bmake (?:me )?an offer\b|\bopen to offers?\b",
}

DEALER_OR_AD = {
    "we_buy": r"\bwe buy (?:cars|any car)\b|\bbuying (?:unwanted|your|any|junk|scrap)\b|\bcash for (?:cars|junk|your)\b|\bget \$?\d+ for your\b|\btop dollar (?:paid|for)\b|\bwanted:? .*\bcars?\b|\bunwanted (?:cars?|vehicles?)\b",
    "wrecker_scrap": r"\bauto wreckers?\b|\bscrap (?:cars?|vehicles?)\b|\bsalvage (?:yard|cars for)\b|\bvehicle disposal\b|\bjunk (?:cars?|removal)\b",
    "dealer_boilerplate": r"\badministration fee\b|\badmin fee\b|\bplus (?:hst|tax|taxes)\b|\bdoc fee\b|\bfinancing available\b|\bwholesale direct\b|\bcall today\b|\bapply (?:online|now)\b|\bo\.?a\.?c\.?\b",
}

# The number in the price field is NOT the sale price - it's a finance payment, a
# lease buy-in, or a rental deposit. These listings are "too cheap" as a data
# artefact, not a deal, so valuation drops them at inference time.
PRICE_NOT_SALE = {
    "finance_payment": r"\bbi[\s-]?weekly\b|\bweekly payment\b|\b(?:payment|pymt)s?\s+(?:of|from|as low as|starting)\b|\bpayments? as low as\b|\bas low as \$?\d+\s*(?:\+ ?tax|bi|/|per|a )|\$\d+\s*(?:\+ ?tax\s*)?(?:bi[\s-]?weekly|/ ?(?:mo|month)|per month|a month|monthly)\b|\b0\$? ?down\b|\b\$0 down\b",
    "lease_takeover": r"\blease (?:transfer|takeover|take[\s-]?over|assumption|buyout)\b|\btake over (?:the |my |this )?lease\b|\bassume (?:the |my )?lease\b|\bmonths? (?:remaining|left) on (?:the |my )?lease\b|\blease [a-z ]{0,12}remaining\b",
    "rental_deposit": r"\bfor rent\b|\bcar rental\b|\brental car\b|\brent[\s-]?to[\s-]?own\b|\bweekly rate\b|\bdaily rate\b|\bdeposit required\b|\brent it\b",
}

_CONTACT_ONLY = re.compile(r"^[\s\d.,+()-]*(?:call|text|phone|for more info(?:rmation)?|contact)?[\s\d.,+()x-]*$", re.I)


def _count_hits(text: str, lexicon: dict[str, str]) -> tuple[int, list[str]]:
    hits = [label for label, pat in lexicon.items() if re.search(pat, text, re.I)]
    return len(hits), hits


def _load_descriptions(path: str = DESCRIPTIONS_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    df["raw_description"] = df["raw_description"].str.strip()
    df["_len"] = df["raw_description"].str.len()
    # a listing may have several rows (retries) - keep the richest
    df = df.sort_values("_len").drop_duplicates("item_id", keep="last")
    return df


def score_descriptions(df: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    now = now or datetime.now(timezone.utc)
    rows = []
    for r in df.itertuples(index=False):
        text = r.raw_description
        has = bool(text) and r.status != "empty"

        # age on the market: what the page said when we scraped it, plus however
        # long ago that scrape was (so re-running months later stays honest).
        age_at_scrape = parse_listed_age(getattr(r, "listed_age", ""))
        listing_age_days = None
        if age_at_scrape is not None:
            fetched = pd.to_datetime(getattr(r, "fetched_at", None), utc=True, errors="coerce")
            since_scrape = 0.0 if pd.isna(fetched) else max(0.0, (now - fetched).total_seconds() / 86400)
            listing_age_days = round(age_at_scrape + since_scrape, 2)

        n_red, red = _count_hits(text, RED_FLAGS)
        n_pos, pos = _count_hits(text, POSITIVES)
        n_urg, urg = _count_hits(text, URGENCY)
        n_ad, ad = _count_hits(text, DEALER_OR_AD)
        n_notsale, notsale = _count_hits(text, PRICE_NOT_SALE)

        # "no rust" / "no accidents" trip both lexicons - the assurance wins.
        for assurance, flag in (("no_rust", "rust"), ("no_accidents", "accident_damage")):
            if assurance in pos and flag in red:
                red.remove(flag)
        n_red = len(red)

        is_ad = bool(n_ad) or (0 < len(text) < 25) or bool(text and _CONTACT_ONLY.match(text))

        # red flags weigh ~2x assurances; squashed to [-1, 1]
        condition_score = float(np.tanh((n_pos - 2 * n_red) / 3.0)) if has else 0.0
        urgency_score = min(1.0, n_urg / 3.0) if has else 0.0

        rows.append({
            "item_id": r.item_id,
            "has_description": has,
            "desc_len": len(text),
            "condition_score": round(condition_score, 3),
            "urgency_score": round(urgency_score, 3),
            "red_flag_count": n_red,
            "positive_count": n_pos,
            "red_flags": ";".join(red),
            "positives": ";".join(pos),
            "urgency_cues": ";".join(urg),
            "is_dealer_or_ad": is_ad,
            "price_not_sale_price": bool(n_notsale),
            "price_not_sale_cues": ";".join(notsale),
            "odometer_km": r.odometer_km,
            "transmission": r.transmission,
            "listed_age": r.listed_age,
            "listing_age_days": listing_age_days,
        })
    return pd.DataFrame(rows)


def main() -> None:
    if not os.path.exists(DESCRIPTIONS_PATH):
        raise FileNotFoundError(
            f"{DESCRIPTIONS_PATH} not found. Run `python -m src.scraper.fetch_descriptions` first."
        )
    os.makedirs(os.path.dirname(SIGNALS_PATH), exist_ok=True)

    src = _load_descriptions()
    signals = score_descriptions(src)
    signals.to_csv(SIGNALS_PATH, index=False)

    scored = signals[signals["has_description"]]
    print(f"[+] Scored {len(scored)} descriptions -> {SIGNALS_PATH}")
    print(f"    dealer/ad or junk : {int(signals['is_dealer_or_ad'].sum())}")
    print(f"    price != sale price: {int(signals['price_not_sale_price'].sum())} (finance/lease/rental)")
    print(f"    >=1 red flag      : {int((scored['red_flag_count'] >= 1).sum())}")
    print(f"    positive condition : {int((scored['condition_score'] > 0.2).sum())}")
    print(f"    negative condition : {int((scored['condition_score'] < -0.2).sum())}")
    print(f"    urgent seller      : {int((scored['urgency_score'] >= 0.33).sum())}")
    age = pd.to_numeric(signals["listing_age_days"], errors="coerce")
    print(f"    age known / >30d   : {int(age.notna().sum())} / {int((age > 30).sum())}")

    worst = scored.nsmallest(8, "condition_score")[["item_id", "condition_score", "red_flags"]]
    best = scored.nlargest(8, "condition_score")[["item_id", "condition_score", "positives"]]
    print("\nMost concerning:\n", worst.to_string(index=False))
    print("\nBest-described:\n", best.to_string(index=False))


if __name__ == "__main__":
    main()
