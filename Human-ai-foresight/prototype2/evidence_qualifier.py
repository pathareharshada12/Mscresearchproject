
import re
import pandas as pd
import numpy as np
from urllib.parse import urlparse

ROLE_PRIMARY = "Primary signal"
ROLE_CORROBORATING = "Corroborating evidence"
ROLE_CONTEXT = "Context"
ROLE_BRAND = "Brand activity"
ROLE_FORECAST = "Forecast / editorial"
ROLE_PROMO = "Commercial / promotional"
ROLE_OOS = "Out of scope"
ROLE_UNVERIFIED = "Candidate lead — retrieve source"

FORECAST_TERMS = [
    "trend", "trends", "outlook", "forecast", "future of", "coming in 2026",
    "state of fashion", "what to wear", "best activewear", "best brands",
    "top trends", "biggest wellness trends"
]
PROMO_TERMS = [
    "best activewear brands", "need now", "get you into gear", "top picks",
    "shop now", "must-have", "sale", "discount", "gift guide"
]
MARKET_SIZE_TERMS = [
    "market size", "market share", "cagr", "to reach usd", "growth [",
    "market outlook", "competitive landscape"
]
PR_SOURCES = ["pr newswire", "business wire", "globe newswire"]
BRAND_SOURCES = [
    "about nike", "adidas", "puma", "gymshark", "lululemon",
    "under armour", "asics", "new balance"
]
OUT_SCOPE_TERMS = [
    "online grocery", "coca-cola", "home gym equipment", "sports betting",
    "winter sports equipment", "dubai fitness", "brazil premium sportswear",
    "france premium sportswear"
]
BEHAVIOUR_TERMS = [
    "consumers", "consumer behaviour", "shopping habits", "brits prefer",
    "demand", "shopping", "social shopping", "wellbeing", "wellness",
    "participation", "community", "culture"
]
INNOVATION_TERMS = [
    "launches", "introduces", "innovation", "circular", "repair",
    "resale", "material", "technology", "ar mirror", "locker"
]

def _txt(v):
    return "" if pd.isna(v) else str(v).strip()

def _contains(blob, terms):
    return any(t in blob for t in terms)

def has_real_body(row):
    """
    The old corpus often repeats title into text.
    A candidate cannot become validated evidence without substantive source text.
    """
    title = re.sub(r"\W+", " ", _txt(row.get("title"))).lower().strip()
    body = re.sub(r"\W+", " ", _txt(row.get("text"))).lower().strip()
    if len(body) < 180:
        return False
    # detect title duplication / near-duplication
    if title and (body == title or body.startswith(title + " " + title)):
        return False
    unique_words = len(set(body.split()))
    return unique_words >= 35

def classify_candidate(row):
    title = _txt(row.get("title")).lower()
    body = _txt(row.get("text")).lower()
    source = _txt(row.get("source")).lower()
    query = _txt(row.get("search_query")).lower()
    blob = " ".join([title, body, source, query])

    semantic = float(row.get("semantic_relevance", 0) or 0)
    margin = float(row.get("scope_margin", 0) or 0)
    source_body_ok = has_real_body(row)

    flags = []
    role = ROLE_UNVERIFIED
    eligible = False

    # Hard exclusions / non-evidence forms first.
    if _contains(blob, OUT_SCOPE_TERMS):
        role = ROLE_OOS
        flags.append("Wrong market/category/geography or accidental semantic match")
    elif source in PR_SOURCES or any(s in source for s in PR_SOURCES):
        role = ROLE_PROMO
        flags.append("Press-release distribution source")
    elif _contains(title, PROMO_TERMS):
        role = ROLE_PROMO
        flags.append("Listicle / shopping / promotional framing")
    elif _contains(title, MARKET_SIZE_TERMS):
        role = ROLE_CONTEXT
        flags.append("Market-size material is context, not a weak signal by itself")
    elif _contains(title, FORECAST_TERMS):
        role = ROLE_FORECAST
        flags.append("Pre-existing forecast/editorial interpretation; avoid circular evidence")
    elif any(s in source for s in BRAND_SOURCES):
        role = ROLE_BRAND
        flags.append("First-party brand activity; useful as an observable action, not consumer proof")
    elif _contains(blob, BEHAVIOUR_TERMS):
        role = ROLE_CORROBORATING
        flags.append("Potential behavioural evidence; source content must be verified")
    elif _contains(blob, INNOVATION_TERMS):
        role = ROLE_BRAND
        flags.append("Potential observable market/brand activity")
    else:
        role = ROLE_UNVERIFIED
        flags.append("Relevant lead but evidence role cannot be established from metadata/title alone")

    # Transparent quality score: qualification, not forecast confidence.
    score = 35
    score += min(25, max(0, semantic) * 35)
    score += min(15, max(0, margin) * 40)

    if source_body_ok:
        score += 20
    else:
        flags.append("Insufficient source body: current text is too short or title-like")

    if role in {ROLE_PROMO, ROLE_OOS}:
        score -= 35
    elif role == ROLE_FORECAST:
        score -= 20
    elif role == ROLE_CONTEXT:
        score -= 5
    elif role in {ROLE_BRAND, ROLE_CORROBORATING}:
        score += 5

    score = int(np.clip(round(score), 0, 100))

    # Strict gate: no substantive body = no entry to theme detection.
    if source_body_ok and role in {ROLE_PRIMARY, ROLE_CORROBORATING, ROLE_BRAND} and score >= 60:
        eligible = True

    if role == ROLE_CONTEXT:
        use = "Context only"
    elif eligible:
        use = "Eligible for AI detection"
    elif role in {ROLE_PROMO, ROLE_OOS, ROLE_FORECAST}:
        use = "Exclude from AI detection"
    else:
        use = "Retrieve / verify before detection"

    return pd.Series({
        "evidence_role": role,
        "evidence_quality_score": score,
        "source_body_verified": source_body_ok,
        "detection_status": use,
        "qualification_reason": "; ".join(flags),
        "eligible_for_detection": eligible
    })

def qualify_dataframe(df):
    out = df.copy()
    q = out.apply(classify_candidate, axis=1)
    return pd.concat([out, q], axis=1)
