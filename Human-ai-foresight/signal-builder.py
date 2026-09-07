
"""
signal_builder.py
-----------------
Builds a defensible signal dataset BEFORE BERTopic / embeddings.

INPUT
-----
data/processed/raw_signal_sources.csv

Minimum columns:
    title, source, url, date, text

Optional:
    source_type, geography, engagement

OUTPUTS
-------
data/processed/validated_signals.csv
data/processed/signal_rejections.csv
data/processed/signal_builder_audit.csv

METHODOLOGICAL RULE
-------------------
A source is not automatically a signal.

The pipeline asks:
1. Is there a traceable source?
2. Is there substantive source text?
3. Can we extract an OBSERVABLE phenomenon rather than somebody else's forecast?
4. Is it relevant to UK sportswear / adjacent consumer behaviour?
5. Which evidence family does it belong to?

Only validated observations proceed to clustering.

Install:
    python -m pip install google-genai pandas

PowerShell:
    $env:GEMINI_API_KEY="YOUR_KEY"
    python Human-ai-foresight/signal_builder.py
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse
import hashlib
import json
import os
import re
import time

import pandas as pd
from google import genai
from google.genai import types


# -------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"

INPUT = DATA / "raw_signal_sources.csv"
VALIDATED = DATA / "validated_signals.csv"
REJECTED = DATA / "signal_rejections.csv"
AUDIT = DATA / "signal_builder_audit.csv"

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set.\n"
        '$env:GEMINI_API_KEY="YOUR_KEY"'
    )

if not INPUT.exists():
    raise FileNotFoundError(
        f"Missing input file: {INPUT}\n\n"
        "Create raw_signal_sources.csv with at least:\n"
        "title, source, url, date, text"
    )

client = genai.Client(api_key=API_KEY)


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

EVIDENCE_FAMILIES = {
    "COMMUNITY",
    "BEHAVIOURAL",
    "BRAND_MARKET_ACTION",
    "RESEARCH_DATA",
    "CULTURAL",
    "CONTEXT",
}

VALIDATION_STATUSES = {
    "VALID_SIGNAL",
    "CONTEXT_ONLY",
    "REJECT",
    "NEEDS_VERIFICATION",
}

# Things that should NOT independently become weak signals.
FORECAST_LANGUAGE = [
    "trend forecast",
    "trends for 2026",
    "future trends",
    "what will trend",
    "market forecast",
    "market outlook",
    "predicted to",
    "expected to reach",
    "cagr",
    "market size",
]

PROMOTIONAL_LANGUAGE = [
    "shop now",
    "best activewear",
    "best brands",
    "must-have",
    "top picks",
    "sale",
    "discount",
    "affiliate",
    "sponsored",
]


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def clean(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except Exception:
        return False


def source_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def substantive_text(title: str, text: str) -> bool:
    """
    Prevents title-only records from becoming signals.
    """
    t = re.sub(r"\W+", " ", title.lower()).strip()
    body = re.sub(r"\W+", " ", text.lower()).strip()

    if len(body) < 180:
        return False

    if body == t:
        return False

    # Old corpus sometimes duplicated the title.
    if t and body.startswith(t + " " + t):
        return False

    return len(set(body.split())) >= 35


def deterministic_signal_id(url: str, observation: str) -> str:
    seed = f"{url}|{observation}".encode("utf-8")
    return "SIG-" + hashlib.sha1(seed).hexdigest()[:10].upper()


def precheck(row: pd.Series) -> tuple[bool, list[str]]:
    flags = []

    title = clean(row.get("title"))
    text = clean(row.get("text"))
    url = clean(row.get("url"))

    if not valid_url(url):
        flags.append("Missing or invalid traceable URL")

    if not substantive_text(title, text):
        flags.append("Insufficient substantive source text")

    return len(flags) == 0, flags


def build_prompt(row: pd.Series) -> str:
    title = clean(row.get("title"))
    source = clean(row.get("source"))
    url = clean(row.get("url"))
    date = clean(row.get("date"))
    text = clean(row.get("text"))
    supplied_type = clean(row.get("source_type"))
    supplied_geo = clean(row.get("geography"))

    # Bound the prompt while keeping enough evidence.
    if len(text) > 7000:
        text = text[:7000].rsplit(" ", 1)[0] + "..."

    return f"""
You are extracting research signals for an MSc Human–AI strategic foresight
prototype focused on the UK sportswear market.

CRITICAL DISTINCTION:
A SOURCE is not automatically a SIGNAL.

A valid signal must describe an observable phenomenon:
- behaviour
- action
- practice
- adoption
- launch
- participation pattern
- cultural manifestation
- measurable change
- repeated community concern or behaviour

Do NOT turn somebody else's forecast into a new signal.

CLASSIFY THE SOURCE INTO ONE EVIDENCE FAMILY:

COMMUNITY
Naturalistic discussion or community activity, such as forums or community spaces.

BEHAVIOURAL
Observed consumer/user behaviour, participation, search, shopping or usage patterns.

BRAND_MARKET_ACTION
A concrete company/market action: launch, investment, partnership, service,
retail experiment, product/material change, programme or business response.

RESEARCH_DATA
Survey, study, official statistics, participation data, credible measured research.

CULTURAL
Observable cultural practice, event, community formation, aesthetic/lifestyle
manifestation or emerging social practice.

CONTEXT
Market size, broad industry background, macro conditions or somebody else's
trend/forecast interpretation. Context can support research but must NOT enter
weak-signal clustering as if it were an observed emerging phenomenon.

VALIDATION STATUS:

VALID_SIGNAL
A traceable, relevant observable phenomenon is present.

CONTEXT_ONLY
Useful background, but not a weak signal.

REJECT
Promotional/listicle/irrelevant/circular/incoherent material.

NEEDS_VERIFICATION
Potentially useful, but the supplied evidence is too ambiguous to establish the observation.

RULES:
1. Use only the supplied source.
2. Do not invent facts.
3. Do not infer growth unless the source itself supports change/growth.
4. Do not call something "emerging" merely because the article says it is a trend.
5. Keep observation factual and neutral.
6. Separate observation from interpretation.
7. UK relevance can be DIRECT, ADJACENT, or UNCLEAR.
8. A global source can be ADJACENT but must not be presented as UK evidence.
9. Promotional articles and shopping listicles should normally be REJECT.
10. Market-size and generic forecast reports should normally be CONTEXT_ONLY.

SOURCE
------
Title: {title}
Publisher/source: {source}
URL: {url}
Date: {date}
Supplied source type: {supplied_type}
Supplied geography: {supplied_geo}

SOURCE TEXT
-----------
{text}

Return ONLY valid JSON:

{{
  "validation_status": "VALID_SIGNAL | CONTEXT_ONLY | REJECT | NEEDS_VERIFICATION",
  "evidence_family": "COMMUNITY | BEHAVIOURAL | BRAND_MARKET_ACTION | RESEARCH_DATA | CULTURAL | CONTEXT",
  "observation": "one concise factual statement describing what is observable; empty if no valid observation",
  "evidence_excerpt": "short supporting excerpt/paraphrase grounded in the source",
  "geography": "specific geography if supported, otherwise Unknown",
  "uk_relevance": "DIRECT | ADJACENT | UNCLEAR",
  "why_relevant": "brief explanation of relevance to UK sportswear/consumer foresight",
  "independence_note": "whether this appears to be original evidence, first-party brand action, secondary reporting, forecast/editorial, or promotional",
  "validation_reason": "why it received this status",
  "confidence": 0
}}

confidence is confidence in the EXTRACTION/CLASSIFICATION, not confidence that
the future trend will happen. Use an integer from 0 to 100.
""".strip()


def parse_json(text: str) -> dict:
    raw = text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def validate_ai_result(result: dict) -> dict:
    status = clean(result.get("validation_status")).upper()
    family = clean(result.get("evidence_family")).upper()

    if status not in VALIDATION_STATUSES:
        raise ValueError(f"Invalid validation_status: {status}")

    if family not in EVIDENCE_FAMILIES:
        raise ValueError(f"Invalid evidence_family: {family}")

    result["validation_status"] = status
    result["evidence_family"] = family

    for field in [
        "observation",
        "evidence_excerpt",
        "geography",
        "uk_relevance",
        "why_relevant",
        "independence_note",
        "validation_reason",
    ]:
        result[field] = clean(result.get(field))

    try:
        result["confidence"] = max(0, min(100, int(result.get("confidence", 0))))
    except Exception:
        result["confidence"] = 0

    # Context is never allowed to sneak into clustering.
    if family == "CONTEXT" and status == "VALID_SIGNAL":
        result["validation_status"] = "CONTEXT_ONLY"
        result["validation_reason"] += " Context evidence is excluded from signal clustering."

    return result


# -------------------------------------------------------------------
# LOAD
# -------------------------------------------------------------------

df = pd.read_csv(INPUT)

required = ["title", "source", "url", "date", "text"]
missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(
        f"raw_signal_sources.csv is missing columns: {missing}"
    )


# -------------------------------------------------------------------
# PROCESS
# -------------------------------------------------------------------

records = []

for idx, row in df.iterrows():
    title = clean(row.get("title"))
    url = clean(row.get("url"))

    print(f"[{idx + 1}/{len(df)}] {title[:80]}")

    passed, precheck_flags = precheck(row)

    if not passed:
        result = {
            "validation_status": "NEEDS_VERIFICATION",
            "evidence_family": "CONTEXT",
            "observation": "",
            "evidence_excerpt": "",
            "geography": clean(row.get("geography")) or "Unknown",
            "uk_relevance": "UNCLEAR",
            "why_relevant": "",
            "independence_note": "",
            "validation_reason": "; ".join(precheck_flags),
            "confidence": 100,
        }
    else:
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=build_prompt(row),
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
            )

            result = validate_ai_result(
                parse_json(response.text)
            )

        except Exception as exc:
            result = {
                "validation_status": "NEEDS_VERIFICATION",
                "evidence_family": "CONTEXT",
                "observation": "",
                "evidence_excerpt": "",
                "geography": clean(row.get("geography")) or "Unknown",
                "uk_relevance": "UNCLEAR",
                "why_relevant": "",
                "independence_note": "",
                "validation_reason": f"AI extraction failed: {exc}",
                "confidence": 0,
            }

    observation = result["observation"]

    signal_id = (
        deterministic_signal_id(url, observation)
        if observation
        else ""
    )

    records.append({
        "source_row": idx,
        "signal_id": signal_id,
        "date": clean(row.get("date")),
        "source": clean(row.get("source")),
        "source_domain": source_domain(url),
        "url": url,
        "title": title,
        "source_type_original": clean(row.get("source_type")),
        "evidence_family": result["evidence_family"],
        "geography": result["geography"],
        "uk_relevance": result["uk_relevance"],
        "observation": observation,
        "evidence_excerpt": result["evidence_excerpt"],
        "engagement": clean(row.get("engagement")),
        "independence_note": result["independence_note"],
        "why_relevant": result["why_relevant"],
        "validation_status": result["validation_status"],
        "validation_reason": result["validation_reason"],
        "extraction_confidence": result["confidence"],
        "retrieved_text_available": substantive_text(
            title,
            clean(row.get("text"))
        ),
        "model": MODEL,
    })

    time.sleep(0.25)


audit = pd.DataFrame(records)


# -------------------------------------------------------------------
# FINAL HARD GATE
# -------------------------------------------------------------------

# A row enters the valid signal pool only when:
# - AI extracted a valid signal
# - observation exists
# - URL is traceable
# - substantive source text exists
# - UK relevance is at least direct/adjacent
valid_mask = (
    (audit["validation_status"] == "VALID_SIGNAL")
    & audit["observation"].astype(bool)
    & audit["url"].apply(valid_url)
    & audit["retrieved_text_available"]
    & audit["uk_relevance"].isin(["DIRECT", "ADJACENT"])
)

validated = audit[valid_mask].copy()

rejections = audit[~valid_mask].copy()


# -------------------------------------------------------------------
# SAVE
# -------------------------------------------------------------------

DATA.mkdir(parents=True, exist_ok=True)

audit.to_csv(
    AUDIT,
    index=False,
    encoding="utf-8-sig"
)

validated.to_csv(
    VALIDATED,
    index=False,
    encoding="utf-8-sig"
)

rejections.to_csv(
    REJECTED,
    index=False,
    encoding="utf-8-sig"
)


# -------------------------------------------------------------------
# REPORT
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("SIGNAL BUILDER COMPLETE")
print("=" * 70)

print(f"Raw sources:        {len(df)}")
print(f"Validated signals:  {len(validated)}")
print(f"Not admitted:       {len(rejections)}")

if len(validated):
    print("\nValidated evidence families:")
    print(
        validated["evidence_family"]
        .value_counts()
        .to_string()
    )

    print("\nUK relevance:")
    print(
        validated["uk_relevance"]
        .value_counts()
        .to_string()
    )

print(f"\nSaved:\n{VALIDATED}")
print(f"{REJECTED}")
print(f"{AUDIT}")

print(
    "\nNEXT: cluster ONLY validated_signals.csv. "
    "Do not feed CONTEXT_ONLY / REJECT / NEEDS_VERIFICATION rows into BERTopic."
)
