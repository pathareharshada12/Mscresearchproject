
"""
Generate data/processed/ai_candidate_trends.csv

Purpose:
Turn BERTopic clusters into candidate trend propositions BEFORE human judgement.

This baseline is deliberately conservative:
- It can label a cluster TREND, CONTEXT, ESTABLISHED, or NOISE.
- It refuses to call generic market-size / forecast-report clusters "trends".
- Emergence is only scored for TREND clusters.
- Evidence remains traceable through cluster_evidence_digest.csv.

Run from the project root:
    python generate_ai_candidate_trends.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
import re

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"

METRICS = DATA / "signal_metrics.csv"
SUMMARY = DATA / "topic_summary.csv"
EVIDENCE = DATA / "cluster_evidence_digest.csv"
OUT = DATA / "ai_candidate_trends.csv"

metrics = pd.read_csv(METRICS)
summary = pd.read_csv(SUMMARY)
evidence = pd.read_csv(EVIDENCE)

def txt(v):
    return "" if pd.isna(v) else str(v).strip()

def topic_blob(topic_id):
    s = summary[summary["Topic"].astype(int) == int(topic_id)]
    e = evidence[evidence["topic_id"].astype(int) == int(topic_id)]
    bits = []
    if not s.empty:
        for c in ["Name", "Representation"]:
            if c in s.columns:
                bits.append(txt(s.iloc[0].get(c)))
    for c in ["title", "available_text", "source"]:
        if c in e.columns:
            bits.extend(e[c].fillna("").astype(str).tolist())
    return " ".join(bits).lower(), e

def has_any(blob, terms):
    return any(t in blob for t in terms)

CONTEXT = [
    "market size", "market share", "market insights", "global market",
    "forecast to 20", "cagr", "industry analysis", "market outlook"
]
PROMO = [
    "best activewear", "best brands", "top picks", "need now",
    "shop", "sale", "must-have"
]
COMMUNITY = ["community", "running club", "run club", "social fitness", "collective", "participation"]
WELLNESS = ["wellness", "wellbeing", "recovery", "mental health", "holistic", "walking"]
RETAIL_TECH = ["ar mirror", "augmented reality", "personalisation", "ecommerce", "digital retail", "shopping experience"]
CIRCULAR = ["secondhand", "resale", "repair", "circular", "recycled", "reuse"]
IDENTITY = ["identity", "lifestyle", "fashion", "streetwear", "culture", "everyday"]
INNOVATION = ["innovation", "material", "technology", "performance clothing", "footwear technology"]

def propose_name(blob):
    # More specific combinations first.
    if has_any(blob, COMMUNITY) and has_any(blob, WELLNESS):
        return "Collective Wellness", "Fitness and wellbeing appear to be becoming more socially embedded, with participation increasingly organised around community, belonging and shared experience."
    if has_any(blob, CIRCULAR):
        return "Circular Sportswear Behaviours", "Evidence suggests growing visibility of resale, repair and secondhand behaviours around sportswear, extending product value beyond first purchase."
    if has_any(blob, RETAIL_TECH):
        return "Augmented Sportswear Retail", "Sportswear retail appears to be integrating digital and interactive technologies into product discovery and the physical shopping experience."
    if has_any(blob, WELLNESS):
        return "Everyday Wellness Performance", "Sportswear demand appears to be broadening beyond high-intensity performance toward everyday wellbeing, recovery and lower-pressure movement."
    if has_any(blob, COMMUNITY):
        return "Community-Led Performance", "Participation in sport and fitness appears increasingly connected to community, social identity and collective experiences rather than individual performance alone."
    if has_any(blob, IDENTITY):
        return "Sportswear as Everyday Identity", "Sportswear appears increasingly embedded in everyday lifestyle and identity, blurring boundaries between performance apparel, fashion and cultural expression."
    if has_any(blob, INNOVATION):
        return "Adaptive Performance Innovation", "Product and material innovation appears to be reshaping expectations of performance, functionality and the role of technology in sportswear."
    return "Unresolved Pattern", "The cluster contains related material, but the available evidence does not yet support a sufficiently specific emerging-trend proposition."

def emergence(metric, e):
    # Transparent, bounded evidence-strength score.
    total = float(metric.get("total_articles", 0) or 0)
    sources = float(metric.get("unique_sources", 0) or 0)
    months = float(metric.get("active_months", 0) or 0)
    growth = metric.get("growth_percent", np.nan)

    volume = min(100, total / 20 * 100)
    diversity = min(100, sources / 8 * 100)
    persistence = min(100, months / 12 * 100)

    if pd.isna(growth):
        momentum = 40
    else:
        # 0% growth = 50; +100% = 100; -100% = 0.
        momentum = float(np.clip(50 + float(growth) / 2, 0, 100))

    # Evidence digest source spread adds corroboration.
    corroboration = min(100, e["source"].nunique() / 5 * 100) if "source" in e.columns and len(e) else 0

    return round(
        0.25 * volume +
        0.25 * diversity +
        0.20 * persistence +
        0.20 * momentum +
        0.10 * corroboration
    )

rows = []
for _, metric in metrics.iterrows():
    tid = int(metric["topic_id"])
    blob, ev = topic_blob(tid)

    # First classify whether this deserves trend status at all.
    context_hits = sum(blob.count(t) for t in CONTEXT)
    promo_hits = sum(blob.count(t) for t in PROMO)
    behavioural = any(has_any(blob, group) for group in [COMMUNITY, WELLNESS, RETAIL_TECH, CIRCULAR, IDENTITY, INNOVATION])

    if context_hits >= 2 and not behavioural:
        status = "CONTEXT"
        name = "Market Context — Not an Emerging Trend"
        hypothesis = "This cluster is dominated by market-size, market-outlook or industry-report material rather than evidence of a distinct emerging change."
        rationale = "The material may help frame the commercial environment, but treating forecasts and market-size reporting as weak signals would create circular evidence."
        limitations = "Use as background context only. Do not include in the emerging-trend forecast unless independent behavioural, cultural, technological or commercial evidence is added."
        score = np.nan
    elif promo_hits >= 2 and not behavioural:
        status = "NOISE"
        name = "Promotional / Editorial Cluster"
        hypothesis = "The cluster contains commercially framed or listicle-style material and does not currently support an emerging foresight proposition."
        rationale = "Similarity between promotional articles is not sufficient evidence of consumer or market change."
        limitations = "Requires independent evidence before reconsideration."
        score = np.nan
    else:
        name, hypothesis = propose_name(blob)
        if name == "Unresolved Pattern":
            status = "NOISE"
            rationale = "BERTopic found semantic similarity, but the evidence does not yet express a clear direction of change."
            limitations = "The human reviewer may still identify a meaningful interpretation, but the AI should not manufacture a trend name from weak evidence."
            score = np.nan
        else:
            status = "TREND"
            score = emergence(metric, ev)
            rationale = (
                f"The cluster contains {int(metric.get('total_articles', 0))} items across "
                f"{int(metric.get('unique_sources', 0))} sources and shows recurring evidence related to this proposition."
            )
            limitations = "This is a candidate interpretation generated from the available corpus. Professional review is required to test novelty, context, strategic relevance and whether the evidence genuinely belongs together."

    rows.append({
        "topic_id": tid,
        "ai_status": status,
        "ai_trend_name": name,
        "ai_hypothesis": hypothesis,
        "ai_rationale": rationale,
        "emergence_score": score,
        "limitations": limitations
    })

out = pd.DataFrame(rows).sort_values("topic_id")
OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, index=False, encoding="utf-8-sig")

print(f"Saved {len(out)} AI assessments to: {OUT}")
print(out[["topic_id", "ai_status", "ai_trend_name", "emergence_score"]].to_string(index=False))
