
# HITL Foresight — Evidence Qualification Layer

This module implements the methodological repair prompted by expert feedback.

## Core rule
Semantic relevance is not evidence validity.

A retrieved item is assigned:
- evidence role
- evidence quality score
- source-body verification
- detection status
- qualification reason
- eligibility for AI detection

The gate deliberately blocks:
- promotional/listicle content
- PR distribution
- pre-existing forecast/editorial articles as direct weak-signal evidence
- generic market-size reports as direct signals
- out-of-scope semantic matches
- title-only records without substantive source text

## Run
pip install -r requirements.txt
streamlit run app.py

## Next
Build the source retrieval/verification stage, then rerun this gate before embeddings/clustering.
