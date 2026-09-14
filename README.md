## Human–AI Foresight

An interactive human-in-the-loop (HITL) research prototype for reviewing AI-assisted fashion-foresight outputs in the UK sportswear market.

The system presents computationally generated candidate evidence, topic groupings and trend interpretations to a professional reviewer. Rather than treating AI output as a finished forecast, the interface records where human judgement changes the result.

Research purpose

The prototype investigates where professional human judgement is required within an AI-assisted forecasting process. Reviewers intervene at three main stages:

Source and evidence validation - keep, reject or mark an item as uncertain.

Grouping validation - keep, reframe, merge or reject a computational cluster.

Interpretation validation - accept, rename, rewrite or reject the AI interpretation.

The reviewer then makes a final forecast decision and may add professional, cultural and strategic context that the computational analysis missed.

Main features

Reviews representative evidence for each candidate signal area

Records evidence-level acceptance and rejection decisions

Evaluates whether semantically similar documents form a meaningful trend group

Compares the original AI interpretation with the human-refined interpretation

Captures confidence, strategic implications and possible future development

Produces downloadable CSV files containing the human interventions

Displays a final human-refined forecast after all candidate areas are reviewed

## Repository structure

`prototype2/` is the final artefact.  `app.hitl.py` runs the Human-in-the-Loop interface , while `prototype2/evidence_qualifier.py` prepares the representative evidence shown to reviewers.

`signal-builder.py` contains the main computational pipeline. It processes the collected articles, generates semantic representations and groups similar documents into candidate signal areas.

`ai_trend_generator.py` uses the grouped evidence to generate the candidate trend names, hypotheses, rationales and limitations evaluated through the interface.

`prototype2/app.py` is an earlier version of the Human-in-the-Loop application retained to document the prototype’s development. The adopted and evaluated version is `app.hitl.py`.

data/processed/ contains the frozen outputs passed between these scripts, including the processed articles, topic assignments, signal metrics, evidence digest and AI-generated candidate trends. data/human_reviews/ contains the evidence-level and cluster-level decisions recorded during the final professional evaluation.



This prototype was developed for academic research and is not a commercial forecasting product. Before making the repository public, remove consent forms, identifiable participant information, API keys, passwords and any restricted or copyrighted datasets. Consider excluding generated human-review files from Git if they contain research responses.

Author

Harshada Dattatray Pathare
MSc Applied Machine Learning for Creatives
University of the Arts London — Creative Computing Institute

Disclaimer

The candidate areas displayed by the system are computational outputs for human evaluation. They should not be interpreted as validated trends or professional forecasting advice without expert review.