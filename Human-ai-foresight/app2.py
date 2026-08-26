import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime


# -----------------------------
# PAGE SETUP
# -----------------------------

st.set_page_config(
    page_title="Human–AI Foresight",
    page_icon="◈",
    layout="wide"
)


# -----------------------------
# FILES
# -----------------------------

ARTICLES_FILE = Path("data/processed/articles_with_topics.csv")
METRICS_FILE = Path("data/processed/signal_metrics.csv")
TOPIC_SUMMARY_FILE = Path("data/processed/topic_summary.csv")
EVIDENCE_FILE = Path("data/processed/cluster_evidence_digest.csv")

REVIEWS_FOLDER = Path("data/human_reviews")
REVIEWS_FOLDER.mkdir(parents=True, exist_ok=True)


# -----------------------------
# LOAD DATA
# -----------------------------

@st.cache_data
def load_data():

    articles = pd.read_csv(ARTICLES_FILE)
    metrics = pd.read_csv(METRICS_FILE)
    topic_summary = pd.read_csv(TOPIC_SUMMARY_FILE)
    evidence = pd.read_csv(EVIDENCE_FILE)

    articles["date"] = pd.to_datetime(
        articles["date"],
        errors="coerce"
    )

    evidence["date"] = pd.to_datetime(
        evidence["date"],
        errors="coerce"
    )

    return articles, metrics, topic_summary, evidence


articles, metrics, topic_summary, evidence = load_data()


# -----------------------------
# SESSION
# -----------------------------

defaults = {
    "started": False,
    "current_index": 0,
    "participant_id": "",
    "participant_role": "",
    "years_experience": 0
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


topic_ids = sorted(
    metrics["topic_id"]
    .dropna()
    .astype(int)
    .tolist()
)


# -----------------------------
# HELPERS
# -----------------------------

def participant_file():

    return (
        REVIEWS_FOLDER
        / f"{st.session_state.participant_id}_reviews.csv"
    )


def get_saved_reviews():

    file = participant_file()

    if file.exists():
        return pd.read_csv(file)

    return pd.DataFrame()


def save_review(review):

    file = participant_file()
    new = pd.DataFrame([review])

    if file.exists():

        existing = pd.read_csv(file)

        existing = existing[
            existing["topic_id"] != review["topic_id"]
        ]

        new = pd.concat(
            [existing, new],
            ignore_index=True
        )

    new.to_csv(
        file,
        index=False,
        encoding="utf-8-sig"
    )


def get_topic(topic_id):

    metric = metrics[
        metrics["topic_id"] == topic_id
    ].iloc[0]

    topic_articles = articles[
        articles["topic_id"] == topic_id
    ].sort_values(
        "date",
        ascending=False
    )

    summary_match = topic_summary[
        topic_summary["Topic"] == topic_id
    ]

    summary = (
        summary_match.iloc[0]
        if not summary_match.empty
        else None
    )

    topic_evidence = evidence[
        evidence["topic_id"] == topic_id
    ].sort_values(
        "evidence_number"
    )

    return (
        metric,
        topic_articles,
        summary,
        topic_evidence
    )


def clean_terms(summary):

    if summary is None:
        return []

    value = summary.get(
        "Representation",
        ""
    )

    if not isinstance(value, str):
        return []

    value = (
        value
        .replace("[", "")
        .replace("]", "")
        .replace("'", "")
    )

    stop = {
        "the", "and", "of", "in",
        "to", "is", "how", "by",
        "for", "with", "on"
    }

    terms = [
        x.strip()
        for x in value.split(",")
        if x.strip()
    ]

    return [
        x for x in terms
        if x.lower() not in stop
    ][:6]


def evidence_cautions(metric, topic_articles):

    cautions = []

    concentration = float(
        metric["source_concentration"]
    )

    growth = metric["growth_percent"]

    if concentration >= 0.30:

        top_source = (
            topic_articles["source"]
            .value_counts()
            .idxmax()
        )

        cautions.append(
            f"High source concentration around {top_source}."
        )

    if (
        not pd.isna(growth)
        and growth <= -40
    ):
        cautions.append(
            "Recent coverage has declined substantially."
        )

    titles = " ".join(
        topic_articles["title"]
        .fillna("")
        .astype(str)
    ).lower()

    market_terms = [
        "market size",
        "market forecast",
        "cagr",
        "billion",
        "market analysis"
    ]

    if sum(
        term in titles
        for term in market_terms
    ) >= 2:
        cautions.append(
            "Cluster is heavily influenced by commercial market reports."
        )

    return cautions


# -----------------------------
# INTRO
# -----------------------------

if not st.session_state.started:

    st.title("Human–AI Foresight")

    st.subheader(
        "UK Sportswear · Technology & Consumer Behaviour"
    )

    st.write(
        """
        This system has analysed a horizon-scanning dataset
        and identified **7 possible signal areas**.

        These are not confirmed trends.

        Your role is to review the evidence and decide
        which patterns are meaningful enough to enter a forecast.
        """
    )

    st.info(
        """
        **What you are contributing**

        The system can detect similarity, recurrence and changes
        in coverage.

        You bring professional judgement:
        context, novelty, relevance, interpretation and strategic meaning.
        """
    )

    col1, col2 = st.columns(2)

    with col1:

        participant_id = st.text_input(
            "Participant ID",
            placeholder="e.g. P01"
        )

        role = st.text_input(
            "Professional role",
            placeholder="e.g. Trend Forecaster"
        )

    with col2:

        experience = st.number_input(
            "Years of relevant experience",
            min_value=0,
            max_value=50,
            value=0
        )

    with st.expander(
        "How the system works"
    ):

        st.write(
            """
            Articles were represented using Sentence-BERT embeddings
            and grouped using BERTopic.

            The system also calculates evidence indicators including:

            - number of articles
            - source diversity
            - time spread
            - recent changes in coverage

            Representative evidence is then surfaced for human review.
            """
        )

    if st.button(
        "Enter signal dashboard →",
        type="primary",
        use_container_width=True
    ):

        if not participant_id.strip():

            st.error(
                "Please enter a Participant ID."
            )

        else:

            st.session_state.participant_id = (
                participant_id.strip()
            )

            st.session_state.participant_role = (
                role.strip()
            )

            st.session_state.years_experience = (
                experience
            )

            st.session_state.started = True

            st.rerun()

    st.stop()


# -----------------------------
# COMPLETE SCREEN
# -----------------------------

saved = get_saved_reviews()

if (
    not saved.empty
    and saved["topic_id"].nunique()
    >= len(topic_ids)
):

    st.title("Forecast review complete")

    included = saved[
        saved["signal_decision"]
        == "Include in forecast"
    ]

    excluded = saved[
        saved["signal_decision"]
        == "Exclude from forecast"
    ]

    more = saved[
        saved["signal_decision"]
        == "Need more evidence"
    ]

    a, b, c, d = st.columns(4)

    a.metric(
        "Reviewed",
        len(saved)
    )

    b.metric(
        "Included",
        len(included)
    )

    c.metric(
        "Excluded",
        len(excluded)
    )

    d.metric(
        "Need more evidence",
        len(more)
    )

    st.divider()

    st.header("Your Human–AI Forecast")

    if included.empty:

        st.warning(
            "No machine-generated signals were included."
        )

    else:

        st.title(
            "UK Sportswear Outlook 2026–2029"
        )

        st.caption(
            "Human–AI Strategic Foresight Output"
        )

        for number, (_, row) in enumerate(
            included.iterrows(),
            start=1
        ):

            st.divider()

            name = row.get(
                "human_signal_name",
                ""
            )

            if pd.isna(name) or not str(name).strip():
                name = f"Signal {number}"

            st.header(
                f"{number:02d} — {name}"
            )

            st.markdown(
                "**What is changing**"
            )

            st.write(
                row.get(
                    "interpretation",
                    ""
                )
            )

            st.markdown(
                "**Why it matters**"
            )

            st.write(
                row.get(
                    "strategic_implication",
                    ""
                )
            )

            st.markdown(
                "**What the system missed**"
            )

            st.write(
                row.get(
                    "missing_context",
                    ""
                )
            )

            st.markdown(
                "**Future direction**"
            )

            st.write(
                row.get(
                    "future_development",
                    ""
                )
            )

            st.caption(
                f"Professional confidence: "
                f"{row.get('confidence', '')}/5"
            )

    st.divider()

    st.header(
        "Anything missing from the machine?"
    )

    missing = st.radio(
        "Did the computational system fail to surface an important signal?",
        ["No", "Yes"],
        horizontal=True
    )

    if missing == "Yes":

        st.text_input(
            "Signal name"
        )

        st.text_area(
            "Describe the missing signal"
        )

        st.text_area(
            "Why do you think the system missed it?"
        )

    st.download_button(
        "Download review data",
        data=saved.to_csv(
            index=False
        ).encode("utf-8"),
        file_name=(
            f"{st.session_state.participant_id}_reviews.csv"
        ),
        mime="text/csv"
    )

    st.stop()


# -----------------------------
# DASHBOARD
# -----------------------------

st.title(
    "Signal Dashboard"
)

st.caption(
    "Review the machine-generated patterns before entering the forecast."
)


summary_cols = st.columns(
    min(len(topic_ids), 4)
)

for i, topic_id in enumerate(
    topic_ids
):

    metric, topic_articles, summary, topic_evidence = (
        get_topic(topic_id)
    )

    growth = metric["growth_percent"]

    if pd.isna(growth):
        growth_text = "N/A"
    else:
        growth_text = f"{growth:+.0f}%"

    cautions = evidence_cautions(
        metric,
        topic_articles
    )

    status = (
        "Caution"
        if cautions
        else "No major caution"
    )

    column = summary_cols[
        i % len(summary_cols)
    ]

    with column:

        st.markdown(
            f"### Signal {i + 1}"
        )

        st.write(
            f"**{int(metric['total_articles'])} articles**"
        )

        st.write(
            f"{int(metric['unique_sources'])} sources"
        )

        st.write(
            f"Coverage: {growth_text}"
        )

        st.caption(
            status
        )


st.divider()


# -----------------------------
# SIGNAL NAVIGATION
# -----------------------------

current_index = st.session_state.current_index

topic_id = topic_ids[
    current_index
]

metric, topic_articles, summary, topic_evidence = (
    get_topic(topic_id)
)

terms = clean_terms(
    summary
)

cautions = evidence_cautions(
    metric,
    topic_articles
)


progress = (
    current_index + 1
) / len(topic_ids)

st.progress(progress)

st.caption(
    f"Signal {current_index + 1} of {len(topic_ids)}"
)


# -----------------------------
# SIGNAL VIEW
# -----------------------------

st.title(
    f"Possible Signal {current_index + 1}"
)

if terms:

    st.write(
        "**Machine-detected themes:** "
        +
        " · ".join(terms)
    )


st.subheader(
    "Evidence snapshot"
)

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Articles",
    int(metric["total_articles"])
)

m2.metric(
    "Sources",
    int(metric["unique_sources"])
)

m3.metric(
    "Active months",
    int(metric["active_months"])
)

growth = metric["growth_percent"]

growth_text = (
    "N/A"
    if pd.isna(growth)
    else f"{growth:+.0f}%"
)

m4.metric(
    "Recent coverage",
    growth_text
)


# -----------------------------
# EVIDENCE
# -----------------------------

st.subheader(
    "Representative evidence"
)

st.caption(
    """
    Five representative items were selected from this grouping
    using semantic similarity.
    """
)

for _, row in topic_evidence.iterrows():

    title = row.get(
        "title",
        "Untitled"
    )

    source = row.get(
        "source",
        "Unknown source"
    )

    date = row.get(
        "date"
    )

    if pd.notna(date):

        date_text = date.strftime(
            "%d %b %Y"
        )

    else:

        date_text = "Unknown date"

    with st.expander(
        title
    ):

        st.caption(
            f"{source} · {date_text}"
        )

        text = row.get(
            "available_text",
            ""
        )

        if (
            isinstance(text, str)
            and text.strip()
        ):
            st.write(
                text[:500]
            )

        url = row.get(
            "url",
            ""
        )

        if (
            isinstance(url, str)
            and url.strip()
        ):

            st.markdown(
                f"[Open original article]({url})"
            )


# -----------------------------
# CAUTIONS
# -----------------------------

st.subheader(
    "Evidence quality"
)

if cautions:

    for caution in cautions:

        st.warning(
            caution
        )

else:

    st.success(
        """
        No major automated evidence-quality warning was detected.

        This does not mean the pattern is a valid signal.
        """
    )


with st.expander(
    f"Browse all {len(topic_articles)} items"
):

    for _, row in topic_articles.iterrows():

        st.markdown(
            f"**{row['title']}**"
        )

        st.caption(
            str(row.get(
                "source",
                ""
            ))
        )

        st.divider()


# -----------------------------
# PROFESSIONAL JUDGEMENT
# -----------------------------

st.divider()

st.header(
    "Professional judgement"
)

st.write(
    """
    The machine has surfaced the pattern.
    You decide whether it deserves a place in the forecast.
    """
)


classification = st.radio(
    "How would you classify it?",
    [
        "Emerging signal",
        "Already established",
        "Relevant context, but not a signal",
        "Noise / unrelated",
        "Need more evidence"
    ],
    index=None,
    key=f"classification_{topic_id}"
)


signal_decision = st.radio(
    "Forecast decision",
    [
        "Include in forecast",
        "Exclude from forecast",
        "Need more evidence"
    ],
    index=None,
    horizontal=True,
    key=f"decision_{topic_id}"
)


human_signal_name = ""
interpretation = ""
missing_context = ""
strategic_implication = ""
future_development = ""
rejection_reason = ""
additional_evidence = []
evidence_notes = ""


if signal_decision == "Include in forecast":

    st.success(
        "This signal will be developed into the final forecast."
    )

    human_signal_name = st.text_input(
        "Signal name",
        placeholder=(
            "How would you frame this professionally?"
        ),
        key=f"name_{topic_id}"
    )

    interpretation = st.text_area(
        "What is actually changing?",
        height=110,
        key=f"interpretation_{topic_id}"
    )

    missing_context = st.text_area(
        "What is missing from the computational analysis?",
        placeholder=(
            "Cultural context, consumer motivation, "
            "commercial knowledge, historical context..."
        ),
        height=100,
        key=f"context_{topic_id}"
    )

    strategic_implication = st.text_area(
        "Why does this matter for UK sportswear?",
        height=100,
        key=f"implication_{topic_id}"
    )

    future_development = st.text_area(
        "How could this develop over the next 2–3 years?",
        height=110,
        key=f"future_{topic_id}"
    )


elif signal_decision == "Exclude from forecast":

    rejection_reason = st.radio(
        "Why should it be excluded?",
        [
            "Already established",
            "Evidence too weak",
            "Articles do not belong together",
            "Not relevant enough to UK sportswear",
            "Not strategically important",
            "Other"
        ],
        index=None,
        key=f"reject_{topic_id}"
    )


elif signal_decision == "Need more evidence":

    additional_evidence = st.multiselect(
        "What evidence would you want next?",
        [
            "Consumer behaviour data",
            "Sales / market data",
            "Competitor activity",
            "Social / cultural signals",
            "Product activity",
            "Search / social media data",
            "Historical context",
            "Expert knowledge",
            "Other"
        ],
        key=f"additional_{topic_id}"
    )

    evidence_notes = st.text_area(
        "What would you investigate?",
        key=f"notes_{topic_id}"
    )


confidence = st.slider(
    "Confidence in this judgement",
    1,
    5,
    3,
    key=f"confidence_{topic_id}"
)


# -----------------------------
# SAVE / NEXT
# -----------------------------

st.divider()

col_back, col_next = st.columns(
    [1, 3]
)

with col_back:

    if current_index > 0:

        if st.button(
            "← Previous"
        ):

            st.session_state.current_index -= 1
            st.rerun()


with col_next:

    if st.button(
        "Save & Next →",
        type="primary",
        use_container_width=True
    ):

        if classification is None:

            st.error(
                "Please classify the signal."
            )

        elif signal_decision is None:

            st.error(
                "Please choose a forecast decision."
            )

        elif (
            signal_decision == "Include in forecast"
            and not human_signal_name.strip()
        ):

            st.error(
                "Please give the signal a name."
            )

        elif (
            signal_decision == "Exclude from forecast"
            and not rejection_reason
        ):

            st.error(
                "Please select a reason for exclusion."
            )

        else:

            review = {

                "participant_id":
                    st.session_state.participant_id,

                "participant_role":
                    st.session_state.participant_role,

                "years_experience":
                    st.session_state.years_experience,

                "topic_id":
                    topic_id,

                "classification":
                    classification,

                "signal_decision":
                    signal_decision,

                "human_signal_name":
                    human_signal_name,

                "interpretation":
                    interpretation,

                "missing_context":
                    missing_context,

                "strategic_implication":
                    strategic_implication,

                "future_development":
                    future_development,

                "rejection_reason":
                    rejection_reason,

                "additional_evidence":
                    " | ".join(
                        additional_evidence
                    ),

                "evidence_notes":
                    evidence_notes,

                "confidence":
                    confidence,

                "machine_total_articles":
                    metric["total_articles"],

                "machine_unique_sources":
                    metric["unique_sources"],

                "machine_growth_percent":
                    metric["growth_percent"],

                "timestamp":
                    datetime.now().isoformat()
            }

            save_review(
                review
            )

            if (
                current_index
                < len(topic_ids) - 1
            ):

                st.session_state.current_index += 1

                st.rerun()

            else:

                st.rerun()