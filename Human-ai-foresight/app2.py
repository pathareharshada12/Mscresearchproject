import streamlit as st
import pandas as pd

from pathlib import Path
from datetime import datetime


# Page setup

st.set_page_config(
    page_title="Human–AI Foresight",
    page_icon="◈",
    layout="wide"
)


# File paths

APP_DIR = Path(__file__).resolve().parent


def find_file(relative_path):
    """
    Looks for project files relative to app2.py
    and one folder above it.
    """

    possible_paths = [
        APP_DIR / relative_path,
        APP_DIR.parent / relative_path
    ]

    for path in possible_paths:

        if path.exists():
            return path

    return possible_paths[0]


ARTICLES_FILE = find_file(
    Path("data/processed/articles_with_topics.csv")
)

METRICS_FILE = find_file(
    Path("data/processed/signal_metrics.csv")
)

TOPIC_SUMMARY_FILE = find_file(
    Path("data/processed/topic_summary.csv")
)

EVIDENCE_FILE = find_file(
    Path("data/processed/cluster_evidence_digest.csv")
)


REVIEWS_FOLDER = (
    APP_DIR
    / "data"
    / "human_reviews"
)

REVIEWS_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# Check files

required_files = {
    "Articles": ARTICLES_FILE,
    "Signal metrics": METRICS_FILE,
    "Topic summary": TOPIC_SUMMARY_FILE,
    "Evidence digest": EVIDENCE_FILE
}


missing_files = [
    f"{name}: {path}"
    for name, path in required_files.items()
    if not path.exists()
]


if missing_files:

    st.error(
        "Some required data files could not be found."
    )

    st.write(
        """
        Please check that the following files exist
        in your GitHub repository:
        """
    )

    for file in missing_files:
        st.code(file)

    st.stop()


# Load data

@st.cache_data
def load_data():

    articles = pd.read_csv(
        ARTICLES_FILE
    )

    metrics = pd.read_csv(
        METRICS_FILE
    )

    topic_summary = pd.read_csv(
        TOPIC_SUMMARY_FILE
    )

    evidence = pd.read_csv(
        EVIDENCE_FILE
    )


    articles["date"] = pd.to_datetime(
        articles["date"],
        errors="coerce"
    )


    evidence["date"] = pd.to_datetime(
        evidence["date"],
        errors="coerce"
    )


    return (
        articles,
        metrics,
        topic_summary,
        evidence
    )


(
    articles,
    metrics,
    topic_summary,
    evidence
) = load_data()


# Session state

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


# Topic IDs

topic_ids = sorted(

    metrics[
        "topic_id"
    ]

    .dropna()

    .astype(int)

    .tolist()
)


# Functions

def participant_file():

    participant_id = (
        st.session_state
        .participant_id
        .strip()
    )

    return (
        REVIEWS_FOLDER
        /
        f"{participant_id}_reviews.csv"
    )


def get_saved_reviews():

    file = participant_file()

    if file.exists():

        return pd.read_csv(
            file
        )

    return pd.DataFrame()


def save_review(review):

    file = participant_file()

    new_review = pd.DataFrame(
        [review]
    )


    if file.exists():

        existing = pd.read_csv(
            file
        )

        existing = existing[
            existing["topic_id"]
            != review["topic_id"]
        ]

        new_review = pd.concat(
            [
                existing,
                new_review
            ],
            ignore_index=True
        )


    new_review.to_csv(

        file,

        index=False,

        encoding="utf-8-sig"
    )


def get_topic(topic_id):

    metric_match = metrics[
        metrics["topic_id"]
        == topic_id
    ]


    if metric_match.empty:

        st.error(
            f"No signal metrics found for topic {topic_id}."
        )

        st.stop()


    metric = metric_match.iloc[0]


    topic_articles = articles[
        articles["topic_id"]
        == topic_id
    ].copy()


    topic_articles = (
        topic_articles
        .sort_values(
            "date",
            ascending=False
        )
    )


    summary_match = topic_summary[
        topic_summary["Topic"]
        == topic_id
    ]


    summary = (

        summary_match.iloc[0]

        if not summary_match.empty

        else None
    )


    topic_evidence = evidence[
        evidence["topic_id"]
        == topic_id
    ].copy()


    topic_evidence = (
        topic_evidence
        .sort_values(
            "evidence_number"
        )
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


    if not isinstance(
        value,
        str
    ):

        return []


    value = (
        value
        .replace("[", "")
        .replace("]", "")
        .replace("'", "")
    )


    stop_words = {

        "the",
        "and",
        "of",
        "in",
        "to",
        "is",
        "how",
        "by",
        "for",
        "with",
        "on",
        "are"
    }


    terms = [

        term.strip()

        for term in value.split(",")

        if term.strip()
    ]


    useful_terms = [

        term

        for term in terms

        if term.lower()
        not in stop_words
    ]


    return useful_terms[:6]



# App introduction

if not st.session_state.started:

    st.title(
        "Human–AI Foresight"
    )


    st.subheader(
        "UK Sportswear · Technology & Consumer Behaviour"
    )


    st.write(
        """
        This system analysed a horizon-scanning dataset
        and identified **7 possible signal areas**.

        These are not confirmed trends.

        Your role is to review the evidence and decide
        which patterns are meaningful enough to contribute
        to a future forecast.
        """
    )


    st.info(
        """
        **Your role as the professional forecaster**

        The computational system identifies patterns,
        similarities and changes in evidence.

        You contribute professional judgement including
        context, novelty, relevance, interpretation and
        strategic meaning.
        """
    )


    col1, col2 = st.columns(
        2
    )


    with col1:

        participant_id = st.text_input(

            "Participant ID",

            placeholder="e.g. P01"
        )


        role = st.text_input(

            "Professional role",

            placeholder=(
                "e.g. Trend Forecaster, "
                "Foresight Strategist"
            )
        )


    with col2:

        experience = st.number_input(

            "Years of relevant experience",

            min_value=0,

            max_value=50,

            value=0
        )


    with st.expander(
        "How the computational system works"
    ):

        st.write(
            """
            Articles were represented using Sentence-BERT
            embeddings and grouped using BERTopic.

            The system then calculated evidence indicators
            including:

            - article volume
            - source diversity
            - time spread
            - recent changes in coverage

            Representative evidence was then selected for
            professional review.

            These computational patterns have not been
            validated as trends.
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


# Saved responses

saved = get_saved_reviews()


# Final report

if (

    not saved.empty

    and saved[
        "topic_id"
    ].nunique()

    >= len(topic_ids)

):

    st.title(
        "Forecast review complete"
    )


    included = saved[
        saved["signal_decision"]
        == "Include in forecast"
    ]


    excluded = saved[
        saved["signal_decision"]
        == "Exclude from forecast"
    ]


    more_evidence = saved[
        saved["signal_decision"]
        == "Need more evidence"
    ]


    col1, col2, col3, col4 = (
        st.columns(4)
    )


    col1.metric(
        "Reviewed",
        len(saved)
    )


    col2.metric(
        "Included",
        len(included)
    )


    col3.metric(
        "Excluded",
        len(excluded)
    )


    col4.metric(
        "Need more evidence",
        len(more_evidence)
    )


    st.divider()


    st.header(
        "Your Human–AI Forecast"
    )


    st.write(
        """
        The following forecast contains the signals
        retained through professional review.
        """
    )


    if included.empty:

        st.warning(
            """
            No machine-generated signal was included
            in the final forecast.
            """
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


            if (
                pd.isna(name)
                or not str(name).strip()
            ):

                name = (
                    f"Signal {number}"
                )


            st.header(
                f"{number:02d} — {name}"
            )


            st.markdown(
                "**Professional interpretation**"
            )


            st.write(
                row.get(
                    "interpretation",
                    ""
                )
            )


            st.markdown(
                "**Why it matters for UK sportswear**"
            )


            st.write(
                row.get(
                    "strategic_implication",
                    ""
                )
            )


            st.markdown(
                "**What the computational analysis missed**"
            )


            st.write(
                row.get(
                    "missing_context",
                    ""
                )
            )


            st.markdown(
                "**Future implication**"
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


    # ----------------------------------------------
    # Missing signals
    # ----------------------------------------------

    st.divider()


    st.header(
        "Did the system miss anything?"
    )


    missing_signal = st.radio(

        """
        Did the computational system fail to surface
        an important signal?
        """,

        [
            "No",
            "Yes"
        ],

        horizontal=True,

        key="missing_signal"
    )


    if missing_signal == "Yes":

        st.text_input(
            "Missing signal name",
            key="missing_signal_name"
        )


        st.text_area(
            "Describe the missing signal",
            key="missing_signal_description"
        )


        st.text_area(
            "Why do you think the system missed it?",
            key="missing_signal_reason"
        )


    # ----------------------------------------------
    # Final restructuring reflection
    # ----------------------------------------------

    st.divider()


    st.header(
        "Final reflection"
    )


    restructure = st.radio(

        """
        Looking across the machine-generated signals,
        would you merge, split or substantially reframe
        any of them?
        """,

        [
            "No",
            "Yes"
        ],

        horizontal=True,

        key="restructure_final"
    )


    if restructure == "Yes":

        st.text_area(

            """
            Please explain which signals you would change
            and how you would restructure them.
            """,

            key="restructure_final_notes"
        )


    # ----------------------------------------------
    # Download
    # ----------------------------------------------

    st.divider()


    st.download_button(

        "Download review data",

        data=saved.to_csv(
            index=False
        ).encode(
            "utf-8"
        ),

        file_name=(

            f"{st.session_state.participant_id}"
            "_reviews.csv"
        ),

        mime="text/csv"
    )


    st.stop()


# Signal dashboard

st.title(
    "Signal Dashboard"
)


st.caption(
    """
    Review each computationally identified pattern
    before deciding whether it belongs in the forecast.
    """
)


dashboard_columns = st.columns(
    min(
        len(topic_ids),
        4
    )
)


for index, topic_id in enumerate(
    topic_ids
):

    (
        metric,
        topic_articles,
        summary,
        topic_evidence
    ) = get_topic(
        topic_id
    )


    growth = metric[
        "growth_percent"
    ]


    if pd.isna(growth):

        growth_text = "N/A"

    else:

        growth_text = (
            f"{growth:+.0f}%"
        )


    column = dashboard_columns[
        index
        % len(dashboard_columns)
    ]


    with column:

        st.markdown(
            f"### Signal {index + 1}"
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


st.divider()


# Current signal

current_index = (
    st.session_state
    .current_index
)


topic_id = topic_ids[
    current_index
]


(
    metric,
    topic_articles,
    summary,
    topic_evidence
) = get_topic(
    topic_id
)


terms = clean_terms(
    summary
)


progress = (

    current_index + 1

) / len(topic_ids)


st.progress(
    progress
)


st.caption(

    f"Possible signal "
    f"{current_index + 1} "
    f"of {len(topic_ids)}"

)


# Signal evidence

st.title(
    f"Possible Signal {current_index + 1}"
)


if terms:

    st.write(

        "**Machine-detected themes:** "

        +
        " · ".join(
            terms
        )
    )


# Evidence metrics

st.subheader(
    "Evidence snapshot"
)


col1, col2, col3, col4 = (
    st.columns(4)
)


col1.metric(

    "Articles",

    int(
        metric[
            "total_articles"
        ]
    )
)


col2.metric(

    "Sources",

    int(
        metric[
            "unique_sources"
        ]
    )
)


col3.metric(

    "Active months",

    int(
        metric[
            "active_months"
        ]
    )
)


growth = metric[
    "growth_percent"
]


growth_text = (

    "N/A"

    if pd.isna(growth)

    else f"{growth:+.0f}%"
)


col4.metric(
    "Recent coverage",
    growth_text
)


# Representative articles

st.subheader(
    "Representative evidence"
)


st.caption(
    """
    Five representative items were selected from
    this grouping using semantic similarity.
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

        date_text = (
            "Unknown date"
        )


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



# All articles

with st.expander(
    f"Browse all {len(topic_articles)} items"
):

    for _, row in topic_articles.iterrows():

        st.markdown(
            f"**{row['title']}**"
        )


        st.caption(
            str(
                row.get(
                    "source",
                    ""
                )
            )
        )


        st.divider()


# Human review

st.divider()


st.header(
    "Professional judgement"
)


st.write(
    """
    The computational system has surfaced this pattern.

    Your role is to determine whether it is meaningful
    enough to contribute to the forecast.
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


# Include signal

if (
    signal_decision
    == "Include in forecast"
):

    st.success(
        """
        This signal will be developed into
        the final Human–AI forecast.
        """
    )


    human_signal_name = st.text_input(

        "Signal name",

        placeholder=(
            "How would you frame this professionally?"
        ),

        key=f"name_{topic_id}"
    )


    interpretation = st.text_area(

        """
        Your interpretation: In your professional
        judgement, what change does this evidence represent?
        """,

        height=120,

        key=f"interpretation_{topic_id}"
    )


    missing_context = st.text_area(

        """
        What does your professional expertise add
        that the computational analysis cannot determine?
        """,

        placeholder=(
            "For example: cultural context, consumer motivation, "
            "commercial knowledge, historical context, "
            "competitor knowledge or industry experience."
        ),

        height=110,

        key=f"context_{topic_id}"
    )


    strategic_implication = st.text_area(

        """
        Why does this matter strategically
        for UK sportswear?
        """,

        height=110,

        key=f"implication_{topic_id}"
    )


    future_development = st.text_area(

        """
        Future implication: Based on your expertise,
        how could this affect UK sportswear over
        the next 2–3 years?
        """,

        height=120,

        key=f"future_{topic_id}"
    )


# Exclude signal

elif (
    signal_decision
    == "Exclude from forecast"
):

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


# More evidence

elif (
    signal_decision
    == "Need more evidence"
):

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


# Confidence

confidence = st.slider(

    "Confidence in this judgement",

    min_value=1,

    max_value=5,

    value=3,

    key=f"confidence_{topic_id}"
)


# Navigation

st.divider()


back_col, next_col = (
    st.columns(
        [1, 3]
    )
)


with back_col:

    if current_index > 0:

        if st.button(
            "← Previous"
        ):

            st.session_state.current_index -= 1

            st.rerun()


with next_col:

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

            signal_decision
            == "Include in forecast"

            and not human_signal_name.strip()

        ):

            st.error(
                "Please give the signal a name."
            )


        elif (

            signal_decision
            == "Exclude from forecast"

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
                    metric[
                        "total_articles"
                    ],

                "machine_unique_sources":
                    metric[
                        "unique_sources"
                    ],

                "machine_growth_percent":
                    metric[
                        "growth_percent"
                    ],

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