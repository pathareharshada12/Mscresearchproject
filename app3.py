import streamlit as st
import pandas as pd
import feedparser
import requests
import re

from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import quote_plus
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


# Page setup

st.set_page_config(
    page_title="Human–AI Foresight Lab",
    page_icon="◈",
    layout="wide"
)


# Project paths

APP_DIR = Path(__file__).resolve().parent


def find_file(relative_path):

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


# Styling

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    .hero {
        padding: 28px;
        border-radius: 20px;
        background: linear-gradient(135deg, #17131d, #22202a);
        border: 1px solid rgba(255,255,255,0.10);
        margin-bottom: 22px;
    }

    .signal-box {
        padding: 18px;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.025);
        margin-bottom: 14px;
    }

    .micro {
        display:inline-block;
        padding:5px 10px;
        border-radius:999px;
        background:rgba(189,123,255,0.15);
        border:1px solid rgba(189,123,255,0.35);
        font-size:.78rem;
    }

    .macro {
        display:inline-block;
        padding:5px 10px;
        border-radius:999px;
        background:rgba(86,179,255,0.13);
        border:1px solid rgba(86,179,255,0.32);
        font-size:.78rem;
    }

    .genx {
        display:inline-block;
        padding:5px 10px;
        border-radius:999px;
        background:rgba(220,197,135,0.12);
        border:1px solid rgba(220,197,135,0.30);
        font-size:.78rem;
    }

    .note {
        opacity:.72;
        font-size:.86rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Search themes
# These define the consumer context, not the final trends.

DEFAULT_QUERIES = [
    "Gen X fitness UK",
    "over 40 fitness UK",
    "over 50 running UK",
    "midlife fitness UK",
    "menopause fitness activewear UK",
    "strength training over 40 UK",
    "recovery fitness UK",
    "walking hiking over 50 UK",
    "sportswear technology UK",
    "wearable fitness over 40 UK"
]


GENX_TERMS = [
    "gen x",
    "generation x",
    "over 40",
    "over 50",
    "midlife",
    "middle aged",
    "menopause",
    "perimenopause",
    "masters",
    "longevity",
    "recovery",
    "joint",
    "mobility",
    "strength",
    "walking",
    "hiking"
]


TECH_TERMS = [
    "ai",
    "artificial intelligence",
    "wearable",
    "smart",
    "technology",
    "digital",
    "app",
    "tracker",
    "sensor",
    "personalisation",
    "personalization"
]


BEHAVIOUR_TERMS = [
    "consumer",
    "community",
    "habit",
    "behaviour",
    "behavior",
    "fitness",
    "training",
    "running",
    "walking",
    "wellness",
    "recovery",
    "gym",
    "activewear",
    "sportswear"
]


# Helpers

def clean_html(text):

    if not isinstance(text, str):
        return ""

    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


def term_score(text, terms):

    text = str(text).lower()

    hits = sum(
        1
        for term in terms
        if term in text
    )

    return hits


def genx_score(text):

    hits = term_score(
        text,
        GENX_TERMS
    )

    return min(
        5,
        hits
    )


def focus_label(text):

    tech = term_score(
        text,
        TECH_TERMS
    )

    behaviour = term_score(
        text,
        BEHAVIOUR_TERMS
    )

    if tech > behaviour:
        return "Technology"

    if behaviour > tech:
        return "Consumer behaviour"

    return "Technology + consumer behaviour"


def recency_score(date):

    if pd.isna(date):
        return 1

    now = pd.Timestamp.now(
        tz="UTC"
    )

    date = pd.Timestamp(date)

    if date.tzinfo is None:
        date = date.tz_localize("UTC")

    days = (
        now - date
    ).days

    if days <= 7:
        return 5

    if days <= 30:
        return 4

    if days <= 90:
        return 3

    if days <= 180:
        return 2

    return 1


# Live macro source
# Google News RSS is used as broad public market/media context.

@st.cache_data(ttl=1800)
def fetch_google_news(query, limit=15):

    url = (
        "https://news.google.com/rss/search?q="
        + quote_plus(query)
        + "&hl=en-GB&gl=GB&ceid=GB:en"
    )

    feed = feedparser.parse(
        url
    )

    rows = []

    for entry in feed.entries[:limit]:

        published = pd.to_datetime(
            entry.get(
                "published",
                ""
            ),
            errors="coerce",
            utc=True
        )

        rows.append(
            {
                "source_type": "Macro",
                "platform": "Google News",
                "query": query,
                "title": clean_html(
                    entry.get(
                        "title",
                        ""
                    )
                ),
                "text": clean_html(
                    entry.get(
                        "summary",
                        ""
                    )
                ),
                "url": entry.get(
                    "link",
                    ""
                ),
                "date": published
            }
        )

    return rows


# Live micro source
# Reddit RSS is used only for public community discussion.
# If Reddit blocks the request, the app continues with macro evidence.

@st.cache_data(ttl=1800)
def fetch_reddit(query, limit=15):

    url = (
        "https://www.reddit.com/search.rss?q="
        + quote_plus(query)
        + "&sort=new&t=year"
    )

    headers = {
        "User-Agent":
            "UAL-MSc-Human-AI-Foresight-Research/1.0"
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=10
        )

        if response.status_code != 200:
            return []

        feed = feedparser.parse(
            response.text
        )

    except Exception:
        return []

    rows = []

    for entry in feed.entries[:limit]:

        published = pd.to_datetime(
            entry.get(
                "updated",
                entry.get(
                    "published",
                    ""
                )
            ),
            errors="coerce",
            utc=True
        )

        rows.append(
            {
                "source_type": "Micro",
                "platform": "Reddit",
                "query": query,
                "title": clean_html(
                    entry.get(
                        "title",
                        ""
                    )
                ),
                "text": clean_html(
                    entry.get(
                        "summary",
                        ""
                    )
                ),
                "url": entry.get(
                    "link",
                    ""
                ),
                "date": published
            }
        )

    return rows


@st.cache_data(ttl=1800)
def run_live_scan(
    queries_tuple,
    items_per_query
):

    rows = []

    for query in queries_tuple:

        rows.extend(
            fetch_google_news(
                query,
                items_per_query
            )
        )

        rows.extend(
            fetch_reddit(
                query,
                items_per_query
            )
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows
    )

    df = df.drop_duplicates(
        subset=[
            "title",
            "url"
        ]
    ).reset_index(
        drop=True
    )

    df["analysis_text"] = (
        df["title"].fillna("")
        + " "
        + df["text"].fillna("")
    )

    df["genx_relevance"] = (
        df["analysis_text"]
        .apply(
            genx_score
        )
    )

    df["focus"] = (
        df["analysis_text"]
        .apply(
            focus_label
        )
    )

    df["recency"] = (
        df["date"]
        .apply(
            recency_score
        )
    )

    return df


def cluster_signals(df, n_clusters=5):

    if len(df) < 4:
        return df, {}

    number = min(
        n_clusters,
        max(
            2,
            len(df) // 4
        )
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=1200,
        ngram_range=(1, 2)
    )

    matrix = vectorizer.fit_transform(
        df["analysis_text"]
        .fillna("")
    )

    model = KMeans(
        n_clusters=number,
        random_state=42,
        n_init=10
    )

    df = df.copy()

    df["live_cluster"] = model.fit_predict(
        matrix
    )

    terms = vectorizer.get_feature_names_out()

    labels = {}

    for cluster_id in sorted(
        df["live_cluster"]
        .unique()
    ):

        centre = model.cluster_centers_[
            cluster_id
        ]

        top_indices = centre.argsort()[
            -5:
        ][::-1]

        top_terms = [
            terms[index]
            for index in top_indices
        ]

        labels[
            cluster_id
        ] = " · ".join(
            top_terms[:4]
        )

    return df, labels


def signal_summary(
    group,
    label
):

    micro_count = int(
        (
            group["source_type"]
            == "Micro"
        ).sum()
    )

    macro_count = int(
        (
            group["source_type"]
            == "Macro"
        ).sum()
    )

    if micro_count > macro_count:
        level = "Micro"

    elif macro_count > micro_count:
        level = "Macro"

    else:
        level = "Mixed"

    avg_genx = round(
        group[
            "genx_relevance"
        ].mean(),
        1
    )

    avg_recency = round(
        group[
            "recency"
        ].mean(),
        1
    )

    platforms = (
        group["platform"]
        .nunique()
    )

    return {
        "label": label,
        "level": level,
        "items": len(group),
        "micro": micro_count,
        "macro": macro_count,
        "genx": avg_genx,
        "recency": avg_recency,
        "platforms": platforms
    }


# Header

st.markdown(
    """
    <div class="hero">
        <div style="text-transform:uppercase; letter-spacing:.13em;
                    opacity:.65; font-size:.75rem;">
            Human–AI Foresight · Experimental Signal Lab
        </div>
        <h1 style="margin-bottom:6px;">
            Gen X × UK Sportswear
        </h1>
        <p style="font-size:1.05rem; max-width:900px;">
            A lightweight live horizon-scanning layer that separates
            community-level <b>micro signals</b> from broader
            market/media <b>macro signals</b>, before professional interpretation.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


tab1, tab2, tab3 = st.tabs(
    [
        "Live Signal Lab",
        "Signal Deep Dive",
        "Method"
    ]
)


# Live scan

with tab1:

    st.subheader(
        "Live horizon scan"
    )

    st.write(
        """
        This scan looks for recent public evidence around
        **Gen X, technology, consumer behaviour and UK sportswear**.

        **Micro** = public community discussion.

        **Macro** = broader news and market/media discussion.

        The labels describe the **level of evidence**, not whether
        something is automatically a trend.
        """
    )

    with st.expander(
        "Search areas",
        expanded=False
    ):

        query_text = st.text_area(
            "One search area per line",
            value="\n".join(
                DEFAULT_QUERIES
            ),
            height=220
        )

        items_per_query = st.slider(
            "Items per source / search area",
            min_value=5,
            max_value=20,
            value=10
        )

    queries = tuple(
        [
            q.strip()
            for q in query_text.splitlines()
            if q.strip()
        ]
    )

    if st.button(
        "Run live signal scan",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "Scanning public evidence..."
        ):

            live_df = run_live_scan(
                queries,
                items_per_query
            )

            if live_df.empty:

                st.error(
                    "No live evidence was returned."
                )

            else:

                clustered_df, labels = (
                    cluster_signals(
                        live_df,
                        n_clusters=5
                    )
                )

                st.session_state[
                    "live_df"
                ] = clustered_df

                st.session_state[
                    "live_labels"
                ] = labels


    live_df = st.session_state.get(
        "live_df",
        pd.DataFrame()
    )

    labels = st.session_state.get(
        "live_labels",
        {}
    )


    if not live_df.empty:

        st.divider()

        c1, c2, c3, c4 = st.columns(
            4
        )

        c1.metric(
            "Live evidence",
            len(
                live_df
            )
        )

        c2.metric(
            "Micro",
            int(
                (
                    live_df[
                        "source_type"
                    ]
                    == "Micro"
                ).sum()
            )
        )

        c3.metric(
            "Macro",
            int(
                (
                    live_df[
                        "source_type"
                    ]
                    == "Macro"
                ).sum()
            )
        )

        c4.metric(
            "Signal groups",
            live_df[
                "live_cluster"
            ].nunique()
        )

        st.subheader(
            "Live signal groups"
        )

        summaries = []

        for cluster_id in sorted(
            live_df[
                "live_cluster"
            ].unique()
        ):

            group = live_df[
                live_df[
                    "live_cluster"
                ]
                == cluster_id
            ]

            summary = signal_summary(
                group,
                labels.get(
                    cluster_id,
                    f"Cluster {cluster_id}"
                )
            )

            summary[
                "cluster_id"
            ] = cluster_id

            summaries.append(
                summary
            )

        summary_df = pd.DataFrame(
            summaries
        )

        for _, row in summary_df.iterrows():

            cluster_id = int(
                row[
                    "cluster_id"
                ]
            )

            group = live_df[
                live_df[
                    "live_cluster"
                ]
                == cluster_id
            ].copy()

            st.markdown(
                '<div class="signal-box">',
                unsafe_allow_html=True
            )

            left, right = st.columns(
                [3, 1]
            )

            with left:

                st.markdown(
                    f"### {row['label']}"
                )

                if row[
                    "level"
                ] == "Micro":

                    st.markdown(
                        '<span class="micro">MICRO</span>',
                        unsafe_allow_html=True
                    )

                elif row[
                    "level"
                ] == "Macro":

                    st.markdown(
                        '<span class="macro">MACRO</span>',
                        unsafe_allow_html=True
                    )

                else:

                    st.markdown(
                        '<span class="micro">MICRO</span> '
                        '<span class="macro">MACRO</span>',
                        unsafe_allow_html=True
                    )

                st.markdown(
                    '<span class="genx">GEN X LENS</span>',
                    unsafe_allow_html=True
                )

            with right:

                st.metric(
                    "Gen X relevance",
                    f"{row['genx']}/5"
                )

            a, b, c = st.columns(
                3
            )

            a.metric(
                "Evidence",
                int(
                    row[
                        "items"
                    ]
                )
            )

            b.metric(
                "Micro / Macro",
                f"{int(row['micro'])} / {int(row['macro'])}"
            )

            c.metric(
                "Recency",
                f"{row['recency']}/5"
            )

            top_items = group.sort_values(
                [
                    "genx_relevance",
                    "recency"
                ],
                ascending=False
            ).head(
                3
            )

            st.caption(
                "Recent evidence examples"
            )

            for _, item in top_items.iterrows():

                badge = (
                    "MICRO"
                    if item[
                        "source_type"
                    ] == "Micro"
                    else "MACRO"
                )

                st.markdown(
                    f"**[{badge}] {item['title']}**"
                )

                st.caption(
                    f"{item['platform']} · "
                    f"{item['focus']} · "
                    f"Gen X relevance "
                    f"{item['genx_relevance']}/5"
                )

                if item[
                    "url"
                ]:

                    st.markdown(
                        f"[Open source ↗]({item['url']})"
                    )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

        st.download_button(
            "Download live evidence CSV",
            data=live_df.to_csv(
                index=False
            ).encode(
                "utf-8"
            ),
            file_name=(
                "genx_live_signal_scan_"
                + datetime.now().strftime(
                    "%Y%m%d_%H%M"
                )
                + ".csv"
            ),
            mime="text/csv"
        )


# Deep dive

with tab2:

    live_df = st.session_state.get(
        "live_df",
        pd.DataFrame()
    )

    labels = st.session_state.get(
        "live_labels",
        {}
    )

    if live_df.empty:

        st.info(
            """
            Run the Live Signal Lab first.
            The signal groups will then appear here for deeper review.
            """
        )

    else:

        cluster_options = sorted(
            live_df[
                "live_cluster"
            ].unique()
        )

        selected = st.selectbox(
            "Choose a signal group",
            cluster_options,
            format_func=lambda x:
                labels.get(
                    x,
                    f"Cluster {x}"
                )
        )

        group = live_df[
            live_df[
                "live_cluster"
            ]
            == selected
        ].copy()

        summary = signal_summary(
            group,
            labels.get(
                selected,
                f"Cluster {selected}"
            )
        )

        st.header(
            summary[
                "label"
            ]
        )

        col1, col2, col3, col4 = (
            st.columns(4)
        )

        col1.metric(
            "Evidence",
            summary[
                "items"
            ]
        )

        col2.metric(
            "Micro",
            summary[
                "micro"
            ]
        )

        col3.metric(
            "Macro",
            summary[
                "macro"
            ]
        )

        col4.metric(
            "Gen X relevance",
            f"{summary['genx']}/5"
        )

        st.subheader(
            "Evidence mix"
        )

        for _, item in group.sort_values(
            "date",
            ascending=False
        ).iterrows():

            with st.expander(
                item[
                    "title"
                ]
            ):

                st.write(
                    f"**Level:** "
                    f"{item['source_type']}"
                )

                st.write(
                    f"**Source:** "
                    f"{item['platform']}"
                )

                st.write(
                    f"**Focus:** "
                    f"{item['focus']}"
                )

                st.write(
                    f"**Gen X relevance:** "
                    f"{item['genx_relevance']}/5"
                )

                if pd.notna(
                    item[
                        "date"
                    ]
                ):

                    st.write(
                        "**Date:** "
                        + pd.Timestamp(
                            item[
                                "date"
                            ]
                        ).strftime(
                            "%d %b %Y"
                        )
                    )

                if item[
                    "text"
                ]:

                    st.write(
                        item[
                            "text"
                        ][:800]
                    )

                if item[
                    "url"
                ]:

                    st.markdown(
                        f"[Open original source ↗]"
                        f"({item['url']})"
                    )

        st.divider()

        st.subheader(
            "Professional interpretation"
        )

        st.radio(
            "What does this evidence represent?",
            [
                "Possible emerging signal",
                "Established behaviour",
                "Macro context only",
                "Weak / fragmented evidence",
                "Noise / unrelated",
                "Need more evidence"
            ],
            index=None,
            key=f"live_class_{selected}"
        )

        st.radio(
            "Which level is most useful for interpreting it?",
            [
                "Micro",
                "Macro",
                "Both"
            ],
            index=None,
            horizontal=True,
            key=f"live_level_{selected}"
        )

        st.slider(
            "Relevance to Gen X",
            1,
            5,
            3,
            key=f"live_genx_{selected}"
        )

        st.text_area(
            "What is changing?",
            height=110,
            key=f"live_change_{selected}"
        )

        st.text_area(
            "Why could this matter for UK sportswear?",
            height=110,
            key=f"live_implication_{selected}"
        )

        st.text_area(
            "What evidence would you look for next?",
            height=100,
            key=f"live_next_{selected}"
        )


# Method

with tab3:

    st.header(
        "Method"
    )

    st.write(
        """
        This experimental layer keeps the method deliberately simple.

        **Micro signals** are evidence from public community discussion.

        **Macro signals** are evidence from broader news and market/media
        discussion.

        The system does not automatically claim that either is a trend.
        It groups similar evidence and gives the professional forecaster
        a structured way to interpret it.
        """
    )

    st.subheader(
        "Technical process"
    )

    st.markdown(
        """
        1. Search predefined Gen X / sportswear / technology / consumer
           behaviour areas.
        2. Collect recent public evidence.
        3. Label the evidence by source level: **Micro** or **Macro**.
        4. Convert the text into TF-IDF features.
        5. Use K-Means to group similar live evidence.
        6. Calculate simple recency and Gen X relevance indicators.
        7. Present the evidence to the human forecaster for interpretation.
        """
    )

    st.info(
        """
        **Important methodological boundary**

        Gen X relevance is a keyword-based retrieval aid, not demographic
        identification. The system does not infer the age or identity of
        individual users.
        """
    )

    st.caption(
        """
        Reddit access may vary because the platform can restrict unauthenticated
        requests. If Reddit returns no data, the app continues to work with the
        macro evidence rather than failing.
        """
    )
