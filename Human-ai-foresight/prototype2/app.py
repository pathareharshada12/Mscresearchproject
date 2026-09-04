
import streamlit as st
import pandas as pd
from evidence_qualifier import qualify_dataframe

st.set_page_config(page_title="Evidence Qualification | HITL Foresight", layout="wide")
st.title("Evidence Qualification Layer")
st.caption("Broad retrieval ≠ foresight evidence. Candidates must qualify before AI theme detection.")

uploaded = st.file_uploader("Upload scoped candidate corpus (.csv)", type="csv")
if uploaded is None:
    st.info("Upload the current scoped corpus to run the qualification gate.")
    st.stop()

raw = pd.read_csv(uploaded)
qdf = qualify_dataframe(raw)

a,b,c,d = st.columns(4)
a.metric("Retrieved candidates", len(qdf))
b.metric("Eligible now", int(qdf["eligible_for_detection"].sum()))
c.metric("Needs retrieval / verification", int((qdf["detection_status"]=="Retrieve / verify before detection").sum()))
d.metric("Excluded / context", int((qdf["detection_status"]!="Eligible for AI detection").sum()))

st.subheader("Why material is being blocked")
st.write(
    "The gate separates semantic relevance from evidential validity. "
    "Forecast articles, promotional/listicle material, market-size context and out-of-scope matches "
    "cannot silently become AI-detected trends."
)

status = st.multiselect(
    "Detection status",
    qdf["detection_status"].dropna().unique().tolist(),
    default=qdf["detection_status"].dropna().unique().tolist()
)
roles = st.multiselect(
    "Evidence role",
    qdf["evidence_role"].dropna().unique().tolist(),
    default=qdf["evidence_role"].dropna().unique().tolist()
)

cols = [c for c in [
    "id","title","source","date","semantic_relevance","scope_margin",
    "evidence_role","evidence_quality_score","source_body_verified",
    "detection_status","qualification_reason","url"
] if c in qdf.columns]

st.dataframe(
    qdf[qdf["detection_status"].isin(status) & qdf["evidence_role"].isin(roles)][cols],
    use_container_width=True, hide_index=True
)

st.download_button(
    "Download qualified corpus",
    qdf.to_csv(index=False).encode("utf-8"),
    "qualified_evidence_corpus.csv",
    "text/csv"
)

eligible = qdf[qdf["eligible_for_detection"]]
st.subheader("AI detection input")
if len(eligible):
    st.success(f"{len(eligible)} rows are currently permitted to enter clustering/theme detection.")
    st.dataframe(eligible[cols], use_container_width=True, hide_index=True)
else:
    st.warning(
        "0 rows currently qualify for AI detection because the corpus does not contain substantive source-body evidence. "
        "This is an intentional methodological safeguard. The next pipeline stage must retrieve/verify the underlying source content."
    )

st.divider()
st.markdown("""
### Next stage
`candidate retrieval → evidence qualification → source retrieval/verification → re-qualification → embeddings/clustering → AI theme interpretation`

The LLM is therefore **downstream of evidence validation**, not responsible for deciding whether a headline is evidence.
""")
