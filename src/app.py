"""Streamlit demo.  Run:  streamlit run src/app.py"""
import json
import streamlit as st
from triage import Triage

st.set_page_config(page_title="Tanglish Message Triage", page_icon="💬")
st.title("Tanglish customer-message triage")
st.caption("SWYNEX AI Internship – Final Application. Predicts intent, or asks for human review when unsure.")


@st.cache_resource
def load():
    thr = json.load(open("data/threshold.json"))["threshold"]
    return Triage.from_csv("data/messages.csv", thr), thr


tri, thr = load()
st.sidebar.write(f"Confidence threshold: **{thr:.2f}**")
examples = ["anna ennoda parcel ippo enga irukku", "payment cut aachu order confirm aagala",
            "refund", "என் ஆர்டர் எங்கே", "12345", ""]
choice = st.selectbox("Try an example (or type your own below)", ["(custom)"] + examples)
text = st.text_area("Customer message", value="" if choice == "(custom)" else choice)
if st.button("Classify"):
    r = tri.predict(text)
    if r["status"] == "ok":
        st.success(f"Intent: **{r['label']}**  (confidence {r['confidence']:.2f})")
    elif r["status"] == "needs_human_review":
        st.warning(f"Not confident ({r['confidence']:.2f}). Sent to human review.")
    else:
        st.error(f"Could not process: {r['reason']}")
    if r.get("top2"):
        st.write("Top 2 guesses:", {k: round(v, 2) for k, v in r["top2"]})
