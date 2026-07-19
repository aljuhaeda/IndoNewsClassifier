"""Streamlit demo: paste an Indonesian news title, get a predicted category."""

import altair as alt
import pandas as pd
import streamlit as st

from src.inference import predict_baseline
from src.training import load_baseline

st.set_page_config(page_title="IndoNewsClassifier", page_icon="📰")

MODEL_PATH = "models/tfidf_logreg.joblib"


@st.cache_resource
def get_model():
    return load_baseline(MODEL_PATH)


st.title("📰 IndoNewsClassifier")
st.caption("TF-IDF + Logistic Regression baseline for Indonesian news title classification.")

title = st.text_area("Paste an Indonesian news title", height=100)

if st.button("Classify", type="primary") and title.strip():
    model = get_model()
    label, scores = predict_baseline(model, title)

    st.subheader(f"Predicted category: `{label}`")

    scores_df = pd.DataFrame(sorted(scores.items(), key=lambda kv: kv[1], reverse=True), columns=["category", "confidence"])
    chart = (
        alt.Chart(scores_df)
        .mark_bar()
        .encode(x="confidence:Q", y=alt.Y("category:N", sort="-x"), tooltip=["category", "confidence"])
    )
    st.altair_chart(chart, use_container_width=True)
