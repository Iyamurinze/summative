"""
Streamlit web app for the Hausa Sentiment Classification project.

Loads the fine-tuned AfriBERTa model (trained in notebooks/hausa_sentiment.ipynb,
pushed to the Hugging Face Hub as Iyamurinze/afriberta-hausa-sentiment — our
best-performing model, see README for the comparison against AfroXLMR and the
TF-IDF baselines) and serves it through a simple text-in / label-out UI.

Run locally:
    pip install -r app/requirements.txt
    streamlit run app/streamlit_app.py

Deploy on Streamlit Community Cloud (share.streamlit.io) — free, no card required:
    1. Push this repo to GitHub.
    2. Go to share.streamlit.io, sign in with GitHub, click "New app".
    3. Pick this repo/branch, set "Main file path" to app/streamlit_app.py, deploy.
"""

import os

import streamlit as st
from transformers import pipeline

MODEL_ID = os.environ.get("HF_MODEL_ID", "Iyamurinze/afriberta-hausa-sentiment")

# Real tweets from the AfriSenti-Hausa test set (with known gold labels), chosen to show
# both correct predictions and a genuine failure mode — not hand-written sentences, since
# the model is trained only on real tweet style and misreads clean formal Hausa (see README
# Limitations). Verified against the deployed model before inclusion here.
EXAMPLES = [
    "allah ya gafartawa mallam allah ya saada shi da annabin rahama",  # gold: positive
    "wai na dawo daga makkah xani madina najeria kasarmu ta gado",  # gold: negative
    "allah sarki talaka ba wan allah allah kashuge mn gaba",  # gold: positive, model says neutral — known failure mode
]


@st.cache_resource
def load_classifier():
    return pipeline("text-classification", model=MODEL_ID, top_k=None)


st.set_page_config(page_title="Hausa Sentiment Classifier", page_icon="🇳🇬")
st.title("Hausa Sentiment Classifier")
st.caption(
    "Fine-tuned AfriBERTa model for sentiment classification (positive / negative / "
    "neutral) of Hausa text, trained on the AfriSenti dataset (Muhammad et al., 2023). "
    "This is an NLP summative project — not a general-purpose chatbot."
)

classifier = load_classifier()

example_choice = st.selectbox(
    "Try an example, or type your own text below",
    ["(type your own)"] + EXAMPLES,
)
default_text = "" if example_choice == "(type your own)" else example_choice

text = st.text_area(
    "Hausa text",
    value=default_text,
    placeholder="Rubuta jimla a Hausa anan... (Type a Hausa sentence here...)",
    height=100,
)

if st.button("Classify", type="primary") and text.strip():
    scores = classifier(text)[0]
    scores_sorted = sorted(scores, key=lambda x: -x["score"])
    top = scores_sorted[0]
    st.subheader(f"Prediction: **{top['label']}** ({top['score']:.1%} confidence)")
    for item in scores_sorted:
        st.progress(item["score"], text=f"{item['label']}: {item['score']:.1%}")
elif text.strip() == "":
    st.info("Enter some Hausa text above, or pick an example, then click Classify.")
