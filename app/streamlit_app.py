"""
Streamlit web app for the Kinyarwanda Sentiment Classification project.

Loads the fine-tuned model (trained in notebooks/kinyarwanda_sentiment.ipynb, pushed
to the Hugging Face Hub — see README for the comparison between AfroXLMR, AfriBERTa,
and the TF-IDF baselines) and serves it through a simple text-in / label-out UI.

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

# AfroXLMR beat AfriBERTa on the real test set (68.3% vs 63.9% macro-F1) — see README
# Results for the full comparison and why (clean Kinyarwanda pretraining vs AfriBERTa's
# mixed Kinyarwanda/Kirundi "Gahuza" exposure).
MODEL_ID = os.environ.get("HF_MODEL_ID", "Iyamurinze/afroxlmr-kinyarwanda-sentiment")

# Real tweets from the AfriSenti-Kinyarwanda test set (with known gold labels), verified
# against this exact deployed model's actual predictions — not hand-written sentences,
# since the model is trained only on real tweet style. One correct example per class, plus
# a genuine misclassification to honestly demonstrate a limitation.
EXAMPLES = [
    "Mwese muzagire Umwaka mushya muhire wa #2022",  # gold: positive, pred: positive (94%)
    "Ibintu 3 byo kureba: Intumbero, Imyitwarire, Ururimi.",  # gold: negative, pred: negative (77%)
    "Ndumva nta bwenge mfite today ndaza kwiba ibitweet byanyuu",  # gold: neutral, pred: neutral (81%)
    "Ngo iyo Imana iza kwica satani, satani yari kuba yitabye Imana #fact",  # gold: negative, pred: neutral (68%) — known failure mode
]


@st.cache_resource
def load_classifier():
    return pipeline("text-classification", model=MODEL_ID, top_k=None)


st.set_page_config(page_title="Kinyarwanda Sentiment Classifier", page_icon="🇷🇼")
st.title("Kinyarwanda Sentiment Classifier")
st.caption(
    "Fine-tuned Transformer model for sentiment classification (positive / negative / "
    "neutral) of Kinyarwanda text, trained on the AfriSenti dataset (Muhammad et al., 2023). "
    "This is an NLP summative project — not a general-purpose chatbot."
)

classifier = load_classifier()

example_choice = st.selectbox(
    "Try an example, or type your own text below",
    ["(type your own)"] + EXAMPLES,
)
default_text = "" if example_choice == "(type your own)" else example_choice

text = st.text_area(
    "Kinyarwanda text",
    value=default_text,
    placeholder="Andika interuro mu Kinyarwanda hano... (Type a Kinyarwanda sentence here...)",
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
    st.info("Enter some Kinyarwanda text above, or pick an example, then click Classify.")
