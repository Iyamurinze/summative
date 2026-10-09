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

import math
import os

import streamlit as st
from transformers import pipeline

from lang_guard import looks_like_kinyarwanda

# Color per sentiment class — used by the circular confidence rings below.
CLASS_COLORS = {
    "positive": "#22c55e",  # green
    "negative": "#ef4444",  # red
    "neutral": "#f59e0b",   # amber
}


def ring_svg(label: str, pct: float, color: str, size: int = 132, stroke: int = 14) -> str:
    """Render one circular (ring/donut) confidence indicator as inline SVG."""
    r = (size - stroke) / 2
    cx = cy = size / 2
    circumference = 2 * math.pi * r
    offset = circumference * (1 - pct)
    return f"""
    <div style="display:flex; flex-direction:column; align-items:center; gap:10px;
                background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
                border-radius: 16px; padding: 14px 10px; backdrop-filter: blur(6px);">
      <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
        <circle cx="{cx}" cy="{cy}" r="{r}" stroke="rgba(255,255,255,0.12)"
                stroke-width="{stroke}" fill="none" />
        <circle cx="{cx}" cy="{cy}" r="{r}" stroke="{color}" stroke-width="{stroke}"
                fill="none" stroke-dasharray="{circumference:.2f}"
                stroke-dashoffset="{offset:.2f}" stroke-linecap="round"
                transform="rotate(-90 {cx} {cy})"
                style="transition: stroke-dashoffset 0.6s ease;" />
        <text x="{cx}" y="{cy + 7}" text-anchor="middle" fill="#fafafa"
              font-size="22" font-weight="700" font-family="inherit">{pct:.0%}</text>
      </svg>
      <span style="font-size: 13px; font-weight: 600; color: #fafafa; text-transform: capitalize;">
        {label}
      </span>
    </div>
    """

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
    if not looks_like_kinyarwanda(text):
        st.warning(
            "⚠️ This doesn't look like Kinyarwanda. The model was trained only on "
            "Kinyarwanda text, so the prediction below may not be meaningful — this is a "
            "simple word-overlap heuristic, not a real language detector, so it can still "
            "be wrong in either direction.",
            icon="⚠️",
        )

    scores = classifier(text)[0]
    scores_sorted = sorted(scores, key=lambda x: -x["score"])
    top = scores_sorted[0]
    st.subheader(f"Prediction: **{top['label']}** ({top['score']:.1%} confidence)")

    cols = st.columns(len(scores_sorted))
    for col, item in zip(cols, scores_sorted):
        color = CLASS_COLORS.get(item["label"], "#60a5fa")
        with col:
            st.markdown(ring_svg(item["label"], item["score"], color), unsafe_allow_html=True)
elif text.strip() == "":
    st.info("Enter some Kinyarwanda text above, or pick an example, then click Classify.")
