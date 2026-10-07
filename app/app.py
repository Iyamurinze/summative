"""
Gradio web app for the Kinyarwanda Sentiment Classification project.

Loads the fine-tuned model (trained in notebooks/kinyarwanda_sentiment.ipynb,
pushed to the Hugging Face Hub — see README for the comparison between AfroXLMR,
AfriBERTa, and the TF-IDF baselines) and serves it through a simple text-in /
label-out UI.

Run locally:
    pip install -r app/requirements-gradio.txt
    python app/app.py

Deploy on Hugging Face Spaces (note: Gradio/Docker SDK Spaces currently require a
paid plan — see app/streamlit_app.py for the actually-deployed version, hosted free
on Streamlit Community Cloud instead):
    1. Create a new Space (SDK: Gradio).
    2. Copy this file to the Space as app.py, and app/requirements-gradio.txt as
       requirements.txt.
    3. (Optional) override HF_MODEL_ID as a Space variable to point at a different
       pushed model repo.
"""

import os

import gradio as gr
from transformers import pipeline

# AfroXLMR beat AfriBERTa on the real test set (68.3% vs 63.9% macro-F1) — see README
# Results for the full comparison and why (clean Kinyarwanda pretraining vs AfriBERTa's
# mixed Kinyarwanda/Kirundi "Gahuza" exposure).
MODEL_ID = os.environ.get("HF_MODEL_ID", "Iyamurinze/afroxlmr-kinyarwanda-sentiment")

classifier = pipeline("text-classification", model=MODEL_ID, top_k=None)

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


def predict(text: str):
    if not text or not text.strip():
        return {}
    scores = classifier(text)[0]
    return {item["label"]: float(item["score"]) for item in scores}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        lines=3,
        placeholder="Andika interuro mu Kinyarwanda hano... (Type a Kinyarwanda sentence here...)",
        label="Kinyarwanda text",
    ),
    outputs=gr.Label(label="Predicted sentiment", num_top_classes=3),
    examples=EXAMPLES,
    title="Kinyarwanda Sentiment Classifier",
    description=(
        "Fine-tuned Transformer model for sentiment classification (positive / "
        "negative / neutral) of Kinyarwanda text, trained on the AfriSenti dataset "
        "(Muhammad et al., 2023). This is an NLP summative project — not a "
        "general-purpose chatbot."
    ),
)

if __name__ == "__main__":
    demo.launch()
