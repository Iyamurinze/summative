"""
Gradio web app for the Hausa Sentiment Classification project.

Loads the fine-tuned AfriBERTa model (trained in notebooks/hausa_sentiment.ipynb,
pushed to the Hugging Face Hub as Iyamurinze/afriberta-hausa-sentiment — our
best-performing model, see README for the comparison against AfroXLMR and the
TF-IDF baselines) and serves it through a simple text-in / label-out UI.

Run locally:
    pip install -r app/requirements.txt
    python app/app.py

Deploy on Hugging Face Spaces:
    1. Create a new Space (SDK: Gradio).
    2. Copy this file to the Space as app.py, and app/requirements.txt as requirements.txt.
    3. (Optional) override HF_MODEL_ID as a Space variable to point at a different
       pushed model repo.
"""

import os

import gradio as gr
from transformers import pipeline

MODEL_ID = os.environ.get("HF_MODEL_ID", "Iyamurinze/afriberta-hausa-sentiment")

classifier = pipeline("text-classification", model=MODEL_ID, top_k=None)

# Real tweets from the AfriSenti-Hausa test set (with known gold labels), chosen to show
# both correct predictions and a genuine failure mode — not hand-written sentences, since
# the model is trained only on real tweet style and misreads clean formal Hausa (see README
# Limitations). Verified against the deployed model before inclusion here.
EXAMPLES = [
    "allah ya gafartawa mallam allah ya saada shi da annabin rahama",  # gold: positive
    "wai na dawo daga makkah xani madina najeria kasarmu ta gado",  # gold: negative
    "allah sarki talaka ba wan allah allah kashuge mn gaba",  # gold: positive, model says neutral — known failure mode
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
        placeholder="Rubuta jimla a Hausa anan... (Type a Hausa sentence here...)",
        label="Hausa text",
    ),
    outputs=gr.Label(label="Predicted sentiment", num_top_classes=3),
    examples=EXAMPLES,
    title="Hausa Sentiment Classifier",
    description=(
        "Fine-tuned AfriBERTa model for sentiment classification (positive / "
        "negative / neutral) of Hausa text, trained on the AfriSenti dataset "
        "(Muhammad et al., 2023). This is an NLP summative project — not a "
        "general-purpose chatbot."
    ),
)

if __name__ == "__main__":
    demo.launch()
