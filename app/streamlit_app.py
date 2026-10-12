"""
Streamlit web app for the Kinyarwanda Sentiment Classification project.

Loads the fine-tuned model (trained in notebooks/kinyarwanda_sentiment.ipynb, pushed
to the Hugging Face Hub — see README for the comparison between AfroXLMR, AfriBERTa,
and the TF-IDF baselines) and serves it through a text-in / label-out UI.

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

# AfroXLMR beat AfriBERTa on the real test set (67.5% vs 63.9% macro-F1) — see README
# Results for the full comparison and why (clean Kinyarwanda pretraining vs AfriBERTa's
# mixed Kinyarwanda/Kirundi "Gahuza" exposure).
MODEL_ID = os.environ.get("HF_MODEL_ID", "Iyamurinze/afroxlmr-kinyarwanda-sentiment")

# Headline evaluation numbers, measured on the full AfriSenti-Kinyarwanda test split by
# running this exact pushed checkpoint (n=1,026). Surfaced in the UI so users can calibrate
# how much to trust any single prediction.
TEST_ACCURACY = "67.2%"
TEST_MACRO_F1 = "67.5%"
TEST_SIZE = "1,026"

# One colour per sentiment class. All three clear 4.5:1 against Streamlit's dark
# background (#0e1117): green 8.2:1, red 5.0:1, amber 8.6:1.
CLASS_COLORS = {
    "positive": "#22c55e",
    "negative": "#ef4444",
    "neutral": "#f59e0b",
}

# Real tweets from the AfriSenti-Kinyarwanda test set, with their gold labels, verified
# against this exact deployed model's predictions. Not hand-written sentences — the model
# is trained only on real tweet style and misreads clean textbook Kinyarwanda. Three the
# model gets right (one per class) plus one genuine failure, so the demo shows both.
EXAMPLES = [
    {
        "text": "Mwese muzagire Umwaka mushya muhire wa #2022",
        "gold": "positive",
        "note": "A new-year greeting — explicit, lexically clear sentiment.",
    },
    {
        "text": "Ibintu 3 byo kureba: Intumbero, Imyitwarire, Ururimi.",
        "gold": "negative",
        "note": "Short and list-like, with the sentiment carried by framing.",
    },
    {
        "text": "Ndumva nta bwenge mfite today ndaza kwiba ibitweet byanyuu",
        "gold": "neutral",
        "note": "Code-switched — Kinyarwanda mixed with English ('today').",
    },
    {
        "text": "Ngo iyo Imana iza kwica satani, satani yari kuba yitabye Imana #fact",
        "gold": "negative",
        "note": "Known failure: rhetorical religious framing, no explicit negative words.",
    },
]

GOLD_BY_TEXT = {ex["text"]: ex["gold"] for ex in EXAMPLES}

# --- Styling -----------------------------------------------------------------------
# Motion gate: classification is an *occasional* interaction (a user runs a handful of
# sentences), so a standard entrance animation is in budget. Purpose: state indication —
# the ring draw communicates the magnitude of each confidence value as it lands. Curves and
# durations follow the house motion scale; prefers-reduced-motion drops movement but keeps
# the final state (the inline stroke-dashoffset already holds the end value).
STYLES = """
<style>
  :root {
    --kin-ease-out: cubic-bezier(0.23, 1, 0.32, 1);
    --kin-surface: rgba(255, 255, 255, 0.035);
    --kin-border: rgba(255, 255, 255, 0.09);
    --kin-text-dim: rgba(250, 250, 250, 0.62);
    --kin-radius-card: 16px;
    --kin-radius-chip: 999px;
  }

  .kin-hero { margin: 0 0 6px; }
  .kin-hero h1 {
    font-size: clamp(2rem, 5vw, 2.9rem);
    font-weight: 800;
    letter-spacing: -0.025em;
    line-height: 1.1;
    margin: 0 0 10px;
    background: linear-gradient(100deg, #fafafa 25%, #22c55e 65%, #60a5fa 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .kin-hero p {
    color: var(--kin-text-dim);
    font-size: 0.95rem;
    line-height: 1.6;
    max-width: 62ch;
    margin: 0;
  }

  .kin-chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 18px 0 4px; }
  .kin-chip {
    display: inline-flex; align-items: center; gap: 7px;
    background: var(--kin-surface);
    border: 1px solid var(--kin-border);
    border-radius: var(--kin-radius-chip);
    padding: 5px 13px;
    font-size: 0.78rem;
    color: var(--kin-text-dim);
    white-space: nowrap;
  }
  .kin-chip b { color: #fafafa; font-weight: 650; }
  .kin-chip-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 0 3px rgba(34, 197, 94, 0.16);
  }

  .kin-section-label {
    font-size: 0.72rem;
    font-weight: 650;
    letter-spacing: 0.02em;
    color: var(--kin-text-dim);
    margin: 26px 0 10px;
  }

  /* Example buttons, styled as pickable cards. If Streamlit changes its button
     internals this selector simply stops matching and the buttons fall back to the
     default look — they keep working either way. */
  div[data-testid="stButton"] button[kind="secondary"] {
    width: 100%;
    text-align: left;
    justify-content: flex-start;
    background: var(--kin-surface);
    border: 1px solid var(--kin-border);
    border-radius: 12px;
    padding: 13px 15px;
    min-height: 94px;
    color: #fafafa;
    font-size: 0.82rem;
    line-height: 1.45;
    font-weight: 450;
    white-space: normal;
    transition: background-color 160ms ease, border-color 160ms ease;
  }
  @media (hover: hover) and (pointer: fine) {
    div[data-testid="stButton"] button[kind="secondary"]:hover {
      background: rgba(255, 255, 255, 0.07);
      border-color: rgba(255, 255, 255, 0.2);
      color: #fafafa;
    }
  }
  div[data-testid="stButton"] button[kind="secondary"]:active {
    transform: scale(0.985);
    transition: transform 160ms var(--kin-ease-out);
  }
  div[data-testid="stButton"] button[kind="secondary"]:focus-visible {
    outline: 2px solid #60a5fa;
    outline-offset: 2px;
  }

  /* --- Result ------------------------------------------------------------------ */
  .kin-verdict {
    display: flex; align-items: baseline; flex-wrap: wrap; gap: 10px;
    padding: 16px 20px;
    background: linear-gradient(100deg, var(--kin-accent-wash), transparent 72%);
    border: 1px solid var(--kin-accent-edge);
    border-left: 3px solid var(--kin-accent);
    border-radius: var(--kin-radius-card);
    margin-bottom: 14px;
    animation: kin-rise 220ms var(--kin-ease-out) both;
  }
  .kin-verdict-label {
    font-size: 1.5rem; font-weight: 750; letter-spacing: -0.015em;
    color: var(--kin-accent); text-transform: capitalize;
  }
  .kin-verdict-conf { font-size: 0.9rem; color: var(--kin-text-dim); }
  .kin-verdict-eyebrow {
    font-size: 0.72rem; font-weight: 650; color: var(--kin-text-dim);
    width: 100%; margin-bottom: -4px;
  }

  .kin-rings {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 12px;
  }
  .kin-ring-card {
    display: flex; flex-direction: column; align-items: center; gap: 10px;
    background: var(--kin-surface);
    border: 1px solid var(--kin-border);
    border-radius: var(--kin-radius-card);
    padding: 18px 12px 15px;
    animation: kin-rise 220ms var(--kin-ease-out) both;
  }
  .kin-ring-name {
    font-size: 0.82rem; font-weight: 600; color: #fafafa; text-transform: capitalize;
  }

  @keyframes kin-rise {
    from { opacity: 0; transform: translateY(8px) scale(0.97); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
  }

  .kin-footnote {
    font-size: 0.78rem; color: var(--kin-text-dim);
    line-height: 1.6; margin-top: 18px;
    padding-top: 14px; border-top: 1px solid var(--kin-border);
  }
  .kin-footnote a { color: #60a5fa; text-decoration: none; }
  .kin-footnote a:hover { text-decoration: underline; }

  @media (prefers-reduced-motion: reduce) {
    .kin-verdict, .kin-ring-card { animation: kin-fade 160ms ease both; }
    .kin-ring-arc { animation: none !important; }
    @keyframes kin-fade { from { opacity: 0; } to { opacity: 1; } }
  }
</style>
"""


RING_SIZE = 128
RING_STROKE = 11
RING_RADIUS = (RING_SIZE - RING_STROKE) / 2
RING_CIRCUMFERENCE = 2 * math.pi * RING_RADIUS


def rgba(hex_color: str, alpha: float) -> str:
    """Translucent version of a #rrggbb colour, as an rgba() string.

    Done in Python rather than with CSS color-mix() so the result does not depend on
    browser support for a relatively recent colour function — if color-mix() were
    unsupported the whole declaration would be dropped and the verdict card would lose
    its background and border entirely.
    """
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r}, {g}, {b}, {alpha})"


def ring_keyframes(index: int, pct: float) -> str:
    """The @keyframes rule that draws ring `index` from empty to its value."""
    offset = RING_CIRCUMFERENCE * (1 - pct)
    return (
        f"@keyframes kin-draw-{index} {{"
        f" from {{ stroke-dashoffset: {RING_CIRCUMFERENCE:.2f}; }}"
        f" to {{ stroke-dashoffset: {offset:.2f}; }} }}"
    )


def ring_svg(label: str, pct: float, color: str, index: int) -> str:
    """One circular confidence indicator, as inline SVG inside a card.

    The arc draws from empty to its value on mount (500ms, ease-out, staggered 60ms per
    card). `stroke-dashoffset` is also set inline to the *final* value so that under
    prefers-reduced-motion — where the animation is switched off — the ring still renders
    at the correct position rather than empty.
    """
    size = RING_SIZE
    cx = cy = size / 2
    offset = RING_CIRCUMFERENCE * (1 - pct)
    card_delay = 40 * index
    arc_delay = 120 + 60 * index
    return f"""
    <div class="kin-ring-card" style="animation-delay: {card_delay}ms;">
      <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" role="img"
           aria-label="{label}: {pct:.0%} confidence">
        <circle cx="{cx}" cy="{cy}" r="{RING_RADIUS}" fill="none"
                stroke="rgba(255,255,255,0.10)" stroke-width="{RING_STROKE}" />
        <circle class="kin-ring-arc" cx="{cx}" cy="{cy}" r="{RING_RADIUS}" fill="none"
                stroke="{color}" stroke-width="{RING_STROKE}" stroke-linecap="round"
                stroke-dasharray="{RING_CIRCUMFERENCE:.2f}"
                stroke-dashoffset="{offset:.2f}"
                transform="rotate(-90 {cx} {cy})"
                style="animation: kin-draw-{index} 500ms cubic-bezier(0.23,1,0.32,1)
                       {arc_delay}ms both;" />
        <text x="{cx}" y="{cy + 9}" text-anchor="middle" fill="#fafafa"
              font-size="26" font-weight="700"
              font-family="system-ui, -apple-system, sans-serif">{pct:.0%}</text>
      </svg>
      <span class="kin-ring-name">{label}</span>
    </div>
    """


@st.cache_resource(show_spinner=False)
def load_classifier():
    return pipeline("text-classification", model=MODEL_ID, top_k=None)


# --- Page --------------------------------------------------------------------------
st.set_page_config(
    page_title="Kinyarwanda Sentiment Classifier",
    page_icon="🇷🇼",
    layout="centered",
)
st.markdown(STYLES, unsafe_allow_html=True)

st.markdown(
    """
    <div class="kin-hero">
      <h1>Kinyarwanda Sentiment Classifier</h1>
      <p>A fine-tuned AfroXLMR Transformer that labels Kinyarwanda text as
      <strong>positive</strong>, <strong>negative</strong>, or <strong>neutral</strong>.
      Trained on the AfriSenti Twitter dataset (Muhammad et al., 2023) — an NLP summative
      project, not a general-purpose chatbot.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="kin-chips">
      <span class="kin-chip"><span class="kin-chip-dot"></span>AfroXLMR · fine-tuned</span>
      <span class="kin-chip">Test accuracy <b>{TEST_ACCURACY}</b></span>
      <span class="kin-chip">Macro-F1 <b>{TEST_MACRO_F1}</b></span>
      <span class="kin-chip">on <b>{TEST_SIZE}</b> held-out tweets</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if "kin_text" not in st.session_state:
    st.session_state.kin_text = ""

with st.spinner("Loading the model… (first run can take a minute on a cold start)"):
    classifier = load_classifier()


def _use_example(example_text: str) -> None:
    st.session_state.kin_text = example_text


st.markdown(
    '<div class="kin-section-label">Try a real tweet from the test set — '
    "each one's correct answer is known, so you can check the model</div>",
    unsafe_allow_html=True,
)

for row_start in (0, 2):
    cols = st.columns(2, gap="small")
    for col, example in zip(cols, EXAMPLES[row_start:row_start + 2]):
        with col:
            st.button(
                f"{example['gold'].capitalize()}  ·  “{example['text']}”",
                key=f"ex-{EXAMPLES.index(example)}",
                help=example["note"],
                on_click=_use_example,
                args=(example["text"],),
                use_container_width=True,
            )

with st.form("classify", border=False):
    text = st.text_area(
        "Kinyarwanda text",
        key="kin_text",
        placeholder="Andika interuro mu Kinyarwanda hano…  (Type a Kinyarwanda sentence here…)",
        height=110,
    )
    submitted = st.form_submit_button(
        "Classify", type="primary", use_container_width=False
    )

if submitted and text.strip():
    if not looks_like_kinyarwanda(text):
        st.warning(
            "This doesn't look like Kinyarwanda. The model was trained only on "
            "Kinyarwanda text, so the prediction below may not be meaningful. This check "
            "is a simple word-overlap heuristic, not a real language detector, so it can "
            "be wrong in either direction.",
            icon="⚠️",
        )

    with st.spinner("Classifying…"):
        scores = classifier(text)[0]

    scores_sorted = sorted(scores, key=lambda item: -item["score"])
    top = scores_sorted[0]
    accent = CLASS_COLORS.get(top["label"], "#60a5fa")

    gold = GOLD_BY_TEXT.get(text.strip())
    if gold is None:
        eyebrow = "Prediction"
    elif gold == top["label"]:
        eyebrow = f"Prediction · matches the known label ({gold})"
    else:
        eyebrow = f"Prediction · known label is {gold} — the model got this one wrong"

    keyframes = "\n".join(
        ring_keyframes(index, item["score"])
        for index, item in enumerate(scores_sorted)
    )
    rings = "".join(
        ring_svg(item["label"], item["score"],
                 CLASS_COLORS.get(item["label"], "#60a5fa"), index)
        for index, item in enumerate(scores_sorted)
    )

    # Emitted as its own markdown call rather than inline with the result markup, so the
    # <style> block never has to survive being parsed alongside sibling elements.
    st.markdown(f"<style>{keyframes}</style>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div style="--kin-accent: {accent};
                    --kin-accent-wash: {rgba(accent, 0.13)};
                    --kin-accent-edge: {rgba(accent, 0.28)};">
          <div class="kin-verdict">
            <span class="kin-verdict-eyebrow">{eyebrow}</span>
            <span class="kin-verdict-label">{top['label']}</span>
            <span class="kin-verdict-conf">{top['score']:.1%} confidence</span>
          </div>
          <div class="kin-rings">{rings}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='kin-footnote'>Each ring shows how much probability the model "
        "assigned to that class — the three always sum to 100%. A low top score means the "
        "model was genuinely unsure, not that the text is neutral.</div>",
        unsafe_allow_html=True,
    )

elif submitted:
    st.info("Enter some Kinyarwanda text, or pick one of the examples above.")

st.markdown(
    f"""
    <div class="kin-footnote">
      Model: <a href="https://huggingface.co/{MODEL_ID}" target="_blank">{MODEL_ID}</a> ·
      Data: <a href="https://arxiv.org/abs/2302.08956" target="_blank">AfriSenti</a>
      (Muhammad et al., 2023)
    </div>
    """,
    unsafe_allow_html=True,
)
