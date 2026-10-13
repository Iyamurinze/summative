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
# running this exact pushed checkpoint (n=1,026).
TEST_ACCURACY = "67.2"
TEST_MACRO_F1 = "67.5"
TEST_SIZE = "1,026"

# One colour per sentiment class. All three clear 4.5:1 against the page background
# (#0e1117): green 8.2:1, red 5.0:1, amber 8.6:1.
CLASS_COLORS = {
    "positive": "#22c55e",
    "negative": "#ef4444",
    "neutral": "#f59e0b",
}
# Real tweets from the AfriSenti-Kinyarwanda test set with their gold labels, verified
# against this exact deployed model's predictions (three it gets right, one genuine
# failure). These are no longer offered as clickable samples, but the lookup below still
# fires if someone pastes one in — so the readout can say whether the model matched the
# known answer instead of silently sounding confident.
EXAMPLES = [
    {
        "text": "Mwese muzagire Umwaka mushya muhire wa #2022",
        "gold": "positive",
    },
    {
        "text": "Ibintu 3 byo kureba: Intumbero, Imyitwarire, Ururimi.",
        "gold": "negative",
    },
    {
        "text": "Ndumva nta bwenge mfite today ndaza kwiba ibitweet byanyuu",
        "gold": "neutral",
    },
    {
        "text": "Ngo iyo Imana iza kwica satani, satani yari kuba yitabye Imana #fact",
        "gold": "negative",
    },
]

GOLD_BY_TEXT = {ex["text"]: ex["gold"] for ex in EXAMPLES}

# --- Styling -----------------------------------------------------------------------
# The visual idea is an instrument readout rather than a dashboard: hairline rules
# instead of boxes, tabular monospace numerals, and colour carried by the data rather
# than by the chrome. Deliberately avoids the stock combination of a card grid, a
# gradient-wash panel and gradient headline text.
#
# Motion gate: classification is an *occasional* interaction, so a standard entrance is
# in budget. Purpose: state indication — the ring and bars convey magnitude as they land.
# prefers-reduced-motion drops movement but keeps the final state.
STYLES = """
<style>
  :root {
    --kin-ease-out: cubic-bezier(0.23, 1, 0.32, 1);
    --kin-rule: rgba(255, 255, 255, 0.10);
    --kin-rule-soft: rgba(255, 255, 255, 0.06);
    --kin-dim: rgba(250, 250, 250, 0.58);
    --kin-mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  }

  /* The spellcheck-disabling helper below renders as a zero-height iframe; keep it
     from reserving vertical space. Covers both the st.iframe and the older
     components.v1.html element names. */
  div[data-testid="stIFrame"]:has(iframe[height="1"]),
  div[data-testid="stCustomComponentV1"]:has(iframe[height="1"]) { display: none; }

  /* --- Masthead ---------------------------------------------------------------- */
  .kin-head { padding-bottom: 20px; border-bottom: 1px solid var(--kin-rule); }
  .kin-head h1 {
    font-size: clamp(1.75rem, 4vw, 2.35rem);
    font-weight: 660;
    letter-spacing: -0.03em;
    line-height: 1.15;
    margin: 0 0 9px;
    color: #fafafa;
  }
  .kin-head p {
    color: var(--kin-dim);
    font-size: 0.9rem;
    line-height: 1.65;
    max-width: 60ch;
    margin: 0;
  }

  /* Metrics as a measured strip divided by rules, not as a row of pills. */
  .kin-metrics {
    display: flex; flex-wrap: wrap;
    margin: 0 0 4px;
    border-bottom: 1px solid var(--kin-rule);
  }
  .kin-metric {
    padding: 14px 22px 14px 0;
    margin-right: 22px;
    border-right: 1px solid var(--kin-rule-soft);
  }
  .kin-metric:last-child { border-right: none; margin-right: 0; }
  .kin-metric-val {
    display: block;
    font-family: var(--kin-mono);
    font-size: 1.12rem;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    color: #fafafa;
    letter-spacing: -0.02em;
  }
  .kin-metric-key {
    display: block;
    font-size: 0.7rem;
    color: var(--kin-dim);
    margin-top: 3px;
  }

  .kin-eyebrow {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--kin-dim);
    font-family: var(--kin-mono);
  }

  /* --- Readout ------------------------------------------------------------------ */
  .kin-readout {
    display: grid;
    grid-template-columns: 1fr auto;
    grid-template-areas: "head ring" "bars ring";
    gap: 4px 28px;
    align-items: start;
    margin-top: 22px;
    padding: 22px 24px;
    background: rgba(255, 255, 255, 0.022);
    border-top: 2px solid var(--kin-accent);
    border-radius: 2px 2px 10px 10px;
    animation: kin-rise 220ms var(--kin-ease-out) both;
  }
  .kin-readout-head { grid-area: head; }
  .kin-verdict {
    display: block;
    font-size: 2.1rem;
    font-weight: 660;
    letter-spacing: -0.035em;
    line-height: 1.1;
    color: var(--kin-accent);
    text-transform: capitalize;
    margin: 7px 0 3px;
  }
  .kin-verdict-note { font-size: 0.8rem; color: var(--kin-dim); line-height: 1.5; }
  .kin-ring-slot { grid-area: ring; }

  /* Runner-up classes: slim bars, so the one ring stays the single loud element. */
  .kin-bars { grid-area: bars; margin-top: 16px; display: grid; gap: 9px; max-width: 340px; }
  .kin-bar {
    display: grid;
    grid-template-columns: 62px 1fr 48px;
    align-items: center;
    gap: 11px;
    font-size: 0.76rem;
  }
  .kin-bar-name { color: var(--kin-dim); text-transform: capitalize; }
  .kin-bar-track {
    height: 4px; border-radius: 2px;
    background: rgba(255, 255, 255, 0.08);
    overflow: hidden;
  }
  .kin-bar-fill {
    display: block; height: 100%; border-radius: 2px;
    transform-origin: left center;
    animation: kin-bar-grow 500ms var(--kin-ease-out) both;
  }
  .kin-bar-val {
    text-align: right;
    font-family: var(--kin-mono);
    font-variant-numeric: tabular-nums;
    color: rgba(250, 250, 250, 0.82);
  }

  @keyframes kin-rise {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes kin-bar-grow {
    from { transform: scaleX(0); }
    to   { transform: scaleX(1); }
  }

  .kin-note {
    font-size: 0.76rem; color: var(--kin-dim); line-height: 1.6;
    margin-top: 16px; padding-top: 13px; border-top: 1px solid var(--kin-rule-soft);
  }
  .kin-note a { color: rgba(250, 250, 250, 0.82); text-decoration: underline;
                text-underline-offset: 2px; text-decoration-color: var(--kin-rule); }
  .kin-note a:hover { text-decoration-color: currentColor; }

  @media (max-width: 640px) {
    .kin-readout {
      grid-template-columns: 1fr;
      grid-template-areas: "head" "ring" "bars";
      justify-items: start;
    }
  }

  @media (prefers-reduced-motion: reduce) {
    .kin-readout { animation: kin-fade 160ms ease both; }
    .kin-ring-arc, .kin-bar-fill { animation: none !important; }
    @keyframes kin-fade { from { opacity: 0; } to { opacity: 1; } }
  }
</style>
"""

RING_SIZE = 148
RING_STROKE = 9
RING_RADIUS = (RING_SIZE - RING_STROKE) / 2
RING_CIRCUMFERENCE = 2 * math.pi * RING_RADIUS


def ring_svg(pct: float, color: str, label: str) -> str:
    """The single large confidence ring for the predicted class.

    Only the winning class gets a ring — the runners-up are slim bars — so the page has
    one loud element instead of three competing ones.

    The arc draws from empty to its value on mount. `stroke-dashoffset` is also set
    inline to the *final* value so that under prefers-reduced-motion, where the animation
    is switched off, the ring still renders at the right position rather than empty.
    """
    cx = cy = RING_SIZE / 2
    offset = RING_CIRCUMFERENCE * (1 - pct)
    return f"""
    <svg class="kin-ring" width="{RING_SIZE}" height="{RING_SIZE}"
         viewBox="0 0 {RING_SIZE} {RING_SIZE}" role="img"
         aria-label="{label}: {pct:.1%} confidence">
      <circle cx="{cx}" cy="{cy}" r="{RING_RADIUS}" fill="none"
              stroke="rgba(255,255,255,0.08)" stroke-width="{RING_STROKE}" />
      <circle class="kin-ring-arc" cx="{cx}" cy="{cy}" r="{RING_RADIUS}" fill="none"
              stroke="{color}" stroke-width="{RING_STROKE}" stroke-linecap="round"
              stroke-dasharray="{RING_CIRCUMFERENCE:.2f}"
              stroke-dashoffset="{offset:.2f}"
              transform="rotate(-90 {cx} {cy})"
              style="animation: kin-draw 560ms cubic-bezier(0.23,1,0.32,1) 110ms both;" />
      <text x="{cx}" y="{cy + 4}" text-anchor="middle" fill="#fafafa"
            font-size="30" font-weight="600" letter-spacing="-1"
            font-family="ui-monospace, SFMono-Regular, Menlo, monospace"
            style="font-variant-numeric: tabular-nums;">{pct * 100:.0f}</text>
      <text x="{cx}" y="{cy + 26}" text-anchor="middle" fill="rgba(250,250,250,0.5)"
            font-size="10" letter-spacing="1.4"
            font-family="ui-monospace, SFMono-Regular, Menlo, monospace">PERCENT</text>
    </svg>
    """


def ring_keyframes(pct: float) -> str:
    offset = RING_CIRCUMFERENCE * (1 - pct)
    return (
        f"@keyframes kin-draw {{"
        f" from {{ stroke-dashoffset: {RING_CIRCUMFERENCE:.2f}; }}"
        f" to {{ stroke-dashoffset: {offset:.2f}; }} }}"
    )


def bar_row(label: str, pct: float, color: str, index: int) -> str:
    return f"""
    <div class="kin-bar">
      <span class="kin-bar-name">{label}</span>
      <span class="kin-bar-track">
        <span class="kin-bar-fill"
              style="width: {pct * 100:.1f}%; background: {color};
                     animation-delay: {200 + 70 * index}ms;"></span>
      </span>
      <span class="kin-bar-val">{pct * 100:.1f}%</span>
    </div>
    """


@st.cache_resource(show_spinner=False)
def load_classifier():
    return pipeline("text-classification", model=MODEL_ID, top_k=None)


def html_block(markup: str) -> str:
    """Flatten indentation so Streamlit's markdown parser passes the HTML through.

    Markdown turns any line indented four or more spaces into a code block, and a blank
    line closes an open HTML block. Multi-line markup built from indented f-strings trips
    both: once an interpolated fragment leaves a whitespace-only line behind, the block
    closes and the following indented tags get rendered as literal code. Stripping
    per-line indentation and dropping blank lines avoids it without changing how the
    HTML renders, since HTML ignores that whitespace anyway.

    Must be applied to the fully composed string, after interpolation, so that nested
    fragments are flattened too.
    """
    return "\n".join(
        stripped for line in markup.splitlines() if (stripped := line.strip())
    )


def embed_html(html: str, height: int = 1) -> None:
    """Embed a raw HTML/JS snippet, across Streamlit versions.

    st.iframe supersedes st.components.v1.html, which is deprecated and scheduled for
    removal; fall back to the old call on installs that predate st.iframe. Height
    defaults to 1 rather than 0 because st.iframe rejects a height of zero — the
    stylesheet hides the 1px container outright.
    """
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:  # pragma: no cover - only on Streamlit versions predating st.iframe
        import streamlit.components.v1 as components

        components.html(html, height=height)


# --- Page --------------------------------------------------------------------------
st.set_page_config(
    page_title="Kinyarwanda Sentiment Classifier",
    page_icon="🇷🇼",
    layout="centered",
)
st.markdown(html_block(STYLES), unsafe_allow_html=True)

# Kinyarwanda is not in any browser's dictionary, so the native spellchecker underlines
# essentially every word typed into the box. Streamlit's text_area exposes no spellcheck
# option, so reach the real <textarea> from a zero-height helper component. The observer
# re-applies it because Streamlit re-creates the node on rerun.
embed_html(
    """
    <script>
      const doc = window.parent.document;
      const disableSpellcheck = () => {
        doc.querySelectorAll('textarea').forEach((node) => {
          node.setAttribute('spellcheck', 'false');
          node.setAttribute('autocorrect', 'off');
          node.setAttribute('autocapitalize', 'off');
        });
      };
      disableSpellcheck();
      new MutationObserver(disableSpellcheck)
        .observe(doc.body, { childList: true, subtree: true });
    </script>
    """,
)

st.markdown(
    html_block(
        """
    <div class="kin-head">
      <h1>Kinyarwanda Sentiment Classifier</h1>
      <p>A fine-tuned AfroXLMR Transformer that labels Kinyarwanda text as positive,
      negative, or neutral. Trained on the AfriSenti Twitter corpus (Muhammad et al.,
      2023) — an NLP summative project, not a general-purpose chatbot.</p>
    </div>
    """
    ),
    unsafe_allow_html=True,
)

st.markdown(
    html_block(
        f"""
    <div class="kin-metrics">
      <div class="kin-metric">
        <span class="kin-metric-val">{TEST_ACCURACY}%</span>
        <span class="kin-metric-key">Test accuracy</span>
      </div>
      <div class="kin-metric">
        <span class="kin-metric-val">{TEST_MACRO_F1}%</span>
        <span class="kin-metric-key">Macro F1</span>
      </div>
      <div class="kin-metric">
        <span class="kin-metric-val">{TEST_SIZE}</span>
        <span class="kin-metric-key">Held-out tweets</span>
      </div>
      <div class="kin-metric">
        <span class="kin-metric-val">3</span>
        <span class="kin-metric-key">Classes</span>
      </div>
    </div>
    """
    ),
    unsafe_allow_html=True,
)

if "kin_text" not in st.session_state:
    st.session_state.kin_text = ""

with st.spinner("Loading the model… (first run can take a minute on a cold start)"):
    classifier = load_classifier()

with st.form("classify", border=False):
    text = st.text_area(
        "Kinyarwanda text",
        key="kin_text",
        placeholder="Andika interuro mu Kinyarwanda hano…  (Type a Kinyarwanda sentence here…)",
        height=108,
    )
    submitted = st.form_submit_button("Classify", type="primary")

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

    ranked = sorted(scores, key=lambda item: -item["score"])
    top, rest = ranked[0], ranked[1:]
    accent = CLASS_COLORS.get(top["label"], "#60a5fa")

    # Only say anything when there is a reference label to compare against; most input
    # is free text, where a "no reference label" line would just be noise on every run.
    gold = GOLD_BY_TEXT.get(text.strip())
    if gold is None:
        note = ""
    elif gold == top["label"]:
        note = f"Matches the known label for this sample ({gold})."
    else:
        note = f"The known label is {gold} — the model got this sample wrong."

    bars = "".join(
        bar_row(item["label"], item["score"],
                CLASS_COLORS.get(item["label"], "#60a5fa"), index)
        for index, item in enumerate(rest)
    )

    # Emitted separately so the <style> block never has to survive being parsed
    # alongside sibling elements.
    st.markdown(f"<style>{ring_keyframes(top['score'])}</style>",
                unsafe_allow_html=True)
    st.markdown(
        html_block(
            f"""
        <div class="kin-readout" style="--kin-accent: {accent};">
          <div class="kin-readout-head">
            <span class="kin-eyebrow">Predicted</span>
            <span class="kin-verdict">{top['label']}</span>
            {f'<span class="kin-verdict-note">{note}</span>' if note else ''}
          </div>
          <div class="kin-ring-slot">
            {ring_svg(top['score'], accent, top['label'])}
          </div>
          <div class="kin-bars">{bars}</div>
        </div>
        """
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='kin-note'>The three class scores always sum to 100%. A low top "
        "score means the model was genuinely unsure — not that the text is neutral.</div>",
        unsafe_allow_html=True,
    )

elif submitted:
    st.info("Enter some Kinyarwanda text above to classify it.")

st.markdown(
    html_block(
        f"""
    <div class="kin-note">
      Model <a href="https://huggingface.co/{MODEL_ID}" target="_blank">{MODEL_ID}</a>
      &nbsp;·&nbsp; Data
      <a href="https://arxiv.org/abs/2302.08956" target="_blank">AfriSenti</a>
      (Muhammad et al., 2023)
    </div>
    """
    ),
    unsafe_allow_html=True,
)
