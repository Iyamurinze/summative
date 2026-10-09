# Demo Video Script — Kinyarwanda Sentiment Classification

Target length: 7-10 minutes (confirm with your instructor whether the 10-15 min range
mentioned elsewhere in the brief applies instead — if so, slow down and add detail where
marked **[EXPAND]** below rather than padding with filler).

Record your screen with voice-over. Suggested windows to have ready before you start:
your Colab notebook (scrolled to key sections), the AfroXLMR model page on Hugging Face,
and the deployed Streamlit app in a browser tab.

Read this as a guide for what to *say*, not a word-for-word script — say it in your own
words so it sounds like you understand it (because you do).

---

## 1. Introduction (≈1 min)

**Say:**
- "This is my NLP summative project: a sentiment classifier for Kinyarwanda."
- Why Kinyarwanda: "I originally built this for Hausa, but realized I couldn't actually
  read Hausa — which meant I couldn't honestly explain the model's errors or defend my
  own project. I rebuilt it around Kinyarwanda, a language I actually speak, specifically
  so every example and error I show you here is something I can genuinely explain."
- "The task is 3-class sentiment classification — positive, negative, neutral — on
  Kinyarwanda tweets."

**Show:** your face/screen intro, maybe the README title.

---

## 2. Dataset (≈1 min)

**Say:**
- "I used AfriSenti, the largest sentiment benchmark for African languages, from a
  SemEval-2023 shared task — real, human-annotated tweets, not synthetic data."
- "The Kinyarwanda subset is small relative to some other languages in this dataset —
  3,302 training tweets, 827 validation, 1,026 test — about 4x smaller than the Hausa
  subset of the same dataset. That matters later when I explain the results."
- Mention preprocessing briefly: "I normalized @mentions and URLs since they carry no
  sentiment signal, but kept emojis and punctuation since those often do."

**Show:** the EDA cell/plot in your notebook (class distribution chart), the dataset
loading cell.

---

## 3. Methodology — baseline and models (≈2 min) **[EXPAND if 10-15 min]**

**Say:**
- "I started with a baseline: TF-IDF features — basically word and bigram frequency —
  with Logistic Regression and a Linear SVM. This tells me what's achievable with no real
  language understanding, just surface word patterns."
- "Then I fine-tuned two different pretrained Transformer models, and this is the core
  experiment of the project. Both are adapted for African languages, but differently:"
  - "**AfroXLMR** took a general multilingual model, XLM-R, and continued pretraining it
    on 17 African languages — Kinyarwanda included as its own standalone language."
  - "**AfriBERTa** was pretrained from scratch on a smaller set of 11 languages, but for
    Kinyarwanda specifically, its only exposure was through something called 'Gahuza' —
    a mixed Kinyarwanda/Kirundi corpus. I confirmed this directly from the model's own
    documentation on Hugging Face, not from memory."
- "So the experiment is: does clean language-specific pretraining actually beat
  code-mixed pretraining, even when the two languages — Kinyarwanda and Kirundi — are
  closely related? That's a real, testable question, not just 'which model is bigger.'"
- Briefly mention training setup: "Both fine-tuned identically — same learning rate,
  same epochs, same preprocessing — so the only variable is the pretrained checkpoint."

**Show:** the model-loading cell, the TrainingArguments cell, maybe the two model cards
on Hugging Face side-by-side (scroll to the "language" / pretraining description).

---

## 4. Results (≈1.5 min)

**Say the numbers and what they mean, not just the numbers:**
- "Baseline: around 58% macro-F1. Both Transformers clearly beat that."
- "AfriBERTa: about 64% macro-F1. AfroXLMR: about 67-68%."
- "AfroXLMR wins by roughly 4 points — and importantly, it wins on *every individual
  class*, not just on average. That tells me this isn't a fluke of how the metric
  averages things — AfroXLMR genuinely understands Kinyarwanda better, which is exactly
  what my hypothesis predicted."
- "One more thing I noticed: the improvement from baseline to best model was actually
  *bigger* here than when I ran the same pipeline on Hausa, which has way more data.
  My read on that: with less training data, the baseline really struggles, but the
  pretrained model's prior knowledge compensates more — so pretraining matters even more
  when you have less data to fine-tune on."

**Show:** your results table/comparison cell output in the notebook.

---

## 5. Error analysis (≈1.5 min) — show successes AND failures

**Say:**
- "Let me show you what the model gets right and wrong, and why."
- Pick 1-2 correct examples: read the Kinyarwanda tweet, say what it means, say the
  predicted label and confidence. E.g. "Mwese muzagire Umwaka mushya muhire — 'may you
  all have a happy new year' — correctly predicted positive, 94% confidence. That's
  explicit, clear sentiment, and the model nails it."
- Pick the failure example: "Ngo iyo Imana iza kwica satani... — a philosophical,
  almost rhetorical statement about God and Satan. It's actually labeled negative in the
  dataset, but my model predicts neutral. I think this happens because the sentiment
  here is implied by tone and cultural context, not by any explicit negative word — and
  with only 3,300 training examples, the model just hasn't seen enough examples of that
  kind of implicit, rhetorical sentiment to learn it reliably."
- "This is a real limitation, not something I'm hiding — it's exactly the kind of error
  I'd expect given the dataset size."

**Show:** run the model live on these exact examples in the notebook or the deployed app.

---

## 6. Deployment (≈1 min)

**Say:**
- "I pushed the fine-tuned AfroXLMR model to the Hugging Face Hub, and built a Streamlit
  app that loads it directly from there."
- "I tried Hugging Face Spaces first, but their Gradio/Docker hosting now requires a
  paid plan, so I deployed on Streamlit Community Cloud instead — free, no card needed."
- Walk through the live app: type a Kinyarwanda sentence, click classify, show the
  prediction and confidence bars. Do this with at least 2 different inputs — one where
  it's clearly right, maybe one where it's less confident.

**Show:** the live deployed app in your browser, actually typing and getting predictions.

---

## 7. Limitations and conclusion (≈1 min)

**Say:**
- "The main limitation is dataset size — Kinyarwanda has much less labeled data than
  some other AfriSenti languages, which caps how much the model can learn, especially
  for implicit sentiment."
- "I also didn't fix a random seed, so there's some natural run-to-run variance — I
  actually retrained AfroXLMR once to check this, and got consistent results within
  about half a point, which gave me confidence the result is stable, not a fluke."
- "With more time, I'd want to try fine-tuning first on the full multilingual AfriSenti
  dataset before specializing on Kinyarwanda, to partially make up for the limited
  Kinyarwanda-specific data."
- Close: "Overall, this project shows that for a low-resource language like Kinyarwanda,
  *which* pretraining data a model saw matters a lot — clean, language-specific data beat
  code-mixed data by a clear, consistent margin, and that's something I could only
  really verify and explain because I chose a language I actually understand."

---

## Quick checklist before you record

- [ ] Can you explain, without looking at notes, why AfroXLMR beat AfriBERTa?
- [ ] Can you explain what macro-F1 is and why it's the right metric here (vs. plain accuracy)?
- [ ] Can you read and translate the 3-4 example tweets you're going to show?
- [ ] Is the Streamlit app actually live and working right now?
- [ ] Do you know what each major hyperparameter in `TrainingArguments` does (learning
      rate, batch size, epochs, weight decay)?
- [ ] Can you explain, in your own words, what TF-IDF is and why it's a reasonable baseline?

If you can answer all of these out loud without this script, you're ready — the video
should sound like you explaining your own work, not reading a script.
