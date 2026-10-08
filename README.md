# Kinyarwanda Sentiment Classification

NLP summative project — **Text Classification for an African Language**.

A sentiment classifier (positive / negative / neutral) for **Kinyarwanda**, fine-tuned on the
[AfriSenti](https://arxiv.org/abs/2302.08956) Twitter sentiment dataset (Muhammad et al., 2023).

**Model (Hugging Face Hub):** [Iyamurinze/afroxlmr-kinyarwanda-sentiment](https://huggingface.co/Iyamurinze/afroxlmr-kinyarwanda-sentiment) · **[TODO: Live demo link]** · **[TODO: Demo video link]** · **[TODO: PDF report link]**

## Problem

Kinyarwanda is spoken by over 12 million people, primarily in Rwanda, yet has comparatively
little labeled NLP data and few deployed applications relative to its speaker population. This
project builds a sentiment classifier for Kinyarwanda social media text, useful e.g. for
monitoring public opinion, customer feedback, or public health messaging in Rwanda. Kinyarwanda
was chosen specifically so that the examples, errors, and model behavior could be genuinely
understood and defended by the author (a Kinyarwanda speaker) — an earlier version of this
project used Hausa, which the author could not read, making honest error analysis and defense
impractical.

## Dataset

- **Source:** [AfriSenti-Twitter](https://huggingface.co/datasets/HausaNLP/AfriSenti-Twitter)
  (Kinyarwanda subset), loaded via the script-free parquet mirror
  [`mteb/AfriSentiClassification`](https://huggingface.co/datasets/mteb/AfriSentiClassification).
- **Size:** 3,302 train / 827 validation / 1,026 test tweets.
- **Labels:** `positive`, `negative`, `neutral` — all three splits (including test) are
  reasonably balanced across all three classes (no data-quality gap in this language subset,
  unlike the Hausa subset of the same dataset — see note below).
- **Characteristics:** real, naturally-occurring, human-annotated tweets (not synthetic);
  contains `@mentions`, URLs, and emojis typical of Twitter text.
- **Preprocessing:** `@mentions` and URLs are normalized to placeholder tokens (`<USER>`,
  `<URL>`) since they carry no sentiment signal; emojis and punctuation are kept, since they
  often carry real sentiment information in tweets.
- **Smaller dataset than some other AfriSenti languages:** Kinyarwanda has ~4x less training
  data than, e.g., the Hausa subset (3,302 vs 14,172 tweets). This is expected to make
  fine-tuning somewhat more data-constrained — discussed further in Limitations.

## Methodology

| Approach | Description |
|---|---|
| Baseline 1 | TF-IDF (unigrams+bigrams, 20k features) + Logistic Regression |
| Baseline 2 (experiment) | TF-IDF + Linear SVM |
| Proposed model 1 | Fine-tuned [`Davlan/afro-xlmr-base`](https://huggingface.co/Davlan/afro-xlmr-base) — an XLM-R checkpoint further pretrained on 17 African languages (Alabi et al., 2022), **explicitly including standalone Kinyarwanda** (separate from Kirundi) |
| Proposed model 2 (experiment) | Fine-tuned [`castorini/afriberta_large`](https://huggingface.co/castorini/afriberta_large) (Ogueji et al., 2021) — pretrained *from scratch* on 11 African languages; for Kinyarwanda specifically, its pretraining data is **"Gahuza," a mixed Kinyarwanda/Kirundi corpus**, confirmed directly from the model's official Hugging Face card — not pure Kinyarwanda |

This sets up a genuine, verifiable research question: does exposure to *clean* Kinyarwanda
(AfroXLMR) outperform exposure to *code-mixed* Kinyarwanda/Kirundi (AfriBERTa) on a
pure-Kinyarwanda downstream task? Both models are fine-tuned identically (same
hyperparameters, same preprocessing) so the comparison isolates the effect of the pretrained
checkpoint.

Full training/evaluation code: [`notebooks/kinyarwanda_sentiment.ipynb`](notebooks/kinyarwanda_sentiment.ipynb).

## Results

Baselines (TF-IDF + classical ML) are measured on the **validation split**; fine-tuned
Transformers are measured on the full **test set** (n=1,026) by evaluating the pushed Hugging
Face Hub models directly — Kinyarwanda's test split is properly 3-way balanced, so test-set
macro-F1 is trustworthy here (unlike the Hausa subset of this same dataset, which had a
test-split class imbalance issue in an earlier iteration of this project). The two splits are
similar in size and class balance, so the comparison below is still meaningful even though the
exact split differs between baseline and Transformer rows.

| Approach | Accuracy | Macro-F1 |
|---|---|---|
| TF-IDF + Logistic Regression (val) | 58.16% | 58.38% |
| TF-IDF + Linear SVM (val) | 56.47% | 56.66% |
| Fine-tuned AfriBERTa (test) | 63.65% | 63.88% |
| **Fine-tuned AfroXLMR (test)** | **67.15%** | **67.47%** |

Both Transformers clearly beat the baseline by a wide margin — AfroXLMR improves macro-F1 by
roughly **+9 points** over the best classical baseline (Logistic Regression). This is a notably
larger gap than the same pipeline showed on the higher-resource Hausa subset of this dataset
(+4 points there): with less training data, the simple bag-of-words baseline struggles
proportionally more, while the pretrained Transformer's prior language knowledge compensates —
pretraining matters *more*, not less, in a lower-resource setting.

(AfroXLMR was retrained once to confirm reproducibility — a second independent run landed at
67.15%/67.47% macro-F1, close to the first run's 67.93%/68.28%, consistent with the expected
run-to-run variance from not fixing a random seed. The currently deployed model is from the
most recent run.)

**AfroXLMR wins by roughly +3.6-4.4 points over AfriBERTa across both runs**, directly confirming the hypothesis laid
out in Methodology: AfroXLMR's pretraining explicitly includes standalone Kinyarwanda, while
AfriBERTa's only Kinyarwanda-adjacent exposure is "Gahuza" (code-mixed Kinyarwanda/Kirundi).
Clean, language-specific pretraining data transferred better than code-mixed data for this
pure-Kinyarwanda downstream task — a genuine, explainable experimental finding, not just a
number. **AfroXLMR is the model we deploy** (see Links).

Both scores are meaningfully lower than what the same pipeline achieved on the Hausa subset of
this dataset in an earlier iteration (~80% macro-F1) — expected, given Kinyarwanda has ~4x less
training data (3,302 vs 14,172 tweets). See Limitations.

> **Note on reproducing these numbers:** training is stochastic (GPU non-determinism, no fixed
> seed), so re-runs will land close to but not exactly on these figures.

## Repository Structure

```
summative/
├── notebooks/
│   └── kinyarwanda_sentiment.ipynb   # data loading, EDA, preprocessing, baselines,
│                                       # AfroXLMR + AfriBERTa fine-tuning, evaluation,
│                                       # error analysis
├── app/
│   ├── app.py                        # Gradio inference app (local use / alternative hosts)
│   ├── streamlit_app.py              # Streamlit inference app (deployed via Streamlit
│   │                                   # Community Cloud)
│   ├── requirements.txt              # deployment deps (Streamlit app)
│   └── requirements-gradio.txt       # local-only deps (Gradio app)
├── data/                              # (gitignored) raw/processed data cache
├── report/                            # PDF report
├── requirements.txt                   # training/notebook dependencies
├── runtime.txt                        # pins Python version for Streamlit Cloud
└── README.md
```

## Reproducing This Project

1. **Train / fine-tune the models** (Google Colab recommended — needs a GPU):
   - Open `notebooks/kinyarwanda_sentiment.ipynb` in Colab (File → Upload notebook).
   - `Runtime → Change runtime type → T4 GPU`.
   - Run all cells. The dataset downloads automatically from the Hugging Face Hub.
   - Push the model(s) to the Hub (`trainer.push_to_hub(...)` for AfroXLMR /
     `afriberta_trainer.push_to_hub(...)` for AfriBERTa, in the final cell) using a
     Hugging Face write token stored as a Colab secret (🔑 icon in the sidebar) — never
     paste tokens directly into notebook cells or chat. AfroXLMR scored best in our run
     (see Results) and is the one the app defaults to.

2. **Run the web app locally:**
   ```bash
   pip install -r app/requirements.txt
   streamlit run app/streamlit_app.py      # or: pip install -r app/requirements-gradio.txt && python app/app.py
   ```
   Set `HF_MODEL_ID` to whichever model repo you pushed in step 1.

3. **Deploy (free, no card required) via Streamlit Community Cloud:**
   - Push this repo to GitHub.
   - Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, click "New app".
   - Pick this repo/branch, set **Main file path** to `app/streamlit_app.py`, deploy.
   - (Hugging Face Spaces' Gradio/Docker SDKs currently require a paid plan, so we deploy
     the Streamlit version instead — the Gradio app in `app/app.py` is kept for local use.)

## Evaluation

- **Metrics:** Accuracy and macro-averaged Precision/Recall/F1 — macro-F1 is the primary
  metric since it weighs all three sentiment classes equally regardless of (near-)balance.
- **Error analysis:** see the notebook's Error Analysis section (Section 9) for the full
  breakdown of most-confused label pairs on the test set. Spot-checking individual
  predictions against AfroXLMR, one representative failure: the model predicts `neutral`
  (68% confidence) for *"Ngo iyo Imana iza kwica satani, satani yari kuba yitabye Imana
  #fact"* (true label: `negative`) — a philosophical/religious statement with no explicit
  negative vocabulary, where sentiment is implied by tone/context rather than stated, which
  is a harder case for a model this size with this little training data.

## Limitations

- **Smaller dataset:** Kinyarwanda has ~4x less labeled training data than some other
  AfriSenti languages (e.g. Hausa), which may cap how much the Transformer models can learn
  relative to a higher-resource language.
- **Domain narrowness:** trained only on Twitter data, so performance on other Kinyarwanda
  text domains (news, formal writing) is untested.
- **Single-run results:** training was run once per configuration; results may vary slightly
  with different random seeds.
- **Lower absolute scores than higher-resource languages:** at 67.9% accuracy / 68.3%
  macro-F1, AfroXLMR's Kinyarwanda performance trails what the same pipeline achieves on
  higher-resource AfriSenti languages. This is expected given the ~4x smaller training set,
  not a flaw in the approach — see Results for the full explanation.

## Acknowledgements / Resources Used

- Dataset: Muhammad, S.H. et al. (2023). *AfriSenti: A Twitter Sentiment Analysis Benchmark
  for African Languages*. [arXiv:2302.08956](https://arxiv.org/abs/2302.08956).
- Pretrained models: Alabi, J.O. et al. (2022). *Adapting Pre-trained Language Models to
  African Languages via Multilingual Adaptive Fine-Tuning*. `Davlan/afro-xlmr-base` on
  Hugging Face. Ogueji, K. et al. (2021). *Small Data? No Problem! Exploring the Viability of
  Pretrained Multilingual Language Models for Low-resourced Languages*. `castorini/afriberta_large`
  on Hugging Face.
- Libraries: Hugging Face `transformers`, `datasets`; `scikit-learn`; `gradio`; `streamlit`.

## Links

- **GitHub Repository:** this repository
- **Deployed Web App:** **[TODO]**
- **Demo Video:** **[TODO]**
- **Full Report (PDF):** **[TODO]**
