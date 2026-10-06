# Hausa Sentiment Classification

NLP summative project — **Text Classification for an African Language**.

A sentiment classifier (positive / negative / neutral) for **Hausa**, fine-tuned on the
[AfriSenti](https://arxiv.org/abs/2302.08956) Twitter sentiment dataset (Muhammad et al., 2023).

**Model (Hugging Face Hub):** [Iyamurinze/afriberta-hausa-sentiment](https://huggingface.co/Iyamurinze/afriberta-hausa-sentiment) · **[TODO: Live demo link]** · **[TODO: Demo video link]** · **[TODO: PDF report link]**

## Problem

Hausa is spoken by over 70 million people, yet has comparatively little labeled NLP data
and few deployed applications relative to its speaker population. This project builds a
sentiment classifier for Hausa social media text, useful e.g. for monitoring public opinion,
customer feedback, or public health messaging in Hausa-speaking regions.

## Dataset

- **Source:** [AfriSenti-Twitter](https://huggingface.co/datasets/HausaNLP/AfriSenti-Twitter)
  (Hausa subset), loaded via the script-free parquet mirror
  [`mteb/AfriSentiClassification`](https://huggingface.co/datasets/mteb/AfriSentiClassification).
- **Size:** 14,172 train / 2,677 validation / 2,048 test tweets.
- **Labels:** `positive`, `negative`, `neutral` — roughly balanced (~33% each) in train and
  validation.
- **Characteristics:** real, naturally-occurring, human-annotated tweets (not synthetic);
  average length ~14 words; contains `@mentions`, URLs, and emojis typical of Twitter text.
- **Preprocessing:** `@mentions` and URLs are normalized to placeholder tokens (`<USER>`,
  `<URL>`) since they carry no sentiment signal; emojis and punctuation are kept, since they
  often carry real sentiment information in tweets.
- **Known data quirk:** the public **test** split (as mirrored via `mteb/AfriSentiClassification`)
  contains **zero `neutral`-labeled examples** (1,755 positive / 293 negative / 0 neutral),
  unlike train/validation which are 3-way balanced. We verified this directly from the raw
  label values, not just class names — it is a genuine property of the released test split
  (likely because AfriSenti's official SemEval-2023 test set had some gold labels withheld
  for leaderboard scoring), not a bug in our code. See **Results** and **Limitations** below
  for how this affects evaluation.

## Methodology

| Approach | Description |
|---|---|
| Baseline 1 | TF-IDF (unigrams+bigrams, 20k features) + Logistic Regression |
| Baseline 2 (experiment) | TF-IDF + Linear SVM |
| Proposed model 1 | Fine-tuned [`Davlan/afro-xlmr-base`](https://huggingface.co/Davlan/afro-xlmr-base) — an XLM-R checkpoint further pretrained on 17 African languages (Alabi et al., 2022), fine-tuned end-to-end with a 3-class classification head |
| Proposed model 2 (experiment) | Fine-tuned [`castorini/afriberta_large`](https://huggingface.co/castorini/afriberta_large) (Ogueji et al., 2021) — pretrained *from scratch* on only 11 African languages (no general multilingual pretraining), fine-tuned identically to model 1 for a direct comparison of pretraining strategy |

Full training/evaluation code: [`notebooks/hausa_sentiment.ipynb`](notebooks/hausa_sentiment.ipynb).

## Results

All approaches below are compared **on the validation split** (3-way balanced — see the data
quirk noted above for why we don't lead with raw test-set numbers):

| Approach | Accuracy | Macro-F1 |
|---|---|---|
| TF-IDF + Logistic Regression | 0.757 | 0.759 |
| TF-IDF + Linear SVM | 0.742 | 0.743 |
| Fine-tuned AfroXLMR (3 epochs) | 0.796 | 0.796 |
| **Fine-tuned AfriBERTa (3 epochs)** | **0.799** | **0.7995** |

Both fine-tuned Transformers beat the strongest baseline (Logistic Regression) by **+3.6-4.0
points** macro-F1, consistent with contextual, Hausa-aware embeddings outperforming TF-IDF's
bag-of-words features. More interestingly, **AfriBERTa (503MB, pretrained from scratch on only
11 African languages) very slightly edges out AfroXLMR (1.1GB, adapted from general
multilingual XLM-R)** despite being under half the size — suggesting that for this in-domain
task, pretraining exclusively on relevant languages can match or beat adapting a larger
general-purpose multilingual model. See Methodology for why we ran this comparison.

**Per-epoch training/validation progress:**

| Epoch | AfroXLMR Train Loss | AfroXLMR Val F1 | AfriBERTa Train Loss | AfriBERTa Val F1 |
|---|---|---|---|---|
| 1 | 0.640 | 0.777 | 0.575 | 0.794 |
| 2 | 0.493 | 0.793 | 0.387 | 0.799 |
| 3 | 0.386 | 0.796 | 0.229 | 0.7995 |

AfriBERTa's train loss drops much faster (0.575→0.229) than AfroXLMR's (0.640→0.386) while its
validation F1 barely moves after epoch 1 (0.794→0.7995) — a classic sign it's fitting the
training data more aggressively per epoch, though it doesn't visibly overfit within 3 epochs
here. For both models, `load_best_model_at_end` selected the epoch-3 checkpoint by F1.

**Test-set result (reported with the caveat above):** AfroXLMR scores accuracy 0.829 but
macro-F1 only **0.514** (AfriBERTa: accuracy 0.833, macro-F1 0.515) — this is *not* a sign
either model is worse than validation suggests. Since `neutral` never occurs in the true test
labels, the classifiers' occasional `neutral` predictions (which would be perfectly reasonable
given their training distribution) register as pure errors with no matching true class, and
`neutral`'s precision/recall/F1 are each forced to 0 — which drags the 3-way macro average down
heavily despite strong positive/negative performance (AfroXLMR F1 0.91 / 0.60 respectively on
those two classes). We treat the **validation macro-F1 figures above** as representative, and
discuss the test-set artifact explicitly in Error Analysis / Limitations.

> **Note on reproducing these exact numbers:** training is stochastic (GPU non-determinism,
> no fixed seed), so a re-run will land close to but not exactly on these figures — we saw
> ~0.79-0.80 validation macro-F1 for AfroXLMR across two separate runs.

## Repository Structure

```
summative/
├── notebooks/
│   └── hausa_sentiment.ipynb   # data loading, EDA, preprocessing, baselines,
│                                 # AfroXLMR fine-tuning, evaluation, error analysis
├── app/
│   ├── app.py                  # Gradio inference app (local use / alternative hosts)
│   ├── streamlit_app.py        # Streamlit inference app (deployed via Streamlit Community Cloud)
│   └── requirements.txt
├── data/                        # (gitignored) raw/processed data cache
├── report/                      # PDF report
├── requirements.txt             # training/notebook dependencies
└── README.md
```

## Reproducing This Project

1. **Train / fine-tune the models** (Google Colab recommended — needs a GPU):
   - Open `notebooks/hausa_sentiment.ipynb` in Colab (File → Upload notebook).
   - `Runtime → Change runtime type → T4 GPU`.
   - Run all cells. The dataset downloads automatically from the Hugging Face Hub.
   - Push the trained model to the Hub (`trainer.push_to_hub(...)`) using a Hugging Face
     write token stored as a Colab secret (🔑 icon in the sidebar) — never paste tokens
     directly into notebook cells or chat.

2. **Run the web app locally:**
   ```bash
   pip install -r app/requirements.txt
   streamlit run app/streamlit_app.py      # or: python app/app.py  (Gradio version)
   ```
   By default this loads [`Iyamurinze/afriberta-hausa-sentiment`](https://huggingface.co/Iyamurinze/afriberta-hausa-sentiment)
   (our best-performing model — see Results). Override with `HF_MODEL_ID` to point at a
   different pushed model repo.

3. **Deploy (free, no card required) via Streamlit Community Cloud:**
   - Push this repo to GitHub.
   - Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, click "New app".
   - Pick this repo/branch, set **Main file path** to `app/streamlit_app.py`, deploy.
   - (Hugging Face Spaces' Gradio/Docker SDKs now require a paid plan, so we deploy the
     Streamlit version instead — the Gradio app in `app/app.py` is kept for local use and
     works identically.)

## Evaluation

- **Metrics:** Accuracy and macro-averaged Precision/Recall/F1 — macro-F1 is the primary
  metric since it weighs all three sentiment classes equally regardless of (near-)balance.
- **Error analysis:** on the test set, AfroXLMR made 351/2048 errors (17.1%).
  Most-confused label pairs (true → predicted):

  | True | Predicted | Count |
  |---|---|---|
  | positive | negative | 131 |
  | positive | neutral | 122 |
  | negative | neutral | 64 |
  | negative | positive | 34 |

  Notably, 186 of the 351 errors (53%) are the model predicting `neutral` — a label that
  never actually occurs in this test split (see the data quirk above). Excluding those, the
  "real" confusion is mostly `positive ↔ negative`, often on short, sarcastic, or
  code-switched (Hausa/English) tweets where sentiment is implicit rather than stated — e.g.
  tweets using irony ("gaskiyar labarinnan" — "that's the real story", flat/ambiguous tone) or
  religious/idiomatic phrases whose sentiment is culturally inferred rather than lexical.

## Limitations

- **Test-split label gap:** the public AfriSenti-Hausa test set (as mirrored via MTEB)
  contains no `neutral`-labeled examples, making the raw 3-class test macro-F1 uninformative
  in isolation (see Results). We rely on the validation split for reporting.
- **Short tweet length** (median 11 words) limits context the model can use for implicit or
  sarcastic sentiment.
- **Single-run results:** training was run once per configuration; results may vary slightly
  with different random seeds.
- **Domain narrowness:** trained only on Twitter data, so performance on other Hausa text
  domains (news, formal writing) is untested.

## Acknowledgements / Resources Used

- Dataset: Muhammad, S.H. et al. (2023). *AfriSenti: A Twitter Sentiment Analysis Benchmark
  for African Languages*. [arXiv:2302.08956](https://arxiv.org/abs/2302.08956).
- Pretrained models: Alabi, J.O. et al. (2022). *Adapting Pre-trained Language Models to
  African Languages via Multilingual Adaptive Fine-Tuning*. `Davlan/afro-xlmr-base` on
  Hugging Face. Ogueji, K. et al. (2021). *Small Data? No Problem! Exploring the Viability of
  Pretrained Multilingual Language Models for Low-resourced Languages*. `castorini/afriberta_large`
  on Hugging Face.
- Libraries: Hugging Face `transformers`, `datasets`; `scikit-learn`; `gradio`.

## Links

- **GitHub Repository:** this repository
- **Deployed Web App:** **[TODO]**
- **Demo Video:** **[TODO]**
- **Full Report (PDF):** **[TODO]**
