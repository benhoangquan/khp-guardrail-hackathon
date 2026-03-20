# Classifier Evaluation Pipeline — `scripts/eval_classifiers.py`

## What It Is

A multi-mode evaluation harness for the six candidate HuggingFace classifiers considered for the composite guardrail. It evaluates each classifier against three data sources and produces per-token attribution output for interpretability.

**This is the tool that decides which classifiers go into `submission.py` and at what weights.**

---

## Why It Exists

The composite guardrail requires knowing:
1. Which classifier has the best recall on `high_risk` (lowest false negative rate)?
2. Which classifiers miss specific signal dimensions (burden language, ASD flat-affect, French)?
3. Where does each classifier's decision land — which tokens are actually driving the score?

Without this, the weights in `submission.py` are arbitrary. `eval_classifiers.py` answers all three questions empirically before the composite is built.

---

## Classifier Registry

Six models, each with a custom `risk_fn` that maps raw pipeline output to a risk score in `[0, 1]`:

### `mental_bert` : `mental/mental-bert-base-uncased`
BERT fine-tuned on Reddit mental health communities (r/depression, r/SuicideWatch, r/Anxiety, r/bipolar). Binary output. `risk_fn` extracts the `high_risk` or `LABEL_1` probability directly.

**Why:** Domain-specific training on the exact communities most similar to KHP's user base. Highest expected recall on explicit SI.

### `mental_roberta` : `mental/mental-roberta-base`
Same training setup as mental_bert but on RoBERTa backbone. RoBERTa removes NSP pre-training task and uses larger batches; typically outperforms BERT on classification. Same `risk_fn`.

**Why:** Candidate for replacing mental_bert if it has better recall without sacrificing precision.

### `hartmann` : `j-hartmann/emotion-english-distilroberta-base`
Fine-tuned on 6 emotion datasets. Returns 7 Ekman emotion scores (anger, disgust, fear, joy, neutral, sadness, surprise). **Not a binary classifier.** Requires custom mapping:

```python
risk = sadness * 0.50 + fear * 0.30 + disgust * 0.20
```

Weights are based on psychological literature linking emotion types to suicidality: sadness most predictive, fear (terror) secondary, disgust (including self-disgust) tertiary.

**Why:** Captures emotional tone without needing domain-specific training. Useful for detecting passive hopelessness that doesn't use clinical vocabulary.

### `goEmotions` : `SamLowe/roberta-base-go_emotions`
Fine-tuned on Google's GoEmotions dataset: 28 fine-grained emotion labels. Risk mapping:

```python
risk_labels = {
    "grief": 0.35, "remorse": 0.20, "fear": 0.20,
    "nervousness": 0.10, "sadness": 0.10, "disapproval": 0.05,
}
```

**Why:** `grief` and `remorse` labels capture passive SI signals that hartmann misses. `grief` = unresolved loss → passive death wish pathway. `remorse` = guilt/burden ideation. The 28-label resolution is superior to 7-label for nuanced mental health signals.

### `multilingual` : `lxyuan/distilbert-base-multilingual-cased-sentiments-student`
DistilBERT multilingual, fine-tuned on sentiment. Covers French. `risk_fn` = negative sentiment probability.

**Why:** French fallback for bilingual KHP users. The French pair in contrastive_pairs.py tests whether this activates for French implicit distress.

### `twitter_xlm` : `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual`
XLM-RoBERTa fine-tuned on multilingual Twitter data. Better coverage of youth informal language and French slang.

**Why:** Youth crisis language on chat platforms is closer to Twitter register than clinical text. This model's training domain is the best match for KHP's user population among the multilingual options.

---

## Evaluation Modes

### Mode 1: Validation (`--mode validation`)

Runs all classifiers against `datasets/seed_validation_set.csv`. For each classifier:

1. Scores every text in the validation set
2. Grid-searches threshold from 0.20 to 0.85 in 0.05 steps to maximise F1
3. Reports F1, precision, recall at best threshold
4. Saves per-example predictions to `results/classifier_eval/validation_{key}.csv`

Output: `results/classifier_eval/summary.csv`

```
Classifier          F1      Precision  Recall   Threshold
mental_bert         0.823   0.791      0.857    0.45
mental_roberta      0.831   0.802      0.862    0.45
goEmotions          0.764   0.724      0.809    0.30
...
```

**Priority metric: Recall.** Missing a `high_risk` conversation (false negative) is far more dangerous than over-flagging a `low_risk` one (false positive). The summary output explicitly names the best-recall classifier.

### Mode 2: Stress Tests (`--mode stress`)

Loads `datasets/stress_test_scenarios_v1_labeled.csv` and computes per-category recall:

```
mental_bert         F1=0.811  R=0.846
  Burden-to-others ideation     recall=0.72
  ASD flat-affect disclosure    recall=0.43   ← gap
  French implicit distress      recall=0.31   ← gap
  Explicit suicidal ideation    recall=0.94
```

Category breakdown reveals which signal types each classifier is blind to. These gaps directly inform which signal dimensions need extra weight in the composite.

### Mode 3: Contrastive Pairs (`--mode pairs`)

Loads `datasets/contrastive_pairs_results.csv` (output of `contrastive_pairs.py`) and scores both variants of each pair:

```python
delta = hr_score - lr_score
correct_direction = hr_score > lr_score
double_blind_spot = (hr_score < 0.4) and (gap == "khp_misses")
```

`double_blind_spot = True` means **both KHP and this classifier miss this signal**. These are the highest-priority gaps for the composite guardrail design. Output: `results/classifier_eval/contrastive_pairs_scores.csv`.

### Mode 4: Attribution (`--mode interp`)

Runs Leave-One-Out (LOO) token masking on false negatives and contrastive pair high-risk inputs. See [LOO methodology](#leave-one-out-attribution).

---

## Leave-One-Out Attribution

**Method:** For each token in the input, replace it with `[MASK]` and re-run the classifier:

```
importance(token_i) = base_score − score(text_with_token_i_masked)
```

- Positive importance means token increases risk score (risk-driving tokens)
- Negative importance means token suppresses risk score (hedge words, neutral context)
- Near-zero means token has negligible effect

## LOO
LOO is model-agnostic, it works with any HuggingFace pipeline regardless of architecture. It requires no access to model weights or gradients. The tradeoff is distribution shift: replacing a token with `[MASK]` creates an OOD input the model wasn't trained on. For gradient-based attribution on white-box models, see `interp_viz.py` which implements Captum Integrated Gradients.

**Output:** `results/classifier_eval/attribution_{key}.json`

```json
[
  {
    "text": "I'm just so tired of being a burden...",
    "label": "high_risk (FN)",
    "score": 0.312,
    "attribution": [["burden", 0.087], ["tired", 0.041], ["just", -0.012], ...]
  }
]
```

This JSON is the input consumed by `interp_viz.py` for heatmap rendering.

---

## How to Run

```bash
# Full evaluation (all modes, all classifiers) — takes ~30 min on CPU
python scripts/eval_classifiers.py

# Fast: just validation set, two classifiers
python scripts/eval_classifiers.py --mode validation --classifiers mental_bert goEmotions

# Interpretability only (requires validation mode to have run first)
python scripts/eval_classifiers.py --mode interp --n-interp 10

# Stress tests only
python scripts/eval_classifiers.py --mode stress
```

Output directory: `results/classifier_eval/`

---

## Sources

- **mental/mental-bert** and **mental/mental-roberta**: Ji et al. (2022) "MentalBERT: Publicly Available Pretrained Language Models for Mental Healthcare."
  https://arxiv.org/abs/2110.01301

- **j-hartmann/emotion-english-distilroberta**: Hartmann et al. (2022) "More than a Feeling: Benchmarks for Sentiment Analysis Accuracy."
  https://arxiv.org/abs/2209.12440

- **SamLowe/roberta-base-go_emotions**: Demszky et al. (2020) "GoEmotions: A Dataset of Fine-Grained Emotions."
  https://arxiv.org/abs/2005.00547

- **cardiffnlp/twitter-xlm-roberta**: Barbieri et al. (2022) "XLM-T: Multilingual Language Models in Twitter for Sentiment Analysis."
  https://arxiv.org/abs/2104.12250

- **LOO attribution** Ribeiro et al. (2016) "Why Should I Trust You? Explaining the Predictions of Any Classifier" (LIME).
  https://arxiv.org/abs/1602.04938
