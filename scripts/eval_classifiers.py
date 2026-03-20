"""
Classifier comparison + interpretability tool.

Runs multiple candidate classifiers against:
  - seed_validation_set.csv  (ground truth labels, F1 / precision / recall)
  - stress_test scenarios    (labeled single-turn inputs)
  - contrastive pairs        (no ground truth, checks delta between variants)

For each classifier, also runs token-masking attribution on the hardest
examples (false negatives + contrastive pair misses) so you can see exactly
WHAT each model learned to attend to.

Interpretability method: Leave-One-Out (LOO) token masking.
  - Tokenize input to N tokens
  - For each token, replace with [MASK] and re-score
  - Importance(token_i) = base_score − masked_score

TODO: Captum?

Usage:
  python scripts/eval_classifiers.py                     # all modes
  python scripts/eval_classifiers.py --mode validation   # validation set only
  python scripts/eval_classifiers.py --mode pairs        # contrastive pairs only
  python scripts/eval_classifiers.py --mode interp       # attribution only
  python scripts/eval_classifiers.py --classifiers mental_bert goEmotions
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
from transformers import pipeline

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VALIDATION_CSV = ROOT / "datasets" / "seed_validation_set.csv"
STRESS_CSV     = ROOT / "datasets" / "stress_test_scenarios_v1_labeled.csv"
PAIRS_CSV      = ROOT / "datasets" / "contrastive_pairs_results.csv"
OUTPUT_DIR     = ROOT / "results" / "classifier_eval"


# CLASSIFIER REGISTRY
# Each entry: model_id, how to map raw output → risk score in [0,1]
@dataclass
class ClassifierDef:
    key: str # name to use
    model_id: str # HuggingFace model id or local path
    description: str
    risk_fn: Callable[[list[dict]], float]  # raw pipeline output to risk score
    lang: str = "en" # "en" | "fr" | "multi"
    top_k: Optional[int] = None # pipeline top_k, None = all labels


def _binary_high_risk(preds: list[dict]) -> float:
    """
        For models with a label named 'high_risk' or LABEL_1.
    """
    for p in preds:
        label = (p.get("label") or "").lower()
        if label in ("high_risk", "label_1", "1"):
            return float(p["score"])
    # fallback: max score
    return max(float(p["score"]) for p in preds)


def _mental_bert(preds: list[dict]) -> float:
    return _binary_high_risk(preds)


def _hartmann_emotion(preds: list[dict]) -> float:
    """
        j-hartmann: 7 Ekman emotions, weighted risk proxy.
        Used in EmoAgent
    """
    scores = {p["label"].lower(): float(p["score"]) for p in preds}
    return min(
        scores.get("sadness", 0) * 0.50
        + scores.get("fear",    0) * 0.30
        + scores.get("disgust", 0) * 0.20,
        1.0,
    )


def _goEmotions(preds: list[dict]) -> float:
    """
        SamLowe/roberta-base-go_emotions: 28 labels to risk proxy.
        Targets grief, remorse, fear, nervousness, sadness, anger (at-self).
    """
    risk_labels = {
        "grief": 0.35, "remorse": 0.20, "fear": 0.20,
        "nervousness": 0.10, "sadness": 0.10, "disapproval": 0.05,
    }
    scores = {p["label"].lower(): float(p["score"]) for p in preds}
    return min(sum(scores.get(l, 0) * w for l, w in risk_labels.items()), 1.0)


def _multilingual_sentiment(preds: list[dict]) -> float:
    """
        lxyuan multilingual: positive/neutral/negative to risk from negative.
    """
    scores = {p["label"].lower(): float(p["score"]) for p in preds}
    return scores.get("negative", 0.0)


def _twitter_xlm(preds: list[dict]) -> float:
    """
        cardiffnlp/twitter-xlm: positive/neutral/negative.
    """
    scores = {p["label"].lower(): float(p["score"]) for p in preds}
    return scores.get("negative", 0.0)


CLASSIFIERS: dict[str, ClassifierDef] = {
    "mental_bert": ClassifierDef(
        key="mental_bert",
        model_id="mental/mental-bert-base-uncased",
        description="BERT trained on Reddit mental health communities (binary)",
        risk_fn=_mental_bert,
        top_k=None,
    ),
    "mental_roberta": ClassifierDef(
        key="mental_roberta",
        model_id="mental/mental-roberta-base",
        description="RoBERTa trained on Reddit mental health communities (binary)",
        risk_fn=_mental_bert,
        top_k=None,
    ),
    "hartmann": ClassifierDef(
        key="hartmann",
        model_id="j-hartmann/emotion-english-distilroberta-base",
        description="7 Ekman emotions : sadness+fear+disgust risk proxy",
        risk_fn=_hartmann_emotion,
        top_k=None,
    ),
    "goEmotions": ClassifierDef(
        key="goEmotions",
        model_id="SamLowe/roberta-base-go_emotions",
        description="28 GoEmotions : grief+remorse+fear+nervousness risk proxy",
        risk_fn=_goEmotions,
        top_k=None,
    ),
    "multilingual": ClassifierDef(
        key="multilingual",
        model_id="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
        description="Multilingual sentiment : negative score as risk proxy (EN+FR)",
        risk_fn=_multilingual_sentiment,
        lang="multi",
        top_k=None,
    ),
    "twitter_xlm": ClassifierDef(
        key="twitter_xlm",
        model_id="cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual",
        description="Twitter multilingual sentiment (better FR youth language coverage)",
        risk_fn=_twitter_xlm,
        lang="multi",
        top_k=None,
    ),
}


# PIPELINE LOADER + SCORING
def load_pipeline(cdef: ClassifierDef):
    print(f"  Loading {cdef.key} ({cdef.model_id})...")
    t0 = time.time()
    pipe = pipeline(
        "text-classification",
        model=cdef.model_id,
        tokenizer=cdef.model_id,
        top_k=None,
        device=-1,  # CPU
    )
    print(f"  Loaded in {time.time() - t0:.1f}s")
    return pipe


def score_text(pipe, cdef: ClassifierDef, text: str) -> float:
    """
        Run pipeline on text and return risk score via cdef.risk_fn.
    """
    text = (text or "").strip()
    if not text:
        return 0.0
    try:
        result = pipe(text, truncation=True, max_length=512)
        preds = result[0] if isinstance(result[0], list) else result
        return cdef.risk_fn(preds)
    except Exception as e:
        print(f"    Score error ({cdef.key}): {e}")
        return 0.0


# METRICS
def compute_metrics(scores: list[float], labels: list[int], threshold: float = 0.5) -> dict:
    preds = [1 if s >= threshold else 0 for s in scores]
    tp = sum(p == 1 and l == 1 for p, l in zip(preds, labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(preds, labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(preds, labels))
    tn = sum(p == 0 and l == 0 for p, l in zip(preds, labels))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "threshold": threshold,
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n": len(labels),
    }


def best_threshold(scores: list[float], labels: list[int]) -> tuple[float, dict]:
    """Grid search threshold maximising F1."""
    best_t, best_m = 0.5, {}
    best_f1 = -1.0
    for t in np.arange(0.20, 0.85, 0.05):
        m = compute_metrics(scores, labels, threshold=float(t))
        if m["f1"] > best_f1:
            best_f1, best_t, best_m = m["f1"], float(t), m
    return best_t, best_m


# VALIDATION SET
def eval_validation(cdefs: list[ClassifierDef]) -> list[dict]:
    """Score each classifier on seed_validation_set.csv."""
    import csv as _csv

    rows = []
    with open(VALIDATION_CSV, newline="", encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            rows.append(r)

    texts  = [r["Text"] for r in rows]
    labels = [1 if r["label"] == "high_risk" else 0 for r in rows]
    print(f"\nValidation set: {len(rows)} rows  |  high_risk={sum(labels)}, low_risk={len(labels)-sum(labels)}")

    summary = []
    for cdef in cdefs:
        pipe = load_pipeline(cdef)
        scores = []
        for i, text in enumerate(texts):
            s = score_text(pipe, cdef, text)
            scores.append(s)
            if i % 200 == 0:
                print(f"    {i}/{len(texts)}")

        threshold, metrics = best_threshold(scores, labels)
        metrics["classifier"] = cdef.key
        metrics["model_id"]   = cdef.model_id
        metrics["best_threshold"] = threshold
        summary.append(metrics)
        print(f"  {cdef.key:20s}  F1={metrics['f1']:.3f}  P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  t={threshold:.2f}")

        # Save per-classifier predictions
        out_path = OUTPUT_DIR / f"validation_{cdef.key}.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = _csv.DictWriter(f, fieldnames=["text", "true_label", "score", "pred"])
            writer.writeheader()
            for text, label, score in zip(texts, labels, scores):
                writer.writerow({
                    "text": text[:200],
                    "true_label": "high_risk" if label == 1 else "low_risk",
                    "score": round(score, 4),
                    "pred": "high_risk" if score >= threshold else "low_risk",
                })

    return summary


# STRESS TESTS
def eval_stress(cdefs: list[ClassifierDef]) -> list[dict]:
    """Score each classifier on labeled single-turn stress test scenarios."""
    import csv as _csv

    rows = []
    if not STRESS_CSV.exists():
        print(f"  Stress test CSV not found: {STRESS_CSV}")
        return []

    with open(STRESS_CSV, newline="", encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            rows.append(r)

    # label column: "PASS" (bot handled it) / "FAIL" / or from the category name
    # The input_column is the user message — that's what we classify
    texts    = [r["input_column"] for r in rows]
    # High-risk inputs are those from "Euphemism (High Risk)" category or labeled FAIL
    labels   = [
        1 if ("high risk" in r.get("category", "").lower() or r.get("label", "").upper() == "FAIL")
        else 0
        for r in rows
    ]
    categories = [r.get("category", "unknown") for r in rows]

    print(f"\nStress test set: {len(rows)} rows  |  high_risk={sum(labels)}")

    summary = []
    for cdef in cdefs:
        pipe = load_pipeline(cdef)
        scores = [score_text(pipe, cdef, t) for t in texts]
        threshold, metrics = best_threshold(scores, labels)
        metrics["classifier"] = cdef.key

        # Per-category breakdown
        unique_cats = sorted(set(categories))
        cat_recall = {}
        for cat in unique_cats:
            cat_idx = [i for i, c in enumerate(categories) if c == cat and labels[i] == 1]
            if not cat_idx:
                continue
            caught = sum(1 for i in cat_idx if scores[i] >= threshold)
            cat_recall[cat] = round(caught / len(cat_idx), 3)

        metrics["category_recall"] = cat_recall
        summary.append(metrics)
        print(f"  {cdef.key:20s}  F1={metrics['f1']:.3f}  R={metrics['recall']:.3f}")
        for cat, r in cat_recall.items():
            print(f"    {cat:35s}  recall={r:.2f}")

    return summary


# CONTRASTIVE PAIRS
def eval_pairs(cdefs: list[ClassifierDef]) -> list[dict]:
    """
    For each contrastive pair, check:
      - Does each classifier score high_risk > low_risk? (correct direction)
      - What is the delta? (confidence of separation)
      - Which classifiers MISS signals that KHP also misses? (double blind spot)
    """
    import csv as _csv

    if not PAIRS_CSV.exists():
        print(f"  Contrastive pairs CSV not found: {PAIRS_CSV}")
        print("  Run scripts/contrastive_pairs.py first.")
        return []

    pairs = []
    with open(PAIRS_CSV, newline="", encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            pairs.append(r)

    print(f"\nContrastive pairs: {len(pairs)}")

    all_results = []
    for cdef in cdefs:
        pipe = load_pipeline(cdef)
        results = []
        for pair in pairs:
            lr_score = score_text(pipe, cdef, pair["low_risk_input"])
            hr_score = score_text(pipe, cdef, pair["high_risk_input"])
            correct_direction = hr_score > lr_score
            delta = hr_score - lr_score

            # "miss" = classifier scores high_risk variant below 0.4
            classifier_misses = hr_score < 0.4
            khp_misses = pair.get("gap") == "khp_misses"

            results.append({
                "classifier":        cdef.key,
                "dimension":         pair["dimension"],
                "lr_score":          round(lr_score, 4),
                "hr_score":          round(hr_score, 4),
                "delta":             round(delta, 4),
                "correct_direction": correct_direction,
                "classifier_misses": classifier_misses,
                "khp_misses":        khp_misses,
                "double_blind_spot": classifier_misses and khp_misses,
            })

        n_correct = sum(r["correct_direction"] for r in results)
        n_doubles = sum(r["double_blind_spot"] for r in results)
        avg_delta = np.mean([r["delta"] for r in results])
        print(f"  {cdef.key:20s}  correct_direction={n_correct}/{len(results)}  "
              f"avg_delta={avg_delta:+.3f}  double_blind_spots={n_doubles}")

        for r in results:
            flag = " DOUBLE BLIND SPOT" if r["double_blind_spot"] else (
                   " classifier miss" if r["classifier_misses"] else "")
            print(f"    {r['dimension']:30s}  delta={r['delta']:+.3f}  "
                  f"lr={r['lr_score']:.3f}  hr={r['hr_score']:.3f}{flag}")

        all_results.extend(results)

    # Save
    out_path = OUTPUT_DIR / "contrastive_pairs_scores.csv"
    if all_results:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = _csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n  Saved → {out_path}")

    return all_results


# TOKEN-MASKING ATTRIBUTION (interpretability)
def _mask_token(tokens: list[str], idx: int, mask_token: str = "[MASK]") -> str:
    """Replace token at idx with mask_token and rejoin."""
    masked = tokens[:idx] + [mask_token] + tokens[idx + 1:]
    return " ".join(masked)


def token_attribution(pipe, cdef: ClassifierDef, text: str, top_n: int = 8) -> list[tuple[str, float]]:
    """
    Leave-One-Out token masking attribution.

    For each token: importance = base_score − score_with_token_masked
    Positive = token pushes toward high_risk
    Negative = token suppresses high_risk (hedge words typically negative)

    Returns top_n tokens by |importance|, sorted descending.
    """
    # Simple whitespace tokenisation — preserves readability
    tokens = text.split()
    if not tokens:
        return []

    base_score = score_text(pipe, cdef, text)
    importances = []

    for i, token in enumerate(tokens):
        masked_text = _mask_token(tokens, i)
        masked_score = score_text(pipe, cdef, masked_text)
        importance = base_score - masked_score  # positive = token increases risk
        importances.append((token, round(importance, 4)))

    # Sort by absolute importance, return top_n
    importances.sort(key=lambda x: abs(x[1]), reverse=True)
    return importances[:top_n]


def _bar(value: float, width: int = 20) -> str:
    """Simple ASCII bar for attribution display."""
    filled = int(abs(value) / 0.15 * width)  # 0.15 = full bar
    filled = min(filled, width)
    bar = "█" * filled + "░" * (width - filled)
    sign = "+" if value >= 0 else "-"
    return f"{sign}{bar}"


def print_attribution(tokens_importance: list[tuple[str, float]], text: str, label: str, score: float):
    print(f"\n  Text:  {text[:120]}")
    print(f"  Label: {label}   Score: {score:.3f}")
    print(f"  {'Token':20s}  {'Importance':10s}  Attribution")
    print(f"  {'─'*20}  {'─'*10}  {'─'*22}")
    for token, imp in tokens_importance:
        print(f"  {token:20s}  {imp:+.4f}      {_bar(imp)}")


def run_attribution(cdefs: list[ClassifierDef], n_examples: int = 5):
    """
    Run token attribution on the most informative examples:
    - False negatives from the validation set (highest-stakes misses)
    - High-risk contrastive inputs where the classifier had low delta
    """
    import csv as _csv

    examples = []

    # Pull false negatives from validation set
    for cdef in cdefs:
        pred_path = OUTPUT_DIR / f"validation_{cdef.key}.csv"
        if not pred_path.exists():
            continue
        with open(pred_path, newline="", encoding="utf-8") as f:
            for r in _csv.DictReader(f):
                if r["true_label"] == "high_risk" and r["pred"] == "low_risk":
                    examples.append({
                        "text": r["text"],
                        "label": "high_risk (FN)",
                        "classifier": cdef.key,
                        "score": float(r["score"]),
                    })
        # Limit to n_examples per classifier
        examples = [e for e in examples if e["classifier"] != cdef.key] + \
                   [e for e in examples if e["classifier"] == cdef.key][:n_examples]

    # Also include contrastive pair high-risk inputs where classifier missed
    if PAIRS_CSV.exists():
        with open(PAIRS_CSV, newline="", encoding="utf-8") as f:
            for r in _csv.DictReader(f):
                examples.append({
                    "text": r["high_risk_input"],
                    "label": f"high_risk pair [{r['dimension']}]",
                    "classifier": "all",
                    "score": None,
                })

    print(f"\nRunning attribution on {len(examples)} examples...")

    for cdef in cdefs:
        pipe = load_pipeline(cdef)
        print(f"\n{'═'*70}")
        print(f"  Classifier: {cdef.key} — {cdef.description}")
        print(f"{'═'*70}")

        cdef_examples = [e for e in examples if e["classifier"] in (cdef.key, "all")]
        for ex in cdef_examples[:n_examples]:
            score = ex["score"] if ex["score"] is not None else score_text(pipe, cdef, ex["text"])
            attribs = token_attribution(pipe, cdef, ex["text"])
            print_attribution(attribs, ex["text"], ex["label"], score)

        # Save attribution to JSON for notebook use
        out = []
        for ex in cdef_examples[:n_examples]:
            score = score_text(pipe, cdef, ex["text"])
            attribs = token_attribution(pipe, cdef, ex["text"])
            out.append({
                "text": ex["text"],
                "label": ex["label"],
                "score": round(score, 4),
                "attribution": attribs,
            })
        out_path = OUTPUT_DIR / f"attribution_{cdef.key}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Attribution saved → {out_path}")


# SUMMARY TABLE
def print_summary(validation_summary: list[dict]):
    if not validation_summary:
        return
    print(f"  CLASSIFIER COMPARISON SUMMARY")
    print(f"  {'Classifier':20s}  {'F1':>6}  {'Precision':>9}  {'Recall':>7}  {'Threshold':>9}")
    for m in sorted(validation_summary, key=lambda x: -x["f1"]):
        print(f"  {m['classifier']:20s}  {m['f1']:6.3f}  {m['precision']:9.3f}  "
              f"{m['recall']:7.3f}  {m['best_threshold']:9.2f}")

    # Recommendation
    best = max(validation_summary, key=lambda x: x["f1"])
    best_recall = max(validation_summary, key=lambda x: x["recall"])
    print(f"Best F1:     {best['classifier']} ({best['f1']:.3f})")
    print(f"Best Recall: {best_recall['classifier']} ({best_recall['recall']:.3f})")
    # For the composite guardrail, use best Recall as the primary component
    # missing high_risk is worse than over-flagging

    # Save summary
    out_path = OUTPUT_DIR / "summary.csv"
    if validation_summary:
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(validation_summary[0].keys()))
            writer.writeheader()
            writer.writerows(validation_summary)
        print(f"\n  Summary saved → {out_path}")


# CLI
def main():
    parser = argparse.ArgumentParser(description="Classifier evaluation + interpretability")
    parser.add_argument(
        "--mode",
        choices=["all", "validation", "stress", "pairs", "interp"],
        default="all",
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        choices=list(CLASSIFIERS.keys()),
        default=list(CLASSIFIERS.keys()),
        help="Which classifiers to evaluate (default: all)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fixed threshold (default: auto-tune per classifier)",
    )
    parser.add_argument(
        "--n-interp",
        type=int,
        default=5,
        help="Number of examples to run attribution on per classifier",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cdefs = [CLASSIFIERS[k] for k in args.classifiers]

    print(f"Evaluating {len(cdefs)} classifier(s): {[c.key for c in cdefs]}")

    validation_summary = []

    if args.mode in ("all", "validation"):
        if VALIDATION_CSV.exists():
            validation_summary = eval_validation(cdefs)
        else:
            print(f"Validation CSV not found: {VALIDATION_CSV}")

    if args.mode in ("all", "stress"):
        eval_stress(cdefs)

    if args.mode in ("all", "pairs"):
        eval_pairs(cdefs)

    if args.mode in ("all", "interp"):
        run_attribution(cdefs, n_examples=args.n_interp)

    if validation_summary:
        print_summary(validation_summary)


if __name__ == "__main__":
    main()
