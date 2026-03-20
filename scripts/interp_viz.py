"""
interp_viz.py — Interpretability visualizations for classifier evaluation.

Two methods, used together:

  1. LOO Token Heatmap (model-agnostic)
     Reads attribution JSON produced by eval_classifiers.py.
     Renders each token as a colored rectangle: red = increases risk score,
     blue = suppresses it. Saves one PNG per example.

  2. Captum Integrated Gradients (white-box, HuggingFace models only)
     Computes attribution at the subword token level by integrating gradients
     of the model's output w.r.t. its input embeddings along a straight-line
     path from a baseline (all-[PAD] sequence) to the actual input.
     This is more principled than LOO — it doesn't suffer from distribution
     shift when tokens are masked — and gives exact gradient-based attribution.

  3. Classifier × Dimension Coverage Matrix
     Heatmap showing which classifiers catch which signal dimensions from
     the contrastive pairs run.  Direct answer to "does mental-bert cover
     burden language?" etc.

  4. Classifier Comparison Bar Chart
     Side-by-side F1/Precision/Recall for all evaluated classifiers.

Usage:
  python scripts/interp_viz.py                      # all plots
  python scripts/interp_viz.py --method loo         # LOO heatmaps only
  python scripts/interp_viz.py --method captum      # Captum IG only
  python scripts/interp_viz.py --method matrix      # coverage matrix only
  python scripts/interp_viz.py --method bars        # comparison bars only
  python scripts/interp_viz.py --classifiers mental_bert goEmotions

References:
  Sundararajan et al. (2017) "Axiomatic Attribution for Deep Networks"
    https://arxiv.org/abs/1703.01365  — Integrated Gradients definition.
  Ribeiro et al. (2016) "Why Should I Trust You?" (LIME)
    https://arxiv.org/abs/1602.04938  — LOO / perturbation attribution.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display required — saves to file
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

# ── paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
EVAL_DIR   = ROOT / "results" / "classifier_eval"
OUTPUT_DIR = ROOT / "results" / "interp_viz"

CLASSIFIER_KEYS = [
    "mental_bert", "mental_roberta", "hartmann", "goEmotions",
    "multilingual", "twitter_xlm",
]


# ─────────────────────────────────────────────────────────────────────────────
# COLOUR HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _importance_to_rgba(importance: float, vmax: float = 0.15) -> tuple:
    """
    Map importance → RGBA colour.
    Positive (increases risk) → red scale
    Negative (suppresses risk) → blue scale
    Near-zero → white
    """
    norm = np.clip(importance / max(vmax, 1e-6), -1.0, 1.0)
    if norm >= 0:
        # white → red
        r, g, b = 1.0, 1.0 - norm * 0.85, 1.0 - norm * 0.85
    else:
        # white → blue
        r, g, b = 1.0 + norm * 0.85, 1.0 + norm * 0.85, 1.0
    return (r, g, b, 1.0)


def _contrasting_text(rgba: tuple) -> str:
    """Black or white text depending on background brightness."""
    r, g, b = rgba[:3]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if luminance > 0.55 else "white"


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOO TOKEN HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def plot_loo_heatmap(
    tokens: list[tuple[str, float]],
    text: str,
    label: str,
    score: float,
    classifier_key: str,
    save_path: Path,
    vmax: float = 0.15,
):
    """
    Render a horizontal token attribution heatmap.

    Each token is a coloured box. Colour intensity encodes importance:
      red  = this token pushes the model toward high_risk
      blue = this token suppresses high_risk (hedge words, neutral context)
      white = negligible effect

    The full token sequence is rendered in order, wrapping at ~10 tokens per row.
    """
    # Full token order from text (LOO JSON only stores top-N; reconstruct order)
    all_tokens = text.split()
    importance_map = {tok: imp for tok, imp in tokens}

    n_cols = 10
    n_rows = max(1, (len(all_tokens) + n_cols - 1) // n_cols)

    fig_w = max(12, n_cols * 1.4)
    fig_h = n_rows * 0.9 + 1.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows + 0.5)
    ax.axis("off")

    fig.suptitle(
        f"{classifier_key} — {label}   (risk score = {score:.3f})",
        fontsize=11, fontweight="bold", y=0.98,
    )

    for idx, token in enumerate(all_tokens):
        row = idx // n_cols
        col = idx %  n_cols
        y = n_rows - row - 1  # top row first
        importance = importance_map.get(token, 0.0)
        colour = _importance_to_rgba(importance, vmax=vmax)
        text_colour = _contrasting_text(colour)

        rect = mpatches.FancyBboxPatch(
            (col + 0.05, y + 0.10), 0.88, 0.72,
            boxstyle="round,pad=0.03",
            facecolor=colour, edgecolor="grey", linewidth=0.4,
        )
        ax.add_patch(rect)
        ax.text(
            col + 0.49, y + 0.46, token,
            ha="center", va="center",
            fontsize=9, color=text_colour, fontweight="normal",
        )
        # importance value below token
        ax.text(
            col + 0.49, y + 0.13, f"{importance:+.3f}",
            ha="center", va="center",
            fontsize=6.5, color="dimgray",
        )

    # Legend
    legend_x, legend_y = 0.01, -0.04
    for colour, label_txt in [
        (_importance_to_rgba(0.12), "increases risk"),
        (_importance_to_rgba(-0.12), "suppresses risk"),
        (_importance_to_rgba(0.01), "negligible"),
    ]:
        patch = mpatches.Patch(facecolor=colour, edgecolor="grey", label=label_txt)
    ax.legend(
        handles=[
            mpatches.Patch(facecolor=_importance_to_rgba(0.12), edgecolor="grey", label="increases risk"),
            mpatches.Patch(facecolor=_importance_to_rgba(-0.12), edgecolor="grey", label="suppresses risk"),
            mpatches.Patch(facecolor=_importance_to_rgba(0.01), edgecolor="grey", label="negligible"),
        ],
        loc="lower left", fontsize=8, framealpha=0.7,
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_path.name}")


def run_loo_heatmaps(classifier_keys: list[str]):
    """Generate LOO heatmaps from attribution JSON files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for key in classifier_keys:
        json_path = EVAL_DIR / f"attribution_{key}.json"
        if not json_path.exists():
            print(f"  No attribution file for {key} — run eval_classifiers.py --mode interp first")
            continue

        with open(json_path) as f:
            records = json.load(f)

        print(f"\n[LOO heatmaps] {key} — {len(records)} examples")
        for i, rec in enumerate(records):
            slug = rec["label"].replace(" ", "_").replace("/", "-")[:40]
            save_path = OUTPUT_DIR / f"loo_{key}_{i:02d}_{slug}.png"
            plot_loo_heatmap(
                tokens=rec["attribution"],
                text=rec["text"],
                label=rec["label"],
                score=rec["score"],
                classifier_key=key,
                save_path=save_path,
            )


# ─────────────────────────────────────────────────────────────────────────────
# 2. CAPTUM INTEGRATED GRADIENTS
# ─────────────────────────────────────────────────────────────────────────────

def _load_model_and_tokenizer(model_id: str):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()
    return model, tokenizer


def _forward_fn(model, inputs_embeds, attention_mask):
    """Forward pass returning logits for the high_risk class (index 1)."""
    import torch
    outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    # Softmax → P(high_risk) = index 1
    probs = torch.softmax(outputs.logits, dim=-1)
    return probs[:, 1]


def run_captum_ig(
    model_id: str,
    classifier_key: str,
    texts: list[str],
    labels: list[str],
    n_steps: int = 50,
):
    """
    Compute Integrated Gradients attribution for each text.

    IG approximates the path integral of gradients from a baseline (all-[PAD])
    to the actual input, attributing credit to each embedding dimension.
    We aggregate across embedding dimensions to get per-token scalar attribution.

    Ref: Sundararajan et al. (2017) https://arxiv.org/abs/1703.01365
    """
    import torch
    from captum.attr import IntegratedGradients

    print(f"\n[Captum IG] {classifier_key} — loading {model_id}")
    model, tokenizer = _load_model_and_tokenizer(model_id)
    ig = IntegratedGradients(_forward_fn_wrapper(model))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for text, label in zip(texts, labels):
        encoding = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding="max_length",
        )
        input_ids     = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # Embedding lookup
        embeddings = model.get_input_embeddings()(input_ids)

        # Baseline: all [PAD] tokens
        baseline_ids = torch.zeros_like(input_ids)
        baseline_emb = model.get_input_embeddings()(baseline_ids)

        # Compute IG
        attributions, delta = ig.attribute(
            embeddings,
            baselines=baseline_emb,
            additional_forward_args=(attention_mask,),
            n_steps=n_steps,
            return_convergence_delta=True,
            target=None,  # already returns scalar from _forward_fn
        )

        # Aggregate: L2 norm across embedding dim → scalar per token
        token_attr = attributions.squeeze(0).norm(dim=-1).detach().numpy()
        tokens_decoded = tokenizer.convert_ids_to_tokens(input_ids[0])

        # Remove [PAD], keep actual tokens
        actual = [
            (tok, float(attr))
            for tok, attr, mask in zip(tokens_decoded, token_attr, attention_mask[0])
            if mask == 1 and tok not in ("[PAD]", "<pad>")
        ]

        score = _forward_fn(model, embeddings, attention_mask).item()
        results.append({
            "text": text, "label": label,
            "score": round(score, 4),
            "attribution": actual,
            "convergence_delta": float(delta.abs().mean()),
        })

        save_path = OUTPUT_DIR / f"ig_{classifier_key}_{len(results):02d}.png"
        _plot_ig_heatmap(actual, text, label, score, classifier_key, save_path)

    out_json = OUTPUT_DIR / f"captum_ig_{classifier_key}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Captum IG results saved → {out_json}")
    return results


def _forward_fn_wrapper(model):
    """Return a function suitable for captum.attr.IntegratedGradients."""
    import torch
    def fn(inputs_embeds, attention_mask):
        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=-1)
        return probs[:, 1]
    return fn


def _plot_ig_heatmap(
    token_attrs: list[tuple[str, float]],
    text: str,
    label: str,
    score: float,
    classifier_key: str,
    save_path: Path,
):
    """
    IG attribution heatmap. Uses subword tokens (e.g. '##burden') directly.
    Colour convention same as LOO: red = increases risk, blue = suppresses.
    """
    if not token_attrs:
        return

    tokens  = [t for t, _ in token_attrs]
    attribs = np.array([a for _, a in token_attrs])
    vmax = max(attribs.max(), 1e-6)

    n_cols = 12
    n_rows = max(1, (len(tokens) + n_cols - 1) // n_cols)
    fig, ax = plt.subplots(figsize=(max(14, n_cols * 1.2), n_rows * 0.9 + 1.5))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows + 0.4)
    ax.axis("off")
    fig.suptitle(
        f"Captum IG — {classifier_key}\n{label}   score={score:.3f}",
        fontsize=10, fontweight="bold",
    )

    for idx, (token, attr) in enumerate(token_attrs):
        row = idx // n_cols
        col = idx %  n_cols
        y = n_rows - row - 1
        colour = _importance_to_rgba(attr, vmax=vmax)
        text_col = _contrasting_text(colour)
        rect = mpatches.FancyBboxPatch(
            (col + 0.04, y + 0.12), 0.90, 0.70,
            boxstyle="round,pad=0.02",
            facecolor=colour, edgecolor="lightgrey", linewidth=0.3,
        )
        ax.add_patch(rect)
        ax.text(col + 0.49, y + 0.47, token, ha="center", va="center",
                fontsize=8.5, color=text_col)
        ax.text(col + 0.49, y + 0.16, f"{attr:.3f}", ha="center", va="center",
                fontsize=6, color="dimgray")

    ax.legend(
        handles=[
            mpatches.Patch(facecolor=_importance_to_rgba(vmax * 0.8), edgecolor="grey",
                           label="increases risk (IG)"),
            mpatches.Patch(facecolor=_importance_to_rgba(-vmax * 0.8), edgecolor="grey",
                           label="suppresses risk (IG)"),
        ],
        loc="lower left", fontsize=8,
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. CLASSIFIER × DIMENSION COVERAGE MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def plot_coverage_matrix(classifier_keys: list[str]):
    """
    Heatmap: rows = classifiers, columns = signal dimensions from contrastive pairs.
    Cell value = delta (high_risk_score − low_risk_score) for that classifier × dimension.
    Green = large positive delta (classifier correctly separates the pair).
    Red = small/negative delta (classifier misses this signal).
    """
    import csv as _csv

    pairs_scores = EVAL_DIR / "contrastive_pairs_scores.csv"
    if not pairs_scores.exists():
        print(f"  No contrastive pairs scores found — run eval_classifiers.py --mode pairs first")
        return

    rows = []
    with open(pairs_scores, newline="") as f:
        for r in _csv.DictReader(f):
            rows.append(r)

    # Build matrix
    dims = sorted(set(r["dimension"] for r in rows))
    keys = [k for k in classifier_keys if any(r["classifier"] == k for r in rows)]
    if not keys:
        print("  No matching classifiers in contrastive pairs scores.")
        return

    matrix = np.zeros((len(keys), len(dims)))
    for r in rows:
        if r["classifier"] not in keys:
            continue
        i = keys.index(r["classifier"])
        j = dims.index(r["dimension"])
        matrix[i, j] = float(r["delta"])

    fig, ax = plt.subplots(figsize=(max(10, len(dims) * 1.4), max(4, len(keys) * 0.8 + 1.5)))
    sns.heatmap(
        matrix,
        xticklabels=[d.replace("_", "\n") for d in dims],
        yticklabels=keys,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0.0,
        vmin=-0.3,
        vmax=0.5,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "delta (hr_score − lr_score)"},
    )
    ax.set_title(
        "Classifier Coverage by Signal Dimension\n"
        "(green = correctly separates pair, red = misses signal)",
        fontsize=11, pad=12,
    )
    ax.set_xlabel("Signal Dimension", fontsize=9)
    ax.set_ylabel("Classifier", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=9)

    save_path = OUTPUT_DIR / "coverage_matrix.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nCoverage matrix saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. CLASSIFIER COMPARISON BAR CHART
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison_bars(classifier_keys: list[str]):
    """Grouped bar chart: F1 / Precision / Recall per classifier."""
    import csv as _csv

    summary_csv = EVAL_DIR / "summary.csv"
    if not summary_csv.exists():
        print("  No summary.csv found — run eval_classifiers.py --mode validation first")
        return

    rows = []
    with open(summary_csv, newline="") as f:
        for r in _csv.DictReader(f):
            if r["classifier"] in classifier_keys:
                rows.append(r)

    if not rows:
        print("  No matching classifiers in summary.csv")
        return

    rows.sort(key=lambda r: float(r["f1"]), reverse=True)
    labels   = [r["classifier"] for r in rows]
    f1s      = [float(r["f1"])        for r in rows]
    precisions = [float(r["precision"]) for r in rows]
    recalls  = [float(r["recall"])    for r in rows]

    x = np.arange(len(labels))
    width = 0.26

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.6), 5))
    bars_f1 = ax.bar(x - width, f1s,        width, label="F1",        color="#2196F3", alpha=0.85)
    bars_p  = ax.bar(x,         precisions, width, label="Precision",  color="#4CAF50", alpha=0.85)
    bars_r  = ax.bar(x + width, recalls,    width, label="Recall",     color="#FF5722", alpha=0.85)

    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Classifier Comparison on seed_validation_set.csv\n(sorted by F1)", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.axhline(0.8, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(len(labels) - 0.5, 0.81, "F1=0.80 target", fontsize=7.5, color="grey")
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Value labels on bars
    for bar in [*bars_f1, *bars_p, *bars_r]:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=7)

    save_path = OUTPUT_DIR / "classifier_comparison.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nComparison bar chart saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Interpretability visualizations")
    parser.add_argument(
        "--method",
        choices=["all", "loo", "captum", "matrix", "bars"],
        default="all",
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        default=CLASSIFIER_KEYS,
    )
    # For captum mode: specify which texts to analyze
    parser.add_argument(
        "--captum-texts",
        nargs="+",
        default=[
            "I'm just so tired of being a burden to my family. They'd be better off without me.",
            "I wouldn't mind if I just didn't wake up tomorrow. I don't really care anymore.",
            "I have been planning to end my life for the past three weeks.",  # ASD flat-affect
            "everyone would be better off if I just wasn't around anymore",
            "je ne vois plus l'utilité de continuer. tout le monde serait mieux sans moi",
        ],
        help="Texts to run Captum IG on",
    )
    parser.add_argument(
        "--captum-model",
        default="mental/mental-bert-base-uncased",
        help="Model ID to use for Captum IG (must be binary classifier)",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.method in ("all", "loo"):
        print("\n── LOO Token Heatmaps ──────────────────────────")
        run_loo_heatmaps(args.classifiers)

    if args.method in ("all", "captum"):
        print("\n── Captum Integrated Gradients ─────────────────")
        try:
            labels = [f"example_{i}" for i in range(len(args.captum_texts))]
            run_captum_ig(
                model_id=args.captum_model,
                classifier_key=Path(args.captum_model).name,
                texts=args.captum_texts,
                labels=labels,
            )
        except ImportError:
            print("  captum not installed — skipping IG. pip install captum")
        except Exception as e:
            print(f"  Captum IG failed: {e}")

    if args.method in ("all", "matrix"):
        print("\n── Coverage Matrix ──────────────────────────────")
        plot_coverage_matrix(args.classifiers)

    if args.method in ("all", "bars"):
        print("\n── Comparison Bar Chart ─────────────────────────")
        plot_comparison_bars(args.classifiers)

    print(f"\nAll outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
