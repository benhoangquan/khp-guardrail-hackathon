# Interpretability Visualizations
@ `scripts/interp_viz.py`

## What It Is

A visualization module that generates four types of classifier interpretability output. It reads from the JSON/CSV produced by `eval_classifiers.py` and saves PNGs to `results/interp_viz/`.

It is a **reporting and analysis tool** Its outputs are used for:
- Understanding what each classifier learned to attend to
- Identifying false negative failure modes
- Report figures (red-team findings section)

---

## Four Visualization Methods

### 1. LOO Token Heatmap (`--method loo`)

**Input:** `results/classifier_eval/attribution_{key}.json` (output of `eval_classifiers.py --mode interp`)

**Output:** One PNG per example, e.g. `loo_mental_bert_00_high_risk_FN.png`

Each token is rendered as a colored `FancyBboxPatch` rectangle. Color encodes importance:
- **Red** : token increases risk score (risk-driving)
- **Blue** : token suppresses risk score (hedges, neutral context)
- **White** : negligible effect

The full token sequence is laid out in reading order, 10 tokens per row. Each box shows the token and its ±importance value. A three-patch legend is rendered at the bottom-left.

**Color mapping:**

```python
def _importance_to_rgba(importance, vmax=0.15):
    norm = clip(importance / vmax, -1.0, 1.0)
    if norm >= 0:
        r, g, b = 1.0, 1.0 - norm*0.85, 1.0 - norm*0.85   # white → red
    else:
        r, g, b = 1.0 + norm*0.85, 1.0 + norm*0.85, 1.0   # white → blue
```

Text inside each box is black or white depending on background luminance (WCAG formula: `0.299r + 0.587g + 0.114b`).

**Limitation:** LOO uses whitespace tokenisation (word-level), not subword BPE. This means `"burden"` stays as `"burden"` rather than splitting into `"bur"` + `"##den"`. This is intentional for readability — exact attributions per BPE subword are less interpretable than word-level attributions.

---

### 2. Captum Integrated Gradients (`--method captum`)

**Input:** Texts specified via `--captum-texts` (defaults to 5 hard examples covering burden language, passive death wish, ASD flat-affect, and French)

**Output:** `ig_{key}_{n}.png` + `captum_ig_{key}.json`

**What it does:**

Integrated Gradients (Sundararajan et al., 2017) computes attribution by integrating the gradient of the model's output with respect to its input embeddings along a straight-line path from a baseline to the actual input:

```
IG_i(x) = (x_i - x'_i) × ∫[α=0→1] (∂F(x' + α(x-x')) / ∂x_i) dα
```

Where:
- `x` = actual input embedding matrix (seq_len × embed_dim)
- `x'` = baseline embedding (all `[PAD]` tokens → zero-ish vectors)
- `F(·)` = model output = `P(high_risk)` = `softmax(logits)[:, 1]`
- `∂F/∂x_i` = gradient of P(high_risk) w.r.t. embedding dimension `i`

The integral is approximated with Riemann sum over `n_steps=50` interpolation points.

**Aggregation to per-token scalars:**

```python
# attributions shape: (1, seq_len, embed_dim)
token_attr = attributions.squeeze(0).norm(dim=-1)  # L2 norm across embed_dim
```

L2 norm aggregates the attribution across all embedding dimensions into a single non-negative scalar per token. This is the standard aggregation method for IG on transformer models.

**Why IG rather than LOO?**

| | LOO | Integrated Gradients |
|--|-----|---------------------|
| Works on | Any pipeline | White-box PyTorch models only |
| Token granularity | Word-level (readability) | Subword BPE (exact) |
| Distribution shift | Yes — `[MASK]` is OOD | No — gradients are exact |
| Completeness axiom | Not guaranteed | Guaranteed: Σ IG_i = F(x) - F(x') |
| Speed | Slow (N forward passes) | Fast (one backward pass per step) |

IG satisfies the **completeness** axiom: the sum of all token attributions equals exactly `F(x) - F(baseline)`. LOO has no such guarantee.

**Forward function wrapper:**

```python
def _forward_fn_wrapper(model):
    def fn(inputs_embeds, attention_mask):
        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=-1)
        return probs[:, 1]  # P(high_risk)
    return fn
```

Captum requires a function that takes tensors and returns a scalar or 1D output. This wrapper passes `inputs_embeds` directly (bypassing the embedding lookup, since we provide the embeddings explicitly) and returns `P(high_risk)`.

**Convergence delta:** `ig.attribute(..., return_convergence_delta=True)` returns the approximation error. A delta close to 0 confirms the Riemann sum converged; a large delta suggests `n_steps` needs to increase.

---

### 3. Coverage Matrix (`--method matrix`)

**Input:** `results/classifier_eval/contrastive_pairs_scores.csv`

**Output:** `coverage_matrix.png`

A seaborn heatmap: rows = classifiers, columns = signal dimensions from the contrastive pairs run, cell values = `delta = hr_score − lr_score`.

```python
sns.heatmap(matrix, cmap="RdYlGn", center=0.0, vmin=-0.3, vmax=0.5, annot=True, fmt=".2f")
```

- **Green** (high delta) classifier correctly separates the pair by a wide margin
- **Yellow** (near zero) marginal separation; unreliable
- **Red** (negative or zero) classifier scores the high-risk variant lower than the low-risk variant (miss or inversion)

This is the most direct answer to "which classifiers cover which failure modes?" and serves as the key figure for the guardrail design section of the report.

---

### 4. Comparison Bar Chart (`--method bars`)

**Input:** `results/classifier_eval/summary.csv`

**Output:** `classifier_comparison.png`

Grouped bar chart: three bars per classifier (F1 in blue, Precision in green, Recall in orange), sorted by F1 descending. A dashed horizontal line at 0.80 marks the target F1. Value labels appear on each bar.

Directly shows where each classifier sits relative to the submission target.

---

## How to Run

```bash
# All visualizations
python scripts/interp_viz.py

# Just LOO heatmaps for two classifiers
python scripts/interp_viz.py --method loo --classifiers mental_bert goEmotions

# Captum IG on custom texts
python scripts/interp_viz.py --method captum \
  --captum-model mental/mental-bert-base-uncased \
  --captum-texts "I wouldn't mind not waking up tomorrow" \
                 "Everyone would be better off without me"

# Coverage matrix + bars only
python scripts/interp_viz.py --method matrix
python scripts/interp_viz.py --method bars
```

**Dependency chain:**

```
contrastive_pairs.py  ->  eval_classifiers.py (pairs, validation, interp)  ->  interp_viz.py
```

All four interp_viz outputs require that eval_classifiers.py has already been run. LOO heatmaps specifically require `--mode interp` to have produced the attribution JSON files.

---

## Installation

```bash
pip install matplotlib seaborn captum
```

`captum` is only required for `--method captum`. All other methods use only matplotlib and seaborn.

`matplotlib.use("Agg")` is set at the top of the file — no display is required. All plots save to file only.

---

## Sources

- **Integrated Gradients** Sundararajan et al. (2017) "Axiomatic Attribution for Deep Networks."
  https://arxiv.org/abs/1703.01365
  — Defines the completeness, sensitivity, and implementation invariance axioms. The L2-norm aggregation across embedding dimensions follows standard practice from this paper.

- **Captum** Kokhlikyan et al. (2020) "Captum: A unified and generic model interpretability library for PyTorch."
  https://arxiv.org/abs/2009.07896
  https://captum.ai/

- **LOO / perturbation attribution** Ribeiro et al. (2016) "Why Should I Trust You?" (LIME).
  https://arxiv.org/abs/1602.04938

- **WCAG luminance formula** W3C Web Content Accessibility Guidelines 2.1, §1.4.3.
  https://www.w3.org/TR/WCAG21/#contrast-minimum
  — Used for `_contrasting_text()` to determine black vs. white text on colored backgrounds.

- **Seaborn** Waskom (2021) "seaborn: statistical data visualization."
  https://doi.org/10.21105/joss.03021
