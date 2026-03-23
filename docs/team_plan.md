# Team 055 — Technical Plan (revised March 20)

## Hackathon Context

Build an **input guardrail** for the Kids Help Phone (KHP) virtual assistant that classifies youth mental health conversations as `low_risk` or `high_risk`.

**Deadline:** March 22, 20:00. Code must be on `main` in `project/src/submission/submission.py`.

### Primary failure modes to address

- Missed passive suicidal ideation: burden language, quiet resignation, hopelessness
- Single-turn analysis that misses risk building across a multi-turn conversation
- Affect-literal detection that misses indirect or coded signals
- Under-sensitivity to neurodivergent communication styles (ADHD/ASD users express distress differently)
- French implicit distress signals

---

## What Has Been Built

### Red Team

| File | Purpose |
|------|---------|
| `scripts/run_drift_tests.py` | Live multi-turn personas against KHP, full transcript logging |
| `scripts/batch_runner.py/v2` | Bulk scenario testing -> CSV |
| `scripts/auto_judge_csv.py/v2` | Nemotron judges each KHP response as PASS/FAIL against policy |
| `scripts/multi_turn_red_team.py` | Escalating adversarial conversations |
| `scripts/delta_reporter.py` | Behavior drift detection between test runs |
| `scripts/generate_test_matrix.py/v2` | Test scenario matrix generation |
| `scripts/contrastive_pairs.py` | Minimal pair probing which maps KHP's decision boundary per signal dimension |
| `triage.py` | Interactive review of drift results |
| `datasets/stress_test_*.csv/json` | Generated stress test data and drift results |
| `tools/EmoAgent/critic_agent.py` | EmoGuard: per-turn policy auditor (Nemotron backend) |
| `tools/EmoAgent/llm_backend.py` | Nemotron drop-in backend for EmoAgent (replaces OpenAI) |
| `docs/eeyore_setup.md` | Eeyore deployment guide for red-team data generation |

### Blue Team (partially built)

| File | Purpose | Status |
|------|---------|--------|
| `project/src/guardrails/convo_adapter.py` | Multi-turn conversation analysis (ConvoKit + pure Python) | Done |
| `project/src/guardrails/ddm_model.py` | DDM evidence accumulation, profile detection, feature extraction | Done |
| `project/src/guardrails/classifier.py` | HuggingFace classifier loader and runner | Exists (organizer-provided) |
| `project/src/submission/submission.py` | **THE DELIVERABLE** composite guardrail | Not started |

---

## Revised Architecture

```
Input: full conversation text (multi-turn user/assistant)
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│              Composite Input Guardrail                   │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  EmoAgent    │  │    Eeyore    │  │     DDM      │  │
│  │  j-hartmann  │  │  mental-bert │  │  ddm_model   │  │
│  │  (7 emotions │  │  (mental     │  │  (NumPy,     │  │
│  │  → risk map) │  │   health     │  │   no PyDDM)  │  │
│  │              │  │   BERT)      │  │              │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                 │                  │          │
│         └─────────────────┴──────────────────┘          │
│                           │                             │
│               ┌───────────┴──────────┐                  │
│               │    ConvoAnalyzer     │                  │
│               │  convo_adapter.py    │                  │
│               └───────────┬──────────┘                  │
│                           │                             │
│               ┌───────────▼──────────┐                  │
│               │   Score Aggregator   │                  │
│               │   weighted average   │                  │
│               │   → PASS / FAIL      │                  │
│               └──────────────────────┘                  │
└─────────────────────────────────────────────────────────┘
```

**No LLM judge in the prediction path.** All inference is local classifiers or pure Python/NumPy. Keeps latency low and results deterministic.

---

## Components

### 1. EmoAgent Emotion Distress Scoring

**Model:** `j-hartmann/emotion-english-distilroberta-base`
Fine-tuned on 6 emotion datasets. Returns 7 Ekman emotion scores.

**Critical note:** This is NOT a binary classifier. It cannot be loaded directly via `load_classifier_guardrail()`. Requires a custom wrapper that maps the 7 emotion scores to a risk scalar:
```python
risk = sadness * 0.50 + fear * 0.30 + disgust * 0.20
```

**French fallback:** `lxyuan/distilbert-base-multilingual-cased-sentiments-student`
Activated when language detection identifies French input.

**Weight in aggregator:** 25%

---

### 2. Eeyore Passive Hopelessness Detector

**Naming clarification:** "Eeyore" in this composite guardrail refers to the `mental/mental-bert-base-uncased` HuggingFace classifier — a BERT model trained on Reddit mental health communities (r/depression, r/SuicideWatch, etc.). It is NOT the Llama 3.1 8B Eeyore simulation model in `tools/Eeyore/`.

**Before wiring, verify output labels:**
```python
from transformers import pipeline
pipe = pipeline("text-classification", model="mental/mental-bert-base-uncased", top_k=None)
print(pipe("I feel like a burden to everyone"))
# Check label names — may need label remapping
```

**Targets:**
- Burden-to-others ideation
- Passive death wish
- Quiet fatalism and resignation
- Finality language

**Weight in aggregator:** 35% (highest — passive SI is the hardest-to-catch failure mode)

---

### 3. DDM Cognitive Profile-Aware Accumulation

**Implementation:** `project/src/guardrails/ddm_model.py` (no PyDDM dependency)

**PyDDM was dropped.** PyDDM is designed for fitting reaction-time distributions from psychology experiments. The simulation itself — Euler-Maruyama integration over text turn features is 8 lines of NumPy and does not need the library. PyDDM would only add value if we had trial-level RT data per profile, which we don't.

**What it does:**
1. Extracts a risk feature φ(t) from each user turn (keyword density, hedging, brevity)
2. Accumulates evidence across turns: `E[t+1] = E[t] + v·φ[t] + σ·ε`
3. Uses profile-specific parameters (ADHD: lower boundary; ASD: higher starting point)
4. Returns normalised score + full trajectory (for report figures)

**Profile parameters:**

| Profile | Boundary `a` | Noise `σ` | Start `z` | Rationale |
|---------|-------------|-----------|-----------|-----------|
| Neurotypical | 0.70 | 0.10 | 0.05 | Default |
| ADHD | 0.60 | 0.20 | 0.05 | Lower boundary (impulsive expression); higher noise [Karalunas et al. 2012] |
| ASD | 0.65 | 0.08 | 0.15 | Higher start (flat affect under-detection); lower noise [South et al. 2010] |

**Weight in aggregator:** 20%

---

### 4. ConvoAnalyzer Multi-Turn Trajectory

**Implementation:** `project/src/guardrails/convo_adapter.py`

**Features extracted:**
- Escalation trajectory (risk slope across turns) — 35% of component score
- Final turn risk density 25%
- Hedge word erosion (early vs late turns) 15%
- Engagement collapse (turn length drop) 15%
- First-vs-last turn delta 10%

**ConvoKit integration:** `score_with_convokit()` adds linguistic coordination (style mirroring) at 15% blend. Falls back to pure-Python `score()` if ConvoKit is not installed.

**Weight in aggregator:** 20%

---

### 5. Score Aggregation

```python
final_score = (
    emo_score   * 0.25 +
    eeyore_score * 0.35 +
    ddm_score   * 0.20 +
    convo_score  * 0.20
)
status = GuardrailStatus.FAIL if final_score >= threshold else GuardrailStatus.PASS
```

Starting threshold: **0.45** (lower than 0.5 to bias toward recall on high_risk).
Tune against `datasets/seed_validation_set.csv` before final submission.

---

## What Was Dropped / Changed

| Original Plan | Revised Plan | Reason |
|--------------|-------------|--------|
| PyDDM library | NumPy DDM in `ddm_model.py` | PyDDM fits RT distributions; we only need forward simulation. 8 lines of NumPy covers it. |
| EmoAgent via `load_classifier_guardrail()` | Custom emotion-to-risk wrapper | j-hartmann outputs 7 scores, not binary. Needs custom mapping. |
| "Eeyore" = Llama 3.1 8B model | "Eeyore" = mental-bert-base-uncased | The Llama model is a text generator (user simulator), not a classifier. mental-bert IS a binary classifier. |
| EmoEval full pipeline | Not used | Hardwired to Character.AI + OpenAI. Requires >1 day of adaptation. |
| ConvoKit as feature extractor | ConvoKit as data structure; features computed manually | Built-in ConvoKit extractors don't cover the specific signals needed. KHP format adapter written in `convo_adapter.py`. |
| Captum mech interp in prediction path | Report-only / post-hoc | No time; use Captum on false negatives after evaluation for the report analysis section. |

---

## Dependencies

Add to `pyproject.toml`:
```toml
convokit>=3.0.0     # conversation structure + coordination
spacy>=3.7.0        # required by convokit
```

Removed from plan: `pyddm`, `captum` (report-only, not in prediction path).

---

## File Changes Required

| File | Change |
|------|--------|
| `project/src/submission/submission.py` | **Replace** with CompositeGuardrail wiring all 4 components |
| `pyproject.toml` | Add `convokit`, `spacy`; remove `pyddm` |
| `hackathon.json` | Set `needs_gpu: true` when Eeyore data generation is confirmed; add HF artifact entry |

Already done:
- `project/src/guardrails/convo_adapter.py` ConvoKit adapter + trajectory features
- `project/src/guardrails/ddm_model.py`DDM simulation, profile detection, feature extraction
- `tools/EmoAgent/llm_backend.py` Nemotron backend for EmoAgent
- `tools/EmoAgent/critic_agent.py` Nemotron swap (EmoGuard)
- `tools/EmoAgent/dialog_manager.py` I would Nemotron swap

---

## Actionable Steps (ordered by priority)

### Today — March 20

**1. Verify mental-bert output labels** (15 min)
```python
from transformers import pipeline
pipe = pipeline("text-classification", model="mental/mental-bert-base-uncased", top_k=None)
print(pipe("I feel like a burden to everyone"))
```
Confirms whether `load_classifier_guardrail()` can be used directly or needs label remapping.

**2. Write `submission.py` composite guardrail** (3–4 hours)
Wire all four components. This is the only file that gets evaluated.

**3. Run `predict.sh` + `evaluate.sh` on `seed_validation_set.csv`** (30 min)
```bash
./project/scripts/predict.sh datasets/seed_validation_set.csv results/predictions.csv
./project/scripts/evaluate.sh results/predictions.csv results/metrics.csv
```

**4. Tune threshold** (1 hour)
Grid search over threshold 0.35–0.65 and per-component weights. Optimise for recall on `high_risk` while keeping precision above 0.6.

**5. Run contrastive pairs in background** (30 min setup, runs overnight)
```bash
python scripts/contrastive_pairs.py
```
Results feed directly into the report's red-team findings section.

---

### March 21

**6. Review false negatives from evaluation** (1 hour)
Pull all false negatives from `results/predictions.csv`. Annotate by failure type:
- Burden language?
- French?
- Multi-turn trajectory?
- ASD flat affect?

These annotations are your strongest report evidence.

**7. Run Nemotron EmoGuard over existing stress test transcripts** (2 hours)
Point the swapped `CriticAgent` at `datasets/stress_test_drift_results_v1.json`.
Output: structured violation records for the red-team findings section.

**8. Write report** (4 hours)
Sections map directly to what exists:
- Red team findings → contrastive pairs results + EmoGuard violation records
- Guardrail design → this document's architecture section
- Data → seed_validation_set stats + stress test coverage
- Results → F1 / precision / recall / latency from evaluation
- DEI coverage → French fallback + DDM profile parameters + ASD pair result

**9. Push to `main`** before writing a single word of the report.

---

### March 22 (buffer)

- Final evaluation run to confirm F1
- Any last threshold adjustment
- Report polish
- Submission by 20:00

---

## Risk Register

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| mental-bert labels don't map to high_risk/low_risk | Medium | Remap manually or swap to `mental/mental-roberta-base` |
| j-hartmann not available on BUZZ cluster | Medium | Pre-download at configure time; declare in `hackathon.json` |
| ConvoKit install fails | Low | `convo_adapter.py` `score()` method works without ConvoKit |
| F1 insufficient after tuning | Medium | Add Nemotron LLM judge as 5th stacked layer (architecture already supports it) |
| French recall poor | Medium | Verify multilingual fallback activates correctly via language detection |
| Eeyore GPU not available | Low-Medium | Data generation is offline/optional; red-team report uses existing stress test data |
