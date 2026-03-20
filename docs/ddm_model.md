# Drift Diffusion Model — `project/src/guardrails/ddm_model.py`

## What It Is

A pure-NumPy implementation of the Drift Diffusion Model (DDM) applied to multi-turn conversation risk accumulation. Takes a list of user turn strings, extracts a per-turn risk feature, simulates evidence accumulation with Euler-Maruyama integration, and returns a normalised risk score in `[0, 1]`.

**Weight in composite guardrail: 20%**

---

## Theory

### The Drift Diffusion Model

The DDM is a continuous-time model of binary decision-making from cognitive psychology (Ratcliff & McKoon, 2008). It describes how noisy evidence is accumulated toward a decision boundary:

```
dE = v · φ(t) · dt + σ · dW
```

| Symbol | Meaning |
|--------|---------|
| `E` | Evidence accumulator |
| `v` | Drift rate, which is how strongly each turn's features push the accumulator |
| `φ(t)` | Risk feature at turn `t` (extracted from user text) |
| `σ` | Noise (variability in evidence accumulation per turn) |
| `dW` | Wiener increment ~ N(0, dt) |
| `a` | Decision boundary or the threshold for `high_risk` classification |
| `z` | Starting point the prior bias toward high_risk before any turns |

### Euler-Maruyama Discretisation

With `dt = 1 turn`:

```
E[t+1] = E[t] + v · φ(t) + σ * ε,    ε ~ N(0, 1)
```

An absorbing lower barrier at `E = 0` prevents the accumulator going negative (no negative evidence). The model crosses boundary `a` when enough risk evidence has built up hopefully this captures multi-turn escalation that single-turn classifiers cannot detect.

### Why DDM Rather Than a Recurrent Model

An LSTM or transformer over the turn sequence would also capture temporal dynamics. DDM is chosen because:
1. **No training data required** the KHP validation set is too small to train a recurrent model.
2. **Interpretable trajectory** the `E[t]` sequence is directly observable and reportable.
3. **Profile parameterisation** DDM parameters have direct cognitive interpretations (boundary conservatism, noise, prior bias) that map onto documented ADHD/ASD differences (and more).
4. **Deterministic** with a fixed seed, results are reproducible.

---

## Profile-Specific Parameters

Standard classifiers are trained on neurotypical crisis language and systematically under-detect risk in ADHD and ASD users. DDM addresses this by adjusting three parameters:

| Profile | Boundary `a` | Noise `σ` | Start `z` | Rationale |
|---------|-------------|-----------|-----------|-----------|
| Neurotypical | 0.70 | 0.10 | 0.05 | Calibrated baseline |
| ADHD | 0.60 | 0.20 | 0.05 | Lower boundary (faster decision); higher noise (variable expression) [Karalunas 2012] |
| ASD | 0.65 | 0.08 | 0.15 | Higher start z (pre-loaded prior for flat-affect under-detection); lower noise (literal, consistent expression) [South 2010] |

**ADHD rationale:** Impulsive disclosure style.
Users may state genuine crises briefly and matter-of-factly. A lower boundary means the accumulator crosses `a` faster. Higher noise reflects the greater variability in how ADHD users express the same underlying state across turns.

**ASD rationale:** "I have been planning to end my life for three weeks. I wanted to mention it."
Can tend to state flatly without affect vocabulary. This looks low-risk to all affect-based classifiers. A higher starting point `z=0.15` pre-loads the accumulator by 15% of the boundary before any text is processed, compensating for the systematic under-detection of flat-affect disclosures.

---

## Feature Extraction

`extract_turn_features(turn_text)` goes to `φ(t) ∈ [0, 1]`

Three components:

**1. Risk keyword density (primary)**

```python
risk_hits = count(keyword in text for keyword in _RISK_TERMS)
risk_score = min(risk_hits / (n_words / 5), 1.0)
```

Scale: 1 hit per 5 words = 1.0. Keywords cover explicit SI, passive SI / burden language, hopelessness / finality, and isolation.

**2. Hedge penalty (inverse)**

```python
hedge_hits = count(hedge in text for hedge in _HEDGE_TERMS)
hedge_penalty = max(0.0, 0.2 - hedge_hits * 0.05)
```

Zero hedges leads `+0.20` to φ. Direct language (no uncertainty markers) = more certain distress. At 4+ hedges, the penalty drops to 0.

**3. Brevity signal**

```python
brevity_signal = 0.10 if n_words <= 5 else 0.0
```

Very short turns (≤5 words) may signal withdrawal or complete disengagement. Adds a small constant to φ.

```
φ(t) = min(risk_score + hedge_penalty + brevity_signal, 1.0)
```

---

## Profile Detection

`detect_profile(user_turns)` returns `"neurotypical" | "adhd" | "asd"`

Priority order:

1. **Explicit self-disclosure** if the user says `"ADHD"`, `"autism"`, `"autistic"`, `"ASD"`, `"asperger"` direct assignment. Most reliable signal.

2. **ADHD heuristic** scores two signals:
   - Fragmentation density: `(…`, `—`, `?!`, `idk`, `lol`, `nvm`, `ugh`, `omg`) per turn
   - Turn length coefficient of variation: `std(lengths) / mean(lengths)` high variance = impulsive bursts
   - Both must exceed threshold to assign `adhd` (conservative; prevents false assignments)

3. **ASD heuristic** requires both:
   - High explicit self-disclosure (`I have been…`, `for X weeks`, `I planned`, `specifically`, `precisely`, `exactly`)
   - Low affect vocabulary (`feel`, `scared`, `hopeless`, `worthless`, `hurt`, etc.)
   - Pattern: factual disclosure without emotional framing

4. **Default** `"neurotypical"` if no profile is confidently detected.

The profile detection is intentionally conservative the boundary shift is ≤0.10, so a wrong assignment has bounded downside.
Also inheriently this is a flawed profiling, we cannot directly assume anything from the user.

---

## Public API

```python
from project.src.guardrails.ddm_model import ddm_score

result = ddm_score(
    user_turns=["I've been feeling down.", "I don't see the point anymore."],
    profile_name=None,   # auto-detect; or "neurotypical" | "adhd" | "asd"
    seed=42,             # optional; None = stochastic
)

result["score"]          # float in [0, 1] — plug into composite guardrail
result["crossed"]        # bool — did accumulator hit boundary mid-conversation?
result["crossing_turn"]  # int or None — turn index of first boundary crossing
result["profile"]        # "neurotypical" | "adhd" | "asd"
result["phi_sequence"]   # list[float] — per-turn feature values
result["trajectory"]     # list[float] — full E[t] accumulator path
```

The `trajectory` and `phi_sequence` fields are useful for report figures and debugging. `interp_viz.py` can plot the DDM trajectory if needed.

---

## Sources

1. **Ratcliff & McKoon (2008)** Foundational DDM paper. Defines v, a, z, σ and Wiener process formalism.
   https://doi.org/10.1162/neco.2008.12-06-420

2. **Ratcliff et al. (2016)** DDM review; boundary and noise parameter interpretation.
   https://doi.org/10.1016/j.tics.2016.01.007

3. **Wiecki, Sofer & Frank (2013)** HDDM Python DDM; Euler-Maruyama discretisation.
   https://doi.org/10.3389/fninf.2013.00014

4. **Karalunas et al. (2012)** ADHD-specific DDM parameters: lower boundary, higher noise.
   https://doi.org/10.1037/a0029936

5. **South et al. (2010) / Brock et al. (2008)** ASD starting-point bias, flat-affect under-detection.
