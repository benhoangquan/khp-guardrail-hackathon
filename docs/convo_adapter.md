# ConvoKit Adapter
@ `project/src/guardrails/convo_adapter.py`

## What It Is

A conversation-level risk feature extractor for KHP multi-turn conversations. It wraps a `ConvoAnalyzer` class that takes a raw conversation string and returns a risk score in `[0, 1]`. Two entry points:

| Method | Dependency | Use |
|--------|-----------|-----|
| `analyzer.score(text)` | None (pure Python) | Production path in `submission.py` |
| `analyzer.score_with_convokit(text)` | `convokit` installed | Adds linguistic coordination feature at 15% |

**Weight in composite guardrail: 20%**

---

## Why It Exists

Single-turn classifiers (mental-bert, j-hartmann, goEmotions) process each user message in isolation. They miss:

- **Escalation trajectory**
ie: risk signal increasing across turns even when no single turn is alarming
- **Hedge word erosion**
ie: early hedging ("maybe", "i guess") that disappears in later turns as certainty increases
- **Engagement collapse**
ie: turn length dropping sharply (withdrawal signal)
- **First-to-last delta**
ie: a low-risk opening that ends in a high-risk turn

These are temporal, multi-turn signals. ConvoAnalyzer computes all of them with a sliding window over the user turn sequence.

ConvoKit is used as: a conversation data structure via `Corpus` / `Utterance` / `Speaker`, and an optional `Coordination` transformer for linguistic style mirroring.

---

## Features and Weights

_Note: feel free to change_

```python
WEIGHTS = {
    "escalation_trajectory": 0.35,   # slope of risk density across turns
    "final_turn_risk":        0.25,   # risk density of last user turn
    "hedge_erosion":          0.15,   # early hedge rate - late hedge rate
    "engagement_collapse":    0.15,   # turn length drop early → late
    "first_last_delta":       0.10,   # last turn risk - first turn risk
}
```

### escalation_trajectory (35%)

Linear regression slope of per-turn risk densities, normalised to `[0, 1]`:

```
slope = Σ(i - x̄)(r_i - ȳ) / Σ(i - x̄)^2
score = clip(slope * 5, 0, 1)
```

A conversation that steadily escalates from low to high risk will have a positive slope. Capped at 1.0 at slope = 0.2 (i.e., 20% risk density increase per turn).

### final_turn_risk (25%)

Risk keyword density of the **last user turn only**. Direct signal: the most recent thing the user said carries the most temporal weight.

### hedge_erosion (15%)

```
early_hedge = mean hedge density of first half of turns
late_hedge  = mean hedge density of second half of turns
erosion     = max(0, early_hedge - late_hedge)
```

Hedge words (`maybe`, `i guess`, `kind of`, `sometimes`, etc.) appear early when users are uncertain or testing the bot. Their disappearance in later turns correlates with resolved crisis intent. A value of 0 means hedging stayed constant or increased.

### engagement_collapse (15%)

```
early_len = mean word count of first 1/3 of turns
late_len  = mean word count of last 1/3 of turns
drop      = clip((early - late) / early, 0, 1)
```

Short final turns indicate withdrawal. This fires on genuine disengagement (the user has given up on the conversation) or extreme distress (can't form full sentences).

### first_last_delta (10%)

```
delta = max(0, risk_density(last_turn) - risk_density(first_turn))
```

Captures the case where a user opens with general conversation and ends with explicit crisis language
This is kind of pattern that single-turn analysis of any individual turn would not flag.

---

## Parsing

`_parse_turns(text)` handles two formats:

**Prefixed format (primary):**
```
user: I've been feeling really down lately.
assistant: I'm sorry to hear that. Can you tell me more?
user: I just don't see the point anymore.
```

**Unlabelled alternating (fallback):**
```
I've been feeling really down lately.
I'm sorry to hear that.
I just don't see the point anymore.
```

The regex pattern handles `user:`, `assistant:`, `bot:`, `human:` prefixes with any amount of whitespace after the colon, and multi-line turn content.

---

## ConvoKit Coordination Feature

When ConvoKit is installed, `score_with_convokit()` builds a `Corpus` object from the parsed turns and applies `Coordination()`:

```python
base = self.analyze(conversation_text)
blended = base["score"] * 0.85 + user_coord * 0.15
```

**Coordination** (Danescu-Niculescu-Mizil et al., 2012) measures how much the user's language style adapts to match the bot's. High coordination from a user leads to dependency / power asymmetry signal.

In a crisis context, a user who begins mirroring the bot's vocabulary may be seeking validation or forming a dependent relationship this could be a risk factor for over-reliance on the virtual assistant.

---

## How to Use

```python
from project.src.guardrails.convo_adapter import ConvoAnalyzer

analyzer = ConvoAnalyzer()

# Pure Python — no external deps
score = analyzer.score(conversation_text)  # float in [0, 1]

# Full feature dict
result = analyzer.analyze(conversation_text)
# result["score"], result["features"]["escalation_trajectory"], etc.

# With ConvoKit coordination
score = analyzer.score_with_convokit(conversation_text)
```

Input format: raw conversation text. Either prefixed (`user:` / `assistant:`) or unlabelled alternating lines.

---

## Installation

```bash
pip install convokit spacy
python -m spacy download en_core_web_sm
```

ConvoKit requires spaCy for its internal NLP pipeline. Without it, `score()` still works Note : the ConvoKit import only runs inside `score_with_convokit()`.

---

## Sources

- **ConvoKit** Conversation analysis toolkit:
  https://convokit.cornell.edu/
  Chang et al. (2020) "ConvoKit: A Toolkit for the Analysis of Conversations."
  https://aclanthology.org/2020.sigdial-1.8/

- **Linguistic Coordination** The coordination model implemented in ConvoKit:
  Danescu-Niculescu-Mizil et al. (2012) "Echoes of Power: Language Effects and Power Differences in Social Interaction."
  https://arxiv.org/abs/1112.3670

- **Hedge erosion as crisis signal** Pennebaker (2011) "The Secret Life of Pronouns" linguistic certainty markers and their relation to psychological state.

- **Engagement collapse / turn length as disengagement signal** Colasante et al. (2020) reduced linguistic output as withdrawal indicator in adolescent crisis conversations.
