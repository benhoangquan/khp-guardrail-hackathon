# Neurodiversity-aware guardrail for youth mental-health chat

Input guardrail that triages chat messages from youth as `low_risk` or `high_risk`, built by Team 055 (UdeM AI) for the Mila x Bell x Kids Help Phone safety hackathon, March 2026. It handles English and French, reads a whole conversation instead of the last message, and returns a reasoning trace for each decision.

Team: Anas El Ghoudane, Cassandre Hamel, Maryam Bani-Otero, Noah Tremblay Taillon, Quan Tran. The submission report is [`docs/report.pdf`](docs/report.pdf).

## Architecture

```
conversation ─┬─▶ multi-turn subsystem ──▶ score sC ─┐
              │     Drift Diffusion Model (w = 0.6)  ├─▶ blend 0.4·sC + 0.6·sL ─▶ low_risk / high_risk + trace
              │     conversation analyser (w = 0.4)  │
              └─▶ LLM judge (few-shot, C-SSRS-grounded examples) ─▶ score sL ┘
```

- **Drift Diffusion Model:** accumulates risk evidence across turns with recency weighting and neurodiversity-aware heuristic parameters (ADHD and ASD profiles).
- **Conversation analyser:** tracks eight longitudinal linguistic features across a conversation.
- **LLM judge:** few-shot prompting with retrieved clinically grounded examples, with negation-scope filtering and temporal-urgency detection.
- **Final submission:** `project/src/submission/submission.py` blends the multi-turn score with the LLM judge score through a logistic meta-learner.

| Path | Contents |
|---|---|
| `project/src/guardrails/` | Guardrail implementations (`ddm_model.py`, `cassandre_guardrail.py`, `llm_judge.py`, `convo_adapter.py`) |
| `project/src/submission/` | Submission variants, including the ensemble and cascade versions |
| `project/scripts/` | Feature builders and meta-learner training |
| `scripts/` | Stress-test generation and judging (`red_teaming/`), DEI data generation, classifier evaluation |
| `docs/` | The report, the team plan and the stress-test reproduction guide |

## Results

From the report, on the organizers' official test set: F1 0.850 (precision 0.871, recall 0.831) at 1,103 ms per sample, where LLM inference takes about 98.8% of the time. F1 was 0.880 in the earlier v2. On the team's stress suite (458 cases, 23 categories, 5 personas, 4 linguistic variants) the pass rate is 61.6%, 8.6 points above v2. In a 15-turn slow-burn test, the system escalates at turn 8.

## What this repository leaves out

This copy omits the organizers' datasets, documentation, dashboard material and credentials, and the team's generated and red-team datasets. The guardrails call models hosted in the hackathon's compute environment, which is closed, so the code cannot run end to end outside it. The provider classes in `project/providers/` read their endpoints and keys from environment variables.
