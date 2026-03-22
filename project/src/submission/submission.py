"""Hackathon submission entry: ensemble cascade guardrail.

Delegates to `submission_ensemble_cascade` (word filter + mmBERT P(high_risk) +
LLM judge score → logistic meta-learner, risk-averse threshold).

Requires under `project/models/`:
  - `mmbert_guardrail_demo/` (fine-tuned classifier)
  - `ensemble_cascade_meta.json` (optional; overrides embedded coefs in `submission_ensemble_cascade`)
"""

from __future__ import annotations

from src.submission.submission_ensemble_cascade import get_guardrails

__all__ = ["get_guardrails"]
