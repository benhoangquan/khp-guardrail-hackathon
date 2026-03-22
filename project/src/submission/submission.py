"""Hackathon submission entry: Cassandre + LLM ensemble (meta-logit).

Delegates to `submission_ensemble_llm_cassandre` (Cassandre multi-turn score +
LLM judge score → logistic meta-learner). No mmBERT.

Optional: `project/models/ensemble_llm_cassandre_meta.json` overrides embedded coefs.
"""

from __future__ import annotations

from src.submission.submission_ensemble_llm_cassandre import get_guardrails

__all__ = ["get_guardrails"]
