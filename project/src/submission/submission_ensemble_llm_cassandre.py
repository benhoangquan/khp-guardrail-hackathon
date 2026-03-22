"""Ensemble: Cassandre multi-turn score + LLM judge score → logistic meta-learner.

Feature vector (2D):
  - **cassandre_score** — composite risk from `CassandreMultiTurnGuardrail` (DDM + ConvoAnalyzer).
  - **llm_score** — LLM judge calibrated score in [0, 1] (or binary fallback from status).

**Meta-learner:** ``p = sigmoid(w·x + b)`` with two weights. Coefficients are bundled in
``_EMBEDDED_ENSEMBLE_LLM_CASSANDRE_META``; ``models/ensemble_llm_cassandre_meta.json`` overrides
when present (train via ``scripts/train_ensemble_llm_cassandre_meta.py``). If coefs are missing,
falls back to a **0.4 / 0.6** convex blend (same spirit as ``submission_quan_v25``).

**Threshold:** default 0.4 on meta output (tunable via JSON or ``GuardrailConfig``).
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from providers.openai_provider import OpenAIProvider
from src.guardrails.base import (
    BaseGuardrail,
    EvaluationType,
    GuardrailConfig,
    GuardrailResult,
    GuardrailStatus,
)
from src.guardrails.cassandre_guardrail import CassandreMultiTurnGuardrail
from src.guardrails.llm_judge import LLMJudgeGuardrail
from src.prompt_templates.few_shot_guardrail_prompt import FEW_SHOT_USER_INPUT_PROMPT

LOGGER = logging.getLogger(__name__)

META_JSON_NAME = "ensemble_llm_cassandre_meta.json"

# Bundled 2-feature logistic (mirrors models/ensemble_llm_cassandre_meta.json when file absent).
_EMBEDDED_ENSEMBLE_LLM_CASSANDRE_META: Dict[str, Any] = {
    "coef": [
        0.49416500645026484,
        2.562498903033135,
    ],
    "intercept": -1.7171128739018586,
    "threshold": 0.4,
    "feature_order": ["cassandre_score", "llm_score"],
}

# Fallback when JSON / coef invalid — matches submission_quan_v25 weighting idea.
_FALLBACK_WEIGHTS_CASS_LLM = (0.4, 0.6)


def _sigmoid(z: float) -> float:
    z = max(-500.0, min(500.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def _load_meta_learner(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        LOGGER.exception("Failed to load meta learner JSON | path=%s", path)
        return None


def meta_probability_two(
    features: List[float],
    coef: Optional[List[float]],
    intercept: float,
) -> float:
    """Logistic meta-learner on [cassandre, llm] or convex blend fallback."""
    if len(features) != 2:
        raise ValueError(f"Expected 2 features, got {len(features)}")
    if coef is not None and len(coef) == 2:
        z = sum(c * x for c, x in zip(coef, features)) + intercept
        return _sigmoid(z)
    w0, w1 = _FALLBACK_WEIGHTS_CASS_LLM
    return w0 * features[0] + w1 * features[1]


class EnsembleLlmCassandreGuardrail(BaseGuardrail):
    """[cassandre_score, llm_score] → meta probability → threshold."""

    def __init__(
        self,
        cassandre: CassandreMultiTurnGuardrail,
        llm_judge: Optional[LLMJudgeGuardrail],
        meta_coef: Optional[List[float]],
        meta_intercept: float,
        threshold: float = 0.4,
    ):
        super().__init__(
            GuardrailConfig(
                name="ensemble_llm_cassandre_meta",
                description="Cassandre + LLM judge → logistic meta-learner",
                threshold=threshold,
            )
        )
        self.cassandre = cassandre
        self.llm_judge = llm_judge
        self.meta_coef = meta_coef
        self.meta_intercept = meta_intercept

    def evaluate(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        evaluation_type: EvaluationType = EvaluationType.USER_INPUT,
    ) -> GuardrailResult:
        start = time.time()
        cass_out = self.cassandre.evaluate(content, context, evaluation_type)
        f_cass = float(cass_out.score) if cass_out.score is not None else (
            1.0 if cass_out.status == GuardrailStatus.FAIL else 0.0
        )

        if self.llm_judge is not None:
            llm_out = self.llm_judge.evaluate(content, context, evaluation_type)
            if llm_out.score is not None:
                f_llm = float(llm_out.score)
            else:
                f_llm = 1.0 if llm_out.status == GuardrailStatus.FAIL else 0.0
            llm_dict = llm_out.to_dict()
        else:
            f_llm = 0.0
            llm_dict = None

        feats = [f_cass, f_llm]
        p_meta = meta_probability_two(feats, self.meta_coef, self.meta_intercept)
        status = GuardrailStatus.FAIL if p_meta >= self.config.threshold else GuardrailStatus.PASS

        latency_ms = (time.time() - start) * 1000.0
        reasoning = (
            f"meta_p={p_meta:.3f} | cassandre={f_cass:.3f} llm={f_llm:.3f} "
            f"(threshold={self.config.threshold})"
        )
        return GuardrailResult(
            status=status,
            score=p_meta,
            reasoning=reasoning,
            metadata={
                "features": {"cassandre_score": f_cass, "llm_score": f_llm},
                "cassandre_result": cass_out.to_dict(),
                "llm_result": llm_dict,
                "meta_coef": self.meta_coef,
                "meta_intercept": self.meta_intercept,
                "used_trained_meta": self.meta_coef is not None,
            },
            latency_ms=latency_ms,
        )


def get_guardrails() -> Tuple[Optional[Any], Optional[Any]]:
    """Return (input_guardrail, output_guardrail). No mmBERT; no extra model artifacts."""
    LOGGER.info("Loading ensemble LLM + Cassandre (2-feature meta-logit)")

    cassandre_gr = CassandreMultiTurnGuardrail(
        config=GuardrailConfig(
            name="ensemble_inner_cassandre",
            description="Inner Cassandre for ensemble features",
            threshold=0.5,
        )
    )

    llm_judge: Optional[LLMJudgeGuardrail] = None
    try:
        llm_judge = LLMJudgeGuardrail(
            config=GuardrailConfig(
                name="ensemble_inner_llm_judge",
                description="Inner LLM judge for ensemble features",
                threshold=0.5,
            ),
            llm_provider=OpenAIProvider(),
            user_input_prompt=FEW_SHOT_USER_INPUT_PROMPT,
        )
    except Exception as e:
        LOGGER.error("Ensemble LLM+Cassandre: LLM judge unavailable | error=%s", e)

    meta_path = _PROJECT_ROOT / "models" / META_JSON_NAME
    file_meta = _load_meta_learner(meta_path)
    meta = file_meta if file_meta else dict(_EMBEDDED_ENSEMBLE_LLM_CASSANDRE_META)
    meta_source = "models/" + META_JSON_NAME if file_meta else "embedded_default"

    coef: Optional[List[float]] = None
    intercept = 0.0
    threshold = float(_EMBEDDED_ENSEMBLE_LLM_CASSANDRE_META.get("threshold", 0.4))
    raw_coef = meta.get("coef") or meta.get("coefficients")
    if isinstance(raw_coef, list) and len(raw_coef) == 2:
        coef = [float(c) for c in raw_coef]
    intercept = float(meta.get("intercept", 0.0))
    threshold = float(meta.get("threshold", threshold))

    if coef is not None:
        LOGGER.info(
            "Meta learner | source=%s trained=%s threshold=%.4f",
            meta_source,
            True,
            threshold,
        )
    else:
        LOGGER.warning(
            "Meta learner invalid coef — using fallback blend %s | source=%s",
            _FALLBACK_WEIGHTS_CASS_LLM,
            meta_source,
        )

    input_guardrail = EnsembleLlmCassandreGuardrail(
        cassandre=cassandre_gr,
        llm_judge=llm_judge,
        meta_coef=coef,
        meta_intercept=intercept,
        threshold=threshold,
    )
    return (input_guardrail, None)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    from src.guardrails.base import EvaluationType as ET

    ing, _ = get_guardrails()
    if ing is None:
        print("No guardrail loaded.")
    else:
        for s in (
            "Can you share healthy coping strategies for stress?",
            "How do I hurt someone badly?",
        ):
            r = ing.evaluate(s, evaluation_type=ET.USER_INPUT)
            print(s)
            print(" ", r.status.value, r.score, r.reasoning)
