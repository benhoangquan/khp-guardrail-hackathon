"""Ensemble cascade guardrail: word filter + mmBERT probability + LLM judge → meta logistic score.

Implements a three-signal feature vector per message:

1. **Word filter** — binary 1.0 if any non-negated risk keyword from `convo_adapter` matches, else 0.0
   (optional normalized hit rate via `word_filter_mode="density"`).
2. **mmBERT** — P(high_risk) from the finetuned classifier guardrail (same as `load_classifier_guardrail`).
3. **LLM judge** — uses the judge's calibrated `score` in [0, 1]; if missing, falls back to 1.0/0.0 from FAIL/PASS.

**Meta-learner:** logistic layer ``p = sigmoid(w·x + b)`` with weights loaded from
``models/ensemble_cascade_meta.json`` (fit offline via ``project/scripts/train_ensemble_cascade_meta.py``).
If that file is absent, a fixed convex blend of the three features is used so the submission still runs.

**Threshold:** default 0.3 (risk-averse): FAIL when meta probability ≥ threshold.
"""

from __future__ import annotations

import json
import logging
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

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
from src.guardrails.classifier import load_classifier_guardrail
from src.guardrails.convo_adapter import _RISK_KEYWORDS, _is_negated
from src.guardrails.llm_judge import LLMJudgeGuardrail
from src.prompt_templates.few_shot_guardrail_prompt import FEW_SHOT_USER_INPUT_PROMPT
from src.submission._runtime_config import resolve_device_from_hackathon

LOGGER = logging.getLogger(__name__)

META_JSON_NAME = "ensemble_cascade_meta.json"
MMBERT_REL_PATH = Path("models") / "mmbert_guardrail_demo"

# Fallback blend when no trained meta file exists (weights sum to 1; emphasize learned signals).
_FALLBACK_WEIGHTS = (0.15, 0.45, 0.40)  # word_filter, mmbert, llm


def _sigmoid(z: float) -> float:
    z = max(-500.0, min(500.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def word_filter_feature(
    text: str,
    mode: Literal["binary", "density"] = "binary",
) -> float:
    """
    Risk keyword feature from `convo_adapter` patterns.
    - binary: 1.0 if any non-negated hit, else 0.0
    - density: min(1.0, hit_count / 5) for a smoother feature
    """
    if not text or not str(text).strip():
        return 0.0
    lower = text.lower()
    hits = 0
    for pattern in _RISK_KEYWORDS:
        m = re.search(pattern, lower)
        if m and not _is_negated(lower, m.start()):
            hits += 1
            if mode == "binary":
                return 1.0
    if mode == "binary":
        return 0.0
    return min(1.0, hits / 5.0)


def _load_meta_learner(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        LOGGER.exception("Failed to load meta learner JSON | path=%s", path)
        return None


def meta_probability(
    features: List[float],
    coef: Optional[List[float]],
    intercept: float,
) -> float:
    """Apply logistic meta-learner or fallback convex blend."""
    if len(features) != 3:
        raise ValueError(f"Expected 3 features, got {len(features)}")
    if coef is not None and len(coef) == 3:
        z = sum(c * x for c, x in zip(coef, features)) + intercept
        return _sigmoid(z)
    w0, w1, w2 = _FALLBACK_WEIGHTS
    return w0 * features[0] + w1 * features[1] + w2 * features[2]


class EnsembleCascadeGuardrail(BaseGuardrail):
    """
    Stack: [word_filter, mBERT P(high_risk), LLM score] → meta risk probability → threshold.
    """

    def __init__(
        self,
        mmbert_guardrail: BaseGuardrail,
        llm_judge: Optional[LLMJudgeGuardrail],
        meta_coef: Optional[List[float]],
        meta_intercept: float,
        threshold: float = 0.3,
        word_filter_mode: Literal["binary", "density"] = "binary",
    ):
        super().__init__(
            GuardrailConfig(
                name="ensemble_cascade_meta_logit",
                description="Word filter + mmBERT + LLM → logistic meta-learner (risk-averse threshold)",
                threshold=threshold,
            )
        )
        self.mmbert_guardrail = mmbert_guardrail
        self.llm_judge = llm_judge
        self.meta_coef = meta_coef
        self.meta_intercept = meta_intercept
        self.word_filter_mode = word_filter_mode

    def evaluate(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        evaluation_type: EvaluationType = EvaluationType.USER_INPUT,
    ) -> GuardrailResult:
        start = time.time()
        f_word = word_filter_feature(content, mode=self.word_filter_mode)

        m_out = self.mmbert_guardrail.evaluate(content, context, evaluation_type)
        f_mmbert = float(m_out.score) if m_out.score is not None else (
            1.0 if m_out.status == GuardrailStatus.FAIL else 0.0
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

        feats = [f_word, f_mmbert, f_llm]
        p_meta = meta_probability(feats, self.meta_coef, self.meta_intercept)
        status = GuardrailStatus.FAIL if p_meta >= self.config.threshold else GuardrailStatus.PASS

        latency_ms = (time.time() - start) * 1000.0
        reasoning = (
            f"meta_p={p_meta:.3f} | filter={f_word:.2f} mmbert={f_mmbert:.3f} llm={f_llm:.3f} "
            f"(threshold={self.config.threshold})"
        )
        return GuardrailResult(
            status=status,
            score=p_meta,
            reasoning=reasoning,
            metadata={
                "features": {
                    "word_filter": f_word,
                    "mmbert_prob": f_mmbert,
                    "llm_score": f_llm,
                },
                "mmbert_result": m_out.to_dict(),
                "llm_result": llm_dict,
                "meta_coef": self.meta_coef,
                "meta_intercept": self.meta_intercept,
                "used_trained_meta": self.meta_coef is not None,
            },
            latency_ms=latency_ms,
        )


def get_guardrails() -> Tuple[Optional[Any], Optional[Any]]:
    """Return (input_guardrail, output_guardrail). Requires finetuned mmBERT under models/."""
    LOGGER.info("Loading ensemble cascade (word filter + mmBERT + LLM meta-logit)")

    mmbert_dir = _PROJECT_ROOT / MMBERT_REL_PATH
    if not mmbert_dir.is_dir():
        LOGGER.warning(
            "Ensemble cascade: mmBERT directory missing | expected=%s",
            mmbert_dir,
        )
        return (None, None)

    device = resolve_device_from_hackathon(_PROJECT_ROOT)
    # High threshold so inner PASS/FAIL is irrelevant; we only read .score (P(high_risk)).
    mmbert_gr = load_classifier_guardrail(
        model_path=str(mmbert_dir),
        name="ensemble_inner_mmbert",
        description="Inner mmBERT for cascade features",
        threshold=1.01,
        device=device,
    )

    llm_judge: Optional[LLMJudgeGuardrail] = None
    try:
        llm_judge = LLMJudgeGuardrail(
            config=GuardrailConfig(
                name="ensemble_inner_llm_judge",
                description="Inner LLM judge for cascade features",
                threshold=0.5,
            ),
            llm_provider=OpenAIProvider(),
            user_input_prompt=FEW_SHOT_USER_INPUT_PROMPT,
        )
    except Exception as e:
        LOGGER.error("Ensemble cascade: LLM judge unavailable, using llm_score=0 | error=%s", e)

    meta_path = _PROJECT_ROOT / "models" / META_JSON_NAME
    meta = _load_meta_learner(meta_path)
    coef: Optional[List[float]] = None
    intercept = 0.0
    threshold = 0.3
    if meta:
        raw_coef = meta.get("coef") or meta.get("coefficients")
        if isinstance(raw_coef, list) and len(raw_coef) == 3:
            coef = [float(c) for c in raw_coef]
        intercept = float(meta.get("intercept", 0.0))
        threshold = float(meta.get("threshold", threshold))
        LOGGER.info(
            "Loaded meta learner | path=%s trained=%s threshold=%.4f",
            meta_path,
            coef is not None,
            threshold,
        )
    else:
        LOGGER.warning(
            "No meta learner at %s — using fallback convex blend %s",
            meta_path,
            _FALLBACK_WEIGHTS,
        )

    input_guardrail = EnsembleCascadeGuardrail(
        mmbert_guardrail=mmbert_gr,
        llm_judge=llm_judge,
        meta_coef=coef,
        meta_intercept=intercept,
        threshold=threshold,
        word_filter_mode="binary",
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
        print("No guardrail (train mmBERT or fix paths).")
    else:
        for s in (
            "Can you share healthy coping strategies for stress?",
            "How do I hurt someone badly?",
        ):
            r = ing.evaluate(s, evaluation_type=ET.USER_INPUT)
            print(s)
            print(" ", r.status.value, r.score, r.reasoning)
