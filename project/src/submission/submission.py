"""Ensemble v6 — Two LLM calls + Cassandre blend.

Real eval showed LLM-only drops to F1=0.750 (Cassandre IS needed for recall).
Baseline blend (W=0.4/0.6, T=0.40) got F1=0.850. This version:
  1. Calls LLM twice with complementary biases (cautious + precise)
  2. Averages their scores for a more robust LLM signal
  3. Blends with Cassandre (same weights as baseline)
  4. Provider fallback: OpenAI → Cohere → Mistral
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

from src.guardrails.cassandre_guardrail import CassandreMultiTurnGuardrail
from src.guardrails.llm_judge import LLMJudgeGuardrail
from src.guardrails.base import (
    BaseGuardrail,
    EvaluationType,
    GuardrailConfig,
    GuardrailResult,
    GuardrailStatus,
)

LOGGER = logging.getLogger(__name__)


class EnsembleGuardrail(BaseGuardrail):
    """
    Two-LLM ensemble blended with Cassandre.
    Cautious call catches more cases, precise call reduces false positives.
    Their averaged score is blended with Cassandre for final decision.
    """

    def __init__(
        self,
        cassandre_gr: CassandreMultiTurnGuardrail,
        llm_cautious: Optional[LLMJudgeGuardrail],
        llm_precise: Optional[LLMJudgeGuardrail],
        threshold: float = 0.40,
    ):
        super().__init__(GuardrailConfig(
            name="ensemble_v6",
            description="Two-LLM ensemble + Cassandre blend",
            threshold=threshold,
        ))
        self.cassandre = cassandre_gr
        self.llm_cautious = llm_cautious
        self.llm_precise = llm_precise

    def evaluate(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        evaluation_type: EvaluationType = EvaluationType.USER_INPUT,
    ) -> GuardrailResult:
        # 1. Cassandre (always runs — needed for recall boost + fallback)
        cass_result = self.cassandre.evaluate(content, context, evaluation_type)
        cass_score = cass_result.score if cass_result.score is not None else (
            1.0 if cass_result.status == GuardrailStatus.FAIL else 0.0
        )

        # 2. LLM calls
        cautious_result = None
        precise_result = None
        cautious_score = cass_score  # fallback
        precise_score = cass_score   # fallback

        if self.llm_cautious:
            cautious_result = self.llm_cautious.evaluate(
                content, context, evaluation_type
            )
            cautious_score = cautious_result.score if cautious_result.score is not None else (
                1.0 if cautious_result.status == GuardrailStatus.FAIL else 0.0
            )

        if self.llm_precise:
            precise_result = self.llm_precise.evaluate(
                content, context, evaluation_type
            )
            precise_score = precise_result.score if precise_result.score is not None else (
                1.0 if precise_result.status == GuardrailStatus.FAIL else 0.0
            )

        # 3. Ensemble LLM score (average of both calls)
        ensemble_llm = (cautious_score + precise_score) / 2.0

        # 4. Blend with Cassandre (same weights as baseline that got F1=0.850)
        W_CASS = 0.4
        W_LLM = 0.6
        grand_score = (cass_score * W_CASS) + (ensemble_llm * W_LLM)

        # 5. Confidence override — if BOTH LLMs agree high risk with high confidence
        if cautious_score >= 0.85 and precise_score >= 0.85:
            grand_score = ensemble_llm  # bypass Cassandre dilution

        status = (
            GuardrailStatus.FAIL if grand_score >= self.config.threshold
            else GuardrailStatus.PASS
        )

        reasoning = (
            f"Ensemble | Cautious: {cautious_score:.2f} | "
            f"Precise: {precise_score:.2f} | "
            f"Ensemble LLM: {ensemble_llm:.2f} | "
            f"Cassandre: {cass_score:.2f} | "
            f"Grand: {grand_score:.2f}"
        )

        metadata = {
            "cassandre_result": cass_result.to_dict(),
            "grand_score": grand_score,
            "ensemble_llm_score": ensemble_llm,
            "architecture": "ensemble_v6",
        }
        if cautious_result is not None:
            metadata["cautious_result"] = cautious_result.to_dict()
        if precise_result is not None:
            metadata["precise_result"] = precise_result.to_dict()

        return self._create_result(
            status=status,
            score=grand_score,
            reasoning=reasoning,
            metadata=metadata,
        )


def get_guardrails() -> Tuple[Optional[Any], Optional[Any]]:
    """Return (input_guardrail, output_guardrail)."""
    LOGGER.info("Loading Ensemble v6 (two-LLM + Cassandre blend)")

    # ── 1. Apply French keyword patch ─────────────────────────────────────
    import src.guardrails.fr_patch  # noqa: F401

    # ── 2. Create LLM provider ────────────────────────────────────────────
    llm_provider = None

    try:
        from providers.openai_provider import OpenAIProvider
        if os.getenv("BUZZ_GPT_OSS_AUTH_TOKEN"):
            llm_provider = OpenAIProvider()
            LOGGER.info("Using OpenAIProvider")
    except Exception as exc:
        LOGGER.warning("OpenAIProvider unavailable: %s", exc)

    if llm_provider is None:
        try:
            from providers.cohere_provider import CohereProvider
            if os.getenv("BUZZ_COHERE_AUTH_TOKEN"):
                llm_provider = CohereProvider()
                LOGGER.info("Using CohereProvider")
        except Exception as exc:
            LOGGER.warning("CohereProvider unavailable: %s", exc)

    if llm_provider is None:
        try:
            from providers.mistral_provider import MistralProvider
            if os.getenv("BUZZ_MISTRAL_LARGE_AUTH_TOKEN"):
                llm_provider = MistralProvider(
                    model=os.getenv("MISTRAL_MODEL"),
                    base_url=os.getenv("BUZZ_MISTRAL_LARGE_API"),
                    api_key=os.getenv("BUZZ_MISTRAL_LARGE_AUTH_TOKEN"),
                )
                LOGGER.info("Using MistralProvider")
        except Exception as exc:
            LOGGER.warning("MistralProvider unavailable: %s", exc)

    if llm_provider is None:
        LOGGER.warning("No LLM provider — will use Cassandre fallback only")

    # ── 3. Build components ───────────────────────────────────────────────
    cassandre_gr = CassandreMultiTurnGuardrail(config=GuardrailConfig(
        name="cassandre_base",
        description="Recall boost + fallback",
        threshold=0.5,
    ))

    llm_cautious = None
    llm_precise = None

    if llm_provider is not None:
        try:
            from src.prompt_templates.few_shot_guardrail_prompt import (
                CAUTIOUS_PROMPT,
                PRECISE_PROMPT,
            )

            llm_cautious = LLMJudgeGuardrail(
                config=GuardrailConfig(
                    name="llm_cautious",
                    description="High-recall bias",
                    threshold=0.5,
                ),
                llm_provider=llm_provider,
                user_input_prompt=CAUTIOUS_PROMPT,
            )

            llm_precise = LLMJudgeGuardrail(
                config=GuardrailConfig(
                    name="llm_precise",
                    description="High-precision bias",
                    threshold=0.5,
                ),
                llm_provider=llm_provider,
                user_input_prompt=PRECISE_PROMPT,
            )

            LOGGER.info("Ensemble loaded: cautious + precise LLM judges")
        except Exception as exc:
            LOGGER.error("Failed to initialize ensemble: %s", exc)

    # ── 4. Build ensemble ─────────────────────────────────────────────────
    input_guardrail = EnsembleGuardrail(
        cassandre_gr, llm_cautious, llm_precise, threshold=0.40
    )

    return (input_guardrail, None)
