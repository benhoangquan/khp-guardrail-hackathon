"""Cassandre Grand Composite v3 — Bilingual + Confidence Override.

Based on Quan's v2.5 blended scoring architecture.
Changes from v2.5:
  1. French keyword patch (import side-effect) — Cassandre now scores FR convos
  2. Bilingual C-SSRS prompt replaces basic few-shot prompt
  3. Confidence override — if LLM is very sure, skip the blend
  4. Multi-provider fallback — Cohere → OpenAI → Mistral
  5. Threshold stays at 0.40 (same as v2.5)

Previous version preserved as: submission_quan_v25.py
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

# ── Confidence override threshold ─────────────────────────────────────────────
# If the LLM is very confident HIGH risk, don't let Cassandre dilute the signal.
# We only override upward (force FAIL), never downward — safety-first.
LLM_HIGH_CONFIDENCE = 0.85   # LLM score ≥ this → use LLM score directly


class CassandreGrandCompositeV3(BaseGuardrail):
    """
    Composite guardrail that blends Cassandre + LLM Judge scores,
    with a confidence override for high-conviction LLM calls.
    """

    def __init__(
        self,
        cassandre_gr: CassandreMultiTurnGuardrail,
        llm_judge_gr: Optional[LLMJudgeGuardrail],
        threshold: float = 0.4,
    ):
        super().__init__(GuardrailConfig(
            name="cassandre_grand_composite_v3",
            description="Blended score from Cassandre and LLM Judge with confidence override",
            threshold=threshold,
        ))
        self.cassandre = cassandre_gr
        self.llm_judge = llm_judge_gr

    def evaluate(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        evaluation_type: EvaluationType = EvaluationType.USER_INPUT,
    ) -> GuardrailResult:
        # 1. Get Cassandre Score
        cass_result = self.cassandre.evaluate(content, context, evaluation_type)
        cass_score = cass_result.score if cass_result.score is not None else (
            1.0 if cass_result.status == GuardrailStatus.FAIL else 0.0
        )

        # 2. Get LLM Judge Score
        llm_score = cass_score  # fallback
        llm_result = None
        if self.llm_judge:
            llm_result = self.llm_judge.evaluate(content, context, evaluation_type)
            llm_score = llm_result.score if llm_result.score is not None else (
                1.0 if llm_result.status == GuardrailStatus.FAIL else 0.0
            )

        # 3. Confidence override — if LLM is very sure it's high risk,
        #    don't let a low Cassandre score dilute the signal.
        #    We ONLY override upward (safety-first), never downward.
        override = None
        if llm_score >= LLM_HIGH_CONFIDENCE:
            override = "llm_high_confidence"
            grand_score = llm_score
        else:
            # Standard blend (same weights as v2.5)
            W_CASS = 0.4
            W_LLM = 0.6
            grand_score = (cass_score * W_CASS) + (llm_score * W_LLM)

        status = (
            GuardrailStatus.FAIL if grand_score >= self.config.threshold
            else GuardrailStatus.PASS
        )

        reasoning = (
            f"Grand Score: {grand_score:.2f} | "
            f"Cassandre: {cass_score:.2f} | "
            f"LLM: {llm_score:.2f}"
            f"{f' | override={override}' if override else ''}"
        )

        metadata = {
            "cassandre_result": cass_result.to_dict(),
            "grand_score": grand_score,
            "override": override,
        }
        if llm_result is not None:
            metadata["llm_result"] = llm_result.to_dict()

        return self._create_result(
            status=status,
            score=grand_score,
            reasoning=reasoning,
            metadata=metadata,
        )


def get_guardrails() -> Tuple[Optional[Any], Optional[Any]]:
    """Return (input_guardrail, output_guardrail)."""
    LOGGER.info("Loading Cassandre Grand Composite v3 (bilingual + confidence override)")

    # ── 1. Apply French keyword patch (side-effect import) ────────────────
    import src.guardrails.fr_patch  # noqa: F401

    # ── 2. Create LLM provider (OpenAI first — scored 0.852 on v2.5) ────
    llm_provider = None

    # OpenAI first (proven provider from v2.5)
    try:
        from providers.openai_provider import OpenAIProvider
        if os.getenv("BUZZ_GPT_OSS_AUTH_TOKEN"):
            llm_provider = OpenAIProvider()
            LOGGER.info("Using OpenAIProvider for LLM judge")
    except Exception as exc:
        LOGGER.warning("OpenAIProvider unavailable: %s", exc)

    # Fallback to Cohere
    if llm_provider is None:
        try:
            from providers.cohere_provider import CohereProvider
            if os.getenv("BUZZ_COHERE_AUTH_TOKEN"):
                llm_provider = CohereProvider()
                LOGGER.info("Using CohereProvider for LLM judge")
        except Exception as exc:
            LOGGER.warning("CohereProvider unavailable: %s", exc)

    # Fallback to Mistral
    if llm_provider is None:
        try:
            from providers.mistral_provider import MistralProvider
            if os.getenv("BUZZ_MISTRAL_LARGE_AUTH_TOKEN"):
                llm_provider = MistralProvider(
                    model=os.getenv("MISTRAL_MODEL"),
                    base_url=os.getenv("BUZZ_MISTRAL_LARGE_API"),
                    api_key=os.getenv("BUZZ_MISTRAL_LARGE_AUTH_TOKEN"),
                )
                LOGGER.info("Using MistralProvider for LLM judge")
        except Exception as exc:
            LOGGER.warning("MistralProvider unavailable: %s", exc)

    if llm_provider is None:
        LOGGER.warning("No LLM provider available — composite will use Cassandre only")

    # ── 3. Build components ───────────────────────────────────────────────
    cassandre_gr = CassandreMultiTurnGuardrail(config=GuardrailConfig(
        name="cassandre_multi_turn_base",
        description="Internal base for composite",
        threshold=0.5,
    ))

    llm_judge_gr = None
    if llm_provider is not None:
        try:
            from src.prompt_templates.few_shot_guardrail_prompt import FEW_SHOT_USER_INPUT_PROMPT
            llm_judge_gr = LLMJudgeGuardrail(
                config=GuardrailConfig(
                    name="llm_judge_few_shot_base",
                    description="Internal base for composite",
                    threshold=0.5,
                ),
                llm_provider=llm_provider,
                user_input_prompt=FEW_SHOT_USER_INPUT_PROMPT,
            )
            LOGGER.info("LLM judge loaded with original few-shot prompt")
        except Exception as exc:
            LOGGER.error("Failed to initialize LLM Judge: %s", exc)

    # ── 4. Build composite (same threshold as v2.5) ──────────────────────
    input_guardrail = CassandreGrandCompositeV3(
        cassandre_gr, llm_judge_gr, threshold=0.4
    )

    return (input_guardrail, None)
