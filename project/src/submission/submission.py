"""LLM-Primary v5 — Drop Cassandre blend, use LLM boolean directly.

Analysis showed Cassandre is non-discriminative (mean=0.588, max=0.650 across
ALL samples) and the blend was adding noise. The simulated eval (v7) confirmed
LLM-only outperforms every blend variant by +6% F1.

Architecture:
  1. French keyword patch still imported (side-effect for Cassandre fallback)
  2. LLM Judge as PRIMARY decision-maker — use its status (high_risk boolean)
  3. Cassandre as FALLBACK only when no LLM provider is available
  4. Provider fallback: OpenAI → Cohere → Mistral
  5. Baseline prompt (FEW_SHOT_USER_INPUT_PROMPT) — simplest prompt, best F1
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


class LLMPrimaryGuardrail(BaseGuardrail):
    """
    LLM-primary guardrail: uses LLM Judge's own binary decision (high_risk
    boolean) as the final PASS/FAIL. Cassandre runs for metadata only and
    serves as fallback when no LLM provider is available.
    """

    def __init__(
        self,
        cassandre_gr: CassandreMultiTurnGuardrail,
        llm_judge_gr: Optional[LLMJudgeGuardrail],
    ):
        super().__init__(GuardrailConfig(
            name="llm_primary_v5",
            description="LLM-primary with Cassandre fallback",
            threshold=0.50,
        ))
        self.cassandre = cassandre_gr
        self.llm_judge = llm_judge_gr

    def evaluate(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        evaluation_type: EvaluationType = EvaluationType.USER_INPUT,
    ) -> GuardrailResult:
        # 1. Always run Cassandre (for metadata + fallback)
        cass_result = self.cassandre.evaluate(content, context, evaluation_type)
        cass_score = cass_result.score if cass_result.score is not None else (
            1.0 if cass_result.status == GuardrailStatus.FAIL else 0.0
        )

        # 2. LLM Judge — primary decision-maker
        if self.llm_judge:
            llm_result = self.llm_judge.evaluate(content, context, evaluation_type)
            llm_score = llm_result.score if llm_result.score is not None else (
                1.0 if llm_result.status == GuardrailStatus.FAIL else 0.0
            )

            # Use LLM's own binary decision (high_risk boolean + internal threshold)
            status = llm_result.status
            grand_score = llm_score

            reasoning = (
                f"LLM-Primary | LLM: {llm_score:.2f} (status={status.value}) | "
                f"Cassandre: {cass_score:.2f} (metadata only)"
            )
        else:
            # Fallback: no LLM available — use Cassandre alone
            llm_result = None
            grand_score = cass_score
            status = cass_result.status

            reasoning = (
                f"Cassandre-Fallback | Cassandre: {cass_score:.2f} | "
                f"No LLM provider available"
            )

        metadata = {
            "cassandre_result": cass_result.to_dict(),
            "grand_score": grand_score,
            "architecture": "llm_primary_v5",
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
    LOGGER.info("Loading LLM-Primary v5 (baseline prompt, no Cassandre blend)")

    # ── 1. Apply French keyword patch (side-effect import) ────────────────
    import src.guardrails.fr_patch  # noqa: F401

    # ── 2. Create LLM provider (OpenAI first) ─────────────────────────────
    llm_provider = None

    try:
        from providers.openai_provider import OpenAIProvider
        if os.getenv("BUZZ_GPT_OSS_AUTH_TOKEN"):
            llm_provider = OpenAIProvider()
            LOGGER.info("Using OpenAIProvider for LLM judge")
    except Exception as exc:
        LOGGER.warning("OpenAIProvider unavailable: %s", exc)

    if llm_provider is None:
        try:
            from providers.cohere_provider import CohereProvider
            if os.getenv("BUZZ_COHERE_AUTH_TOKEN"):
                llm_provider = CohereProvider()
                LOGGER.info("Using CohereProvider for LLM judge")
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
                LOGGER.info("Using MistralProvider for LLM judge")
        except Exception as exc:
            LOGGER.warning("MistralProvider unavailable: %s", exc)

    if llm_provider is None:
        LOGGER.warning("No LLM provider — will use Cassandre fallback only")

    # ── 3. Build components ───────────────────────────────────────────────
    cassandre_gr = CassandreMultiTurnGuardrail(config=GuardrailConfig(
        name="cassandre_multi_turn_base",
        description="Internal base for fallback + metadata",
        threshold=0.5,
    ))

    llm_judge_gr = None
    if llm_provider is not None:
        try:
            from src.prompt_templates.few_shot_guardrail_prompt import (
                FEW_SHOT_USER_INPUT_PROMPT,
            )
            llm_judge_gr = LLMJudgeGuardrail(
                config=GuardrailConfig(
                    name="llm_judge_baseline",
                    description="LLM judge with baseline few-shot prompt",
                    threshold=0.5,
                ),
                llm_provider=llm_provider,
                user_input_prompt=FEW_SHOT_USER_INPUT_PROMPT,
            )
            LOGGER.info("LLM judge loaded with baseline few-shot prompt")
        except Exception as exc:
            LOGGER.error("Failed to initialize LLM Judge: %s", exc)

    # ── 4. Build LLM-primary guardrail ────────────────────────────────────
    input_guardrail = LLMPrimaryGuardrail(cassandre_gr, llm_judge_gr)

    return (input_guardrail, None)
