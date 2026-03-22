"""Cassandre Grand Composite v3 — FR keywords + few-shot LLM + confidence override.

Based on Quan's v2.5 blended scoring architecture (0.880-style config).
  1. French keyword patch (import side-effect) — Cassandre scores FR convos
  2. Few-shot LLM prompt (`few_shot_guardrail_prompt`) — includes EN + FR examples
  3. Confidence override when LLM score ≥ LLM_HIGH_CONFIDENCE
  4. Provider fallback: OpenAI → Cohere → Mistral
  5. Composite threshold 0.40; blend W_CASS=0.4 / W_LLM=0.6

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


def _truncate_for_llm(content: str, first_n: int = 4, last_n: int = 8) -> str:
    """
    Keep only the first `first_n` + last `last_n` user turns to prevent
    'lost in the middle' degradation (Liu et al. 2023).  Assistant turns
    adjacent to kept user turns are preserved to maintain conversational
    context.  If the conversation is short enough, returns it unchanged.
    """
    from src.guardrails.convo_adapter import parse_turns
    turns = parse_turns(content)
    user_indices = [i for i, (role, _) in enumerate(turns) if role == "user"]
    if len(user_indices) <= first_n + last_n:
        return content
    # boundary turn indices
    first_boundary = user_indices[first_n - 1]   # last turn of "opening" block
    last_boundary = user_indices[-last_n]         # first turn of "closing" block
    kept = [
        (role, text) for i, (role, text) in enumerate(turns)
        if i <= first_boundary or i >= last_boundary
    ]
    return "\n".join(f"{role}: {text}" for role, text in kept)


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
        threshold: float = 0.40,
    ):
        super().__init__(GuardrailConfig(
            name="cassandre_grand_composite_v3",
            description="Blended score from Cassandre and LLM Judge with confidence override",
            threshold=threshold,
        ))
        self.cassandre = cassandre_gr
        self.llm_judge = llm_judge_gr

    def _self_consistent_llm_score(
        self,
        content: str,
        context: Optional[Dict[str, Any]],
        evaluation_type: EvaluationType,
        n: int = 3,
    ) -> tuple:
        """
        Sample LLM judge n times and return (mean_score, first_result).

        Averaging independent samples reduces score variance with CoT prompts,
        especially for borderline cases near the decision threshold.
        Short-circuits on the first high-confidence sample (≥ LLM_HIGH_CONFIDENCE)
        to limit latency when the answer is unambiguous.

        Requires temperature > 0 on the LLM provider for meaningful diversity
        (most providers default to temperature=1.0).

        Source: Wang et al. (2022) "Self-Consistency Improves Chain of Thought
        Reasoning in Language Models", ICLR 2023 — majority-vote / score
        averaging over multiple CoT samples consistently outperforms single-pass
        on reasoning tasks; applied here to safety classification.
        """
        scores: list[float] = []
        first_result = None
        for _ in range(n):
            try:
                result = self.llm_judge.evaluate(content, context, evaluation_type)
                s = result.score if result.score is not None else (
                    1.0 if result.status == GuardrailStatus.FAIL else 0.0
                )
                if first_result is None:
                    first_result = result
                scores.append(s)
                if s >= LLM_HIGH_CONFIDENCE:
                    break  # high confidence; no need for further samples
            except Exception:
                break
        if not scores:
            return 0.0, None
        return sum(scores) / len(scores), first_result

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

        # 2. Get LLM Judge Score (self-consistent sampling)
        llm_score = cass_score  # fallback when no LLM
        llm_result = None
        if self.llm_judge:
            llm_content = _truncate_for_llm(content)
            # Dynamic few-shot: tune the system prompt to this conversation
            try:
                from src.guardrails.fr_patch import build_prompt as _build_prompt
                self.llm_judge.user_input_prompt = _build_prompt(llm_content)
            except Exception:
                pass  # keep existing static prompt on any failure
            # Self-consistency: average n=3 samples, short-circuit at high confidence
            # Source: Wang et al. (2022) ICLR 2023
            llm_score, llm_result = self._self_consistent_llm_score(
                llm_content, context, evaluation_type
            )

        # 3. Confidence override: if LLM is very confident of high risk, use its score directly.
        #    Only upward override (safety-first), never downward.
        override = None
        if llm_score >= LLM_HIGH_CONFIDENCE:
            override = "llm_high_confidence"
            grand_score = llm_score
        else:
            W_CASS = 0.4
            W_LLM = 0.6
            grand_score = (cass_score * W_CASS) + (llm_score * W_LLM)

        # 4. Ensemble disagreement: if Cassandre and LLM strongly disagree (gap > 0.30),
        #    lower the threshold to 0.30. When uncertain, flag rather than miss.
        effective_threshold = self.config.threshold
        if abs(cass_score - llm_score) > 0.30:
            effective_threshold = min(self.config.threshold, 0.30)
            override = f"{override}+disagreement" if override else "disagreement"

        status = (
            GuardrailStatus.FAIL if grand_score >= effective_threshold
            else GuardrailStatus.PASS
        )

        reasoning = (
            f"Grand Score: {grand_score:.2f} | "
            f"Cassandre: {cass_score:.2f} | "
            f"LLM: {llm_score:.2f} | "
            f"threshold={effective_threshold:.2f}"
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

    # ── 1. Apply French keyword patch + import bilingual prompt ──────────
    from src.guardrails.fr_patch import BILINGUAL_PROMPT  # noqa: F401 (also patches keyword lists)

    # ── 2. Create LLM provider (OpenAI first — scored 0.852 on v2.5) ────
    llm_provider = None

    # OpenAI first (proven provider from v2.5)
    try:
        from providers.openai_provider import OpenAIProvider
        if os.getenv("BUZZ_GPT_OSS_AUTH_TOKEN"):
            llm_provider = OpenAIProvider(max_tokens=400)
            LOGGER.info("Using OpenAIProvider for LLM judge")
    except Exception as exc:
        LOGGER.warning("OpenAIProvider unavailable: %s", exc)

    # Fallback to Cohere
    if llm_provider is None:
        try:
            from providers.cohere_provider import CohereProvider
            if os.getenv("BUZZ_COHERE_AUTH_TOKEN"):
                llm_provider = CohereProvider(max_tokens=400)
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
                    max_tokens=400,
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
            llm_judge_gr = LLMJudgeGuardrail(
                config=GuardrailConfig(
                    name="llm_judge_bilingual_base",
                    description="Internal base for composite",
                    threshold=0.5,
                    max_retries=2,
                ),
                llm_provider=llm_provider,
                user_input_prompt=BILINGUAL_PROMPT,
            )
            LOGGER.info("LLM judge loaded with BILINGUAL_PROMPT (C-SSRS grounded, EN+FR)")
        except Exception as exc:
            LOGGER.error("Failed to initialize LLM Judge: %s", exc)

    # ── 4. Build composite (same threshold as v2.5) ──────────────────────
    input_guardrail = CassandreGrandCompositeV3(
        cassandre_gr, llm_judge_gr, threshold=0.40
    )

    return (input_guardrail, None)
