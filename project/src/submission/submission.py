"""Hackathon submission entry: Cassandre + LLM ensemble (meta-logit).

Delegates to `submission_ensemble_llm_cassandre` (Cassandre multi-turn score +
LLM judge score → logistic meta-learner). No mmBERT.

Optional: `project/models/ensemble_llm_cassandre_meta.json` overrides embedded coefs.
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


class MultiModelEnsemble(BaseGuardrail):
    """
    Multi-model ensemble with boolean voting.
    When models agree, trust them. When they disagree, Cassandre breaks the tie.
    """

    def __init__(
        self,
        cassandre_gr: CassandreMultiTurnGuardrail,
        llm_a: Optional[LLMJudgeGuardrail],
        llm_b: Optional[LLMJudgeGuardrail],
    ):
        super().__init__(GuardrailConfig(
            name="multi_model_ensemble_v7",
            description="Multi-model boolean voting + Cassandre tiebreaker",
            threshold=0.40,
        ))
        self.cassandre = cassandre_gr
        self.llm_a = llm_a
        self.llm_b = llm_b

    def evaluate(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        evaluation_type: EvaluationType = EvaluationType.USER_INPUT,
    ) -> GuardrailResult:
        # 1. Cassandre (always — for tiebreaker + fallback)
        cass_result = self.cassandre.evaluate(content, context, evaluation_type)
        cass_score = cass_result.score if cass_result.score is not None else (
            1.0 if cass_result.status == GuardrailStatus.FAIL else 0.0
        )

        # 2. Model A
        a_result = None
        a_score = cass_score
        a_fail = cass_result.status == GuardrailStatus.FAIL
        if self.llm_a:
            a_result = self.llm_a.evaluate(content, context, evaluation_type)
            a_score = a_result.score if a_result.score is not None else (
                1.0 if a_result.status == GuardrailStatus.FAIL else 0.0
            )
            a_fail = a_result.status == GuardrailStatus.FAIL

        # 3. Model B
        b_result = None
        b_score = cass_score
        b_fail = cass_result.status == GuardrailStatus.FAIL
        if self.llm_b:
            b_result = self.llm_b.evaluate(content, context, evaluation_type)
            b_score = b_result.score if b_result.score is not None else (
                1.0 if b_result.status == GuardrailStatus.FAIL else 0.0
            )
            b_fail = b_result.status == GuardrailStatus.FAIL

        # 4. Single-model fallback (only one LLM available)
        if self.llm_a and not self.llm_b:
            # Single model — blend with Cassandre like baseline
            grand_score = 0.4 * cass_score + 0.6 * a_score
            status = (
                GuardrailStatus.FAIL if grand_score >= 0.40
                else GuardrailStatus.PASS
            )
            vote = "single_model_a"
        elif self.llm_b and not self.llm_a:
            grand_score = 0.4 * cass_score + 0.6 * b_score
            status = (
                GuardrailStatus.FAIL if grand_score >= 0.40
                else GuardrailStatus.PASS
            )
            vote = "single_model_b"
        elif not self.llm_a and not self.llm_b:
            # No LLM — Cassandre only
            grand_score = cass_score
            status = cass_result.status
            vote = "cassandre_only"
        else:
            # 5. Multi-model voting
            if a_fail and b_fail:
                # Both agree HIGH RISK — high confidence
                grand_score = max(a_score, b_score)
                status = GuardrailStatus.FAIL
                vote = "both_fail"
            elif not a_fail and not b_fail:
                # Both agree LOW RISK — high confidence
                grand_score = min(a_score, b_score)
                status = GuardrailStatus.PASS
                vote = "both_pass"
            else:
                # Disagree — Cassandre tiebreaker + max LLM score
                max_llm = max(a_score, b_score)
                grand_score = 0.4 * cass_score + 0.6 * max_llm
                status = (
                    GuardrailStatus.FAIL if grand_score >= 0.40
                    else GuardrailStatus.PASS
                )
                vote = "disagree_cass_tiebreak"

        reasoning = (
            f"Ensemble | A: {a_score:.2f}({'F' if a_fail else 'P'}) | "
            f"B: {b_score:.2f}({'F' if b_fail else 'P'}) | "
            f"Cass: {cass_score:.2f} | Vote: {vote} → {grand_score:.2f}"
        )

        metadata = {
            "cassandre_result": cass_result.to_dict(),
            "grand_score": grand_score,
            "vote": vote,
            "architecture": "multi_model_ensemble_v7",
        }
        if a_result is not None:
            metadata["model_a_result"] = a_result.to_dict()
        if b_result is not None:
            metadata["model_b_result"] = b_result.to_dict()

        return self._create_result(
            status=status,
            score=grand_score,
            reasoning=reasoning,
            metadata=metadata,
        )


def _try_provider(name, provider_cls, env_key, **kwargs):
    """Try to create a provider, return None if unavailable."""
    try:
        if os.getenv(env_key):
            provider = provider_cls(**kwargs)
            LOGGER.info("Created %s provider", name)
            return provider
    except Exception as exc:
        LOGGER.warning("%s unavailable: %s", name, exc)
    return None


def get_guardrails() -> Tuple[Optional[Any], Optional[Any]]:
    """Return (input_guardrail, output_guardrail)."""
    LOGGER.info("Loading Multi-Model Ensemble v7")

    # ── 1. French keyword patch ───────────────────────────────────────────
    import src.guardrails.fr_patch  # noqa: F401

    # ── 2. Create ALL available providers ─────────────────────────────────
    providers = {}

    try:
        from providers.openai_provider import OpenAIProvider
        p = _try_provider("OpenAI", OpenAIProvider, "BUZZ_GPT_OSS_AUTH_TOKEN")
        if p:
            providers["openai"] = p
    except ImportError:
        pass

    try:
        from providers.cohere_provider import CohereProvider
        p = _try_provider("Cohere", CohereProvider, "BUZZ_COHERE_AUTH_TOKEN")
        if p:
            providers["cohere"] = p
    except ImportError:
        pass

    try:
        from providers.mistral_provider import MistralProvider
        p = _try_provider(
            "Mistral", MistralProvider, "BUZZ_MISTRAL_LARGE_AUTH_TOKEN",
            model=os.getenv("MISTRAL_MODEL"),
            base_url=os.getenv("BUZZ_MISTRAL_LARGE_API"),
            api_key=os.getenv("BUZZ_MISTRAL_LARGE_AUTH_TOKEN"),
        )
        if p:
            providers["mistral"] = p
    except ImportError:
        pass

    LOGGER.info("Available providers: %s", list(providers.keys()))

    # ── 3. Build components ───────────────────────────────────────────────
    cassandre_gr = CassandreMultiTurnGuardrail(config=GuardrailConfig(
        name="cassandre_base",
        description="Tiebreaker + fallback",
        threshold=0.5,
    ))

    # ── 4. Assign models to ensemble slots ────────────────────────────────
    # Model A: primary (OpenAI preferred) + baseline prompt
    # Model B: secondary (different model) + bilingual prompt
    from src.prompt_templates.few_shot_guardrail_prompt import (
        FEW_SHOT_USER_INPUT_PROMPT,
    )
    from src.guardrails.fr_patch import BILINGUAL_PROMPT

    llm_a = None
    llm_b = None

    # Pick Model A (primary): OpenAI > Cohere > Mistral
    provider_a = providers.get("openai") or providers.get("cohere") or providers.get("mistral")
    provider_a_name = (
        "openai" if "openai" in providers else
        "cohere" if "cohere" in providers else
        "mistral" if "mistral" in providers else None
    )

    if provider_a:
        try:
            llm_a = LLMJudgeGuardrail(
                config=GuardrailConfig(
                    name="model_a_baseline",
                    description=f"Primary ({provider_a_name}) + baseline prompt",
                    threshold=0.5,
                ),
                llm_provider=provider_a,
                user_input_prompt=FEW_SHOT_USER_INPUT_PROMPT,
            )
            LOGGER.info("Model A: %s + FEW_SHOT prompt", provider_a_name)
        except Exception as exc:
            LOGGER.error("Failed to create Model A: %s", exc)

    # Pick Model B (secondary): different model than A
    provider_b = None
    provider_b_name = None
    for name in ["mistral", "cohere", "openai"]:
        if name in providers and name != provider_a_name:
            provider_b = providers[name]
            provider_b_name = name
            break

    if provider_b:
        try:
            llm_b = LLMJudgeGuardrail(
                config=GuardrailConfig(
                    name="model_b_bilingual",
                    description=f"Secondary ({provider_b_name}) + bilingual prompt",
                    threshold=0.5,
                ),
                llm_provider=provider_b,
                user_input_prompt=BILINGUAL_PROMPT,
            )
            LOGGER.info("Model B: %s + BILINGUAL prompt", provider_b_name)
        except Exception as exc:
            LOGGER.error("Failed to create Model B: %s", exc)

    # If only one provider exists, use it for both with different prompts
    if llm_a and not llm_b and provider_a:
        try:
            llm_b = LLMJudgeGuardrail(
                config=GuardrailConfig(
                    name="model_b_bilingual_same",
                    description=f"Same provider ({provider_a_name}) + bilingual prompt",
                    threshold=0.5,
                ),
                llm_provider=provider_a,
                user_input_prompt=BILINGUAL_PROMPT,
            )
            LOGGER.info("Model B: same provider (%s) + BILINGUAL prompt (single-provider mode)", provider_a_name)
        except Exception as exc:
            LOGGER.error("Failed to create Model B fallback: %s", exc)

    n_models = sum(1 for x in [llm_a, llm_b] if x is not None)
    LOGGER.info("Ensemble ready: %d model(s), Cassandre tiebreaker", n_models)

    # ── 5. Build ensemble ─────────────────────────────────────────────────
    input_guardrail = MultiModelEnsemble(cassandre_gr, llm_a, llm_b)

    return (input_guardrail, None)
