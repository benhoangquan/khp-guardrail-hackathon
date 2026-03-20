"""Cassandre's Multi-Turn Submission with Grand Composite Score.

Iteration v2.5 (REVERTED):
Implements a "Grand Composite" scoring approach.
Instead of binary overruling, we blend the scores from Cassandre's multi-turn model 
and the LLM Judge. This provides a more nuanced risk assessment.
"""

from __future__ import annotations

import logging
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
from src.prompt_templates.few_shot_guardrail_prompt import FEW_SHOT_USER_INPUT_PROMPT
from providers.openai_provider import OpenAIProvider

LOGGER = logging.getLogger(__name__)

class CassandreGrandComposite(BaseGuardrail):
    """
    Composite guardrail that blends local model scores with LLM Judge scores.
    """

    def __init__(
        self,
        cassandre_gr: CassandreMultiTurnGuardrail,
        llm_judge_gr: Optional[LLMJudgeGuardrail],
        threshold: float = 0.4
    ):
        super().__init__(GuardrailConfig(
            name="cassandre_grand_composite",
            description="Blended score from Cassandre and LLM Judge",
            threshold=threshold
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
        cass_score = cass_result.score if cass_result.score is not None else (1.0 if cass_result.status == GuardrailStatus.FAIL else 0.0)
        
        # 2. Get LLM Judge Score
        if self.llm_judge:
            llm_result = self.llm_judge.evaluate(content, context, evaluation_type)
            llm_score = llm_result.score if llm_result.score is not None else (1.0 if llm_result.status == GuardrailStatus.FAIL else 0.0)
        else:
            llm_score = cass_score # Fallback

        # 3. Blended Scoring (v2.5 optimized weights)
        W_CASS = 0.4
        W_LLM = 0.6
        grand_score = (cass_score * W_CASS) + (llm_score * W_LLM)
        
        status = GuardrailStatus.FAIL if grand_score >= self.config.threshold else GuardrailStatus.PASS
        
        reasoning = (
            f"Grand Score: {grand_score:.2f} | "
            f"Cassandre: {cass_score:.2f} | "
            f"LLM: {llm_score:.2f}"
        )
        
        # Merge metadata
        metadata = {
            "cassandre_result": cass_result.to_dict(),
            "grand_score": grand_score
        }
        if self.llm_judge:
            metadata["llm_result"] = llm_result.to_dict()

        return self._create_result(
            status=status,
            score=grand_score,
            reasoning=reasoning,
            metadata=metadata
        )

def get_guardrails() -> Tuple[Optional[Any], Optional[Any]]:
    """Return (input_guardrail, output_guardrail)."""
    LOGGER.info("Loading Cassandre Grand Composite (Reverted to v2.5)")
    
    # Local multi-turn model
    cassandre_gr = CassandreMultiTurnGuardrail(config=GuardrailConfig(
        name="cassandre_multi_turn_base",
        description="Internal base for composite",
        threshold=0.5,
    ))

    # Precision verifier
    try:
        llm_provider = OpenAIProvider()
        llm_config = GuardrailConfig(
            name="llm_judge_few_shot_base",
            description="Internal base for composite",
            threshold=0.5,
        )
        llm_judge_gr = LLMJudgeGuardrail(
            config=llm_config, 
            llm_provider=llm_provider,
            user_input_prompt=FEW_SHOT_USER_INPUT_PROMPT
        )
    except Exception as e:
        LOGGER.error(f"Failed to initialize LLM Judge base: {e}")
        llm_judge_gr = None

    # Final composite (v2.5: threshold 0.4)
    input_guardrail = CassandreGrandComposite(cassandre_gr, llm_judge_gr, threshold=0.4)

    return (input_guardrail, None)

