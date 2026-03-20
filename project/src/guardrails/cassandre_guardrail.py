"""
Cassandre's Multi-Turn Guardrail.

Combines the ConvoAnalyzer (linguistic features) and DDMModel (cognitive accumulation)
into a single guardrail that handles multi-turn conversation context.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.guardrails.base import (
    BaseGuardrail,
    EvaluationType,
    GuardrailConfig,
    GuardrailResult,
    GuardrailStatus,
)
from src.guardrails.convo_adapter import ConvoAnalyzer, parse_turns
from src.guardrails.ddm_model import ddm_score

LOGGER = logging.getLogger(__name__)

class CassandreMultiTurnGuardrail(BaseGuardrail):
    """
    Guardrail that uses Cassandre's multi-turn analysis.
    
    It accumulates risk over a conversation using a Drift Diffusion Model (DDM)
    and linguistic trajectory analysis.
    """

    def __init__(self, config: Optional[GuardrailConfig] = None):
        super().__init__(config or GuardrailConfig(
            name="cassandre_multi_turn",
            description="Multi-turn risk accumulation via DDM and ConvoAnalyzer",
            threshold=0.5,
        ))
        self.analyzer = ConvoAnalyzer()

    def evaluate(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        evaluation_type: EvaluationType = EvaluationType.USER_INPUT,
    ) -> GuardrailResult:
        """
        Evaluate a turn or a whole conversation string.
        
        Handles both single turn strings and multi-turn strings with role prefixes.
        Expects context to contain 'history' (list of previous turns) for 
        incremental multi-turn analysis.
        """
        # 1. Parse content to see if it's already a multi-turn string
        content_turns = parse_turns(content)
        
        # 2. Reconstruct conversation from history + content
        history = (context or {}).get("history", [])
        
        # Case A: content is a single turn (no role prefixes detected by parse_turns)
        # Case B: content is a multi-turn string (already has user: / assistant: etc.)
        
        if len(content_turns) == 1 and "user:" not in content.lower() and "assistant:" not in content.lower():
            # Standard incremental mode: history + current single turn
            user_turns = [turn["content"] for turn in history if turn.get("role") == "user"]
            user_turns.append(content)
            
            full_conversation = ""
            for turn in history:
                role = turn.get("role", "user")
                full_conversation += f"{role}: {turn['content']}\n"
            full_conversation += f"user: {content}\n"
        else:
            # Multi-turn string mode (e.g. from a dataset where 'content' is the whole convo)
            # We ignore history here as the whole convo is in 'content'
            user_turns = [c for role, c in content_turns if role == "user"]
            full_conversation = content

        # 3. Run ConvoAnalyzer (linguistic features)
        convo_result = self.analyzer.analyze(full_conversation)
        convo_score = convo_result["score"]

        # 4. Run DDM Model (cognitive accumulation)
        ddm_result = ddm_score(user_turns)
        ddm_score_val = ddm_result["score"]

        # 4. Composite score (weighted average)
        # We give more weight to DDM as it captures the accumulation logic
        composite_score = (ddm_score_val * 0.6) + (convo_score * 0.4)
        
        status = GuardrailStatus.FAIL if composite_score >= self.config.threshold else GuardrailStatus.PASS
        
        reasoning = (
            f"Composite Risk: {composite_score:.2f} | "
            f"DDM (Accumulation): {ddm_score_val:.2f} | "
            f"Convo (Linguistic): {convo_score:.2f} | "
            f"Profile: {ddm_result['profile']}"
        )

        return self._create_result(
            status=status,
            score=composite_score,
            reasoning=reasoning,
            metadata={
                "ddm_result": ddm_result,
                "convo_features": convo_result["features"],
                "n_turns": len(user_turns)
            }
        )
