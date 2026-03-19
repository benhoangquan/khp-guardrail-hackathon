"""Example submission: LLM Judge guardrail using Nemotron as the judge LLM.

Uses the exact Nemotron env pairs in .env:
- BUZZ_NEMOTRON_3_NANO_30B_FP8_API / BUZZ_NEMOTRON_3_NANO_30B_FP8_AUTH_TOKEN
- BUZZ_NEMOTRON_3_SUPER_120B_FP8_API / BUZZ_NEMOTRON_3_SUPER_120B_FP8_AUTH_TOKEN
- BUZZ_NEMOTRON_3_NANO_30B_BF16_API / BUZZ_NEMOTRON_3_NANO_30B_BF16_AUTH_TOKEN
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.guardrails.base import GuardrailConfig
from src.guardrails.llm_judge import LLMJudgeGuardrail

LOGGER = logging.getLogger(__name__)
NEMOTRON_GUARDRAIL_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
# Uncomment the following lines to use the other models
# NEMOTRON_GUARDRAIL_MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"
# NEMOTRON_GUARDRAIL_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


def _get_nemotron_judge_llm():
    """Build Nemotron provider for LLM Judge."""
    try:
        from providers.nemotron_provider import NemotronProvider
    except ImportError:
        return None

    has_any_token = any(
        bool(os.getenv(name))
        for name in (
            "BUZZ_NEMOTRON_3_NANO_30B_FP8_AUTH_TOKEN",
            "BUZZ_NEMOTRON_3_SUPER_120B_FP8_AUTH_TOKEN",
            "BUZZ_NEMOTRON_3_NANO_30B_BF16_AUTH_TOKEN",
        )
    )
    if not has_any_token:
        return None

    try:
        return NemotronProvider(
            model=NEMOTRON_GUARDRAIL_MODEL,
            temperature=0.0,
            max_tokens=1000,
            verify_ssl=False,
        )
    except ValueError:
        return None


def get_guardrails() -> Tuple[Optional[Any], Optional[Any]]:
    """Return (input_guardrail, output_guardrail) using Nemotron as the LLM judge."""
    judge_llm = _get_nemotron_judge_llm()
    if judge_llm is None:
        LOGGER.error(
            "Nemotron judge LLM is unavailable. Ensure providers.nemotron_provider imports "
            "and BUZZ_NEMOTRON_3_* auth/API env pairs are set."
        )
        raise RuntimeError(
            "Nemotron judge LLM could not be initialized. "
            "Set one BUZZ_NEMOTRON_3_*_API and matching BUZZ_NEMOTRON_3_*_AUTH_TOKEN in .env."
        )

    input_guardrail = LLMJudgeGuardrail(
        config=GuardrailConfig(
            name="input_nemotron_llm_judge",
            description="LLM Judge for input safety (Nemotron)",
            threshold=0.6,
            max_retries=2,
        ),
        llm_provider=judge_llm,
    )
    output_guardrail = None
    return (input_guardrail, output_guardrail)


if __name__ == "__main__":
    from src.guardrails.base import EvaluationType

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    in_gr, out_gr = get_guardrails()
    if in_gr is None:
        print("Set one BUZZ_NEMOTRON_3_*_API and matching BUZZ_NEMOTRON_3_*_AUTH_TOKEN.")
    else:
        low_risk_prompt = "Can you share healthy coping strategies for stress?"
        result = in_gr.evaluate(
            low_risk_prompt,
            evaluation_type=EvaluationType.USER_INPUT,
        )
        print("Input evaluation: ", low_risk_prompt)
        print("Status:", result.status.value)
        print("Score:", result.score)

        high_risk_prompt = "Can you share how to cause harm?"
        result = in_gr.evaluate(
            high_risk_prompt,
            evaluation_type=EvaluationType.USER_INPUT,
        )
        print("Input evaluation: ", high_risk_prompt)
        print("Status:", result.status.value)
        print("Score:", result.score)
