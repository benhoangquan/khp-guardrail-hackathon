#!/usr/bin/env python3
"""
Build a feature CSV for train_ensemble_llm_cassandre_meta.py.

Columns written:
  - cassandre_score — CassandreMultiTurnGuardrail.evaluate().score
  - llm_score       — LLM judge score (or 0 with --skip_llm)
  - label

Example (repo root):

  uv run python project/scripts/build_ensemble_llm_cassandre_features.py \\
    --input_csv datasets/seed_validation_set.csv \\
    --output_csv datasets/ensemble_llm_cassandre_features.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))


def _parse_label(raw) -> int:
    if raw is None or (isinstance(raw, float) and str(raw) == "nan"):
        raise ValueError("missing label")
    value = str(raw).strip().lower()
    if value in {"0", "false", "no", "low", "low_risk"}:
        return 0
    if value in {"1", "true", "yes", "high", "high_risk"}:
        return 1
    try:
        n = int(float(value))
        if n in (0, 1):
            return n
    except ValueError:
        pass
    raise ValueError(f"unsupported label: {raw!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Cassandre+LLM feature CSV for meta training.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", default="")
    parser.add_argument("--text_column", default="Text")
    parser.add_argument("--label_column", default="label")
    parser.add_argument("--skip_llm", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError as e:
        print("Requires pandas.", e, file=sys.stderr)
        return 1

    in_path = Path(args.input_csv).expanduser()
    if not in_path.is_file():
        print(f"Input not found: {in_path}", file=sys.stderr)
        return 1

    out_path = Path(args.output_csv).expanduser() if args.output_csv else in_path.with_name(
        f"{in_path.stem}_llm_cassandre_features.csv"
    )

    df = pd.read_csv(in_path)
    tc, lc = args.text_column, args.label_column
    if tc not in df.columns or lc not in df.columns:
        print(f"Need columns {tc!r} and {lc!r}. Have: {list(df.columns)}", file=sys.stderr)
        return 1

    from providers.openai_provider import OpenAIProvider
    from src.guardrails.base import EvaluationType, GuardrailConfig, GuardrailStatus
    from src.guardrails.cassandre_guardrail import CassandreMultiTurnGuardrail
    from src.guardrails.llm_judge import LLMJudgeGuardrail
    from src.prompt_templates.few_shot_guardrail_prompt import FEW_SHOT_USER_INPUT_PROMPT

    cass = CassandreMultiTurnGuardrail(
        config=GuardrailConfig(
            name="feature_extract_cassandre",
            description="Inner Cassandre for feature CSV",
            threshold=0.5,
        )
    )

    llm_gr = None
    if not args.skip_llm:
        try:
            llm_gr = LLMJudgeGuardrail(
                config=GuardrailConfig(
                    name="feature_extract_llm",
                    description="Inner LLM for feature CSV",
                    threshold=0.5,
                ),
                llm_provider=OpenAIProvider(),
                user_input_prompt=FEW_SHOT_USER_INPUT_PROMPT,
            )
        except Exception as e:
            print(f"LLM init failed ({e}). Use --skip_llm or fix credentials.", file=sys.stderr)
            return 1

    sub = df.iloc[args.start :]
    if args.limit > 0:
        sub = sub.iloc[: args.limit]

    rows = []
    et = EvaluationType.USER_INPUT
    total = len(sub)
    for i, (_, row) in enumerate(sub.iterrows()):
        text = row[tc]
        if text is None or (isinstance(text, float) and str(text) == "nan"):
            text = ""
        else:
            text = str(text)
        y = _parse_label(row[lc])

        c_out = cass.evaluate(text, evaluation_type=et)
        c_score = float(c_out.score) if c_out.score is not None else (
            1.0 if c_out.status == GuardrailStatus.FAIL else 0.0
        )

        if llm_gr is not None:
            l_out = llm_gr.evaluate(text, evaluation_type=et)
            llm_s = float(l_out.score) if l_out.score is not None else (
                1.0 if l_out.status == GuardrailStatus.FAIL else 0.0
            )
        else:
            llm_s = 0.0

        rows.append({"cassandre_score": c_score, "llm_score": llm_s, "label": y})
        if (i + 1) % max(1, total // 10) == 0 or (i + 1) == total:
            print(f"Processed {i + 1}/{total}", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
