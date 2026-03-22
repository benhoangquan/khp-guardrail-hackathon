#!/usr/bin/env python3
"""
Build a feature CSV for train_ensemble_cascade_meta.py from a labeled validation CSV.

For each row, computes (matching submission_ensemble_cascade):
  - word_filter  — risk-keyword binary feature (same as cascade)
  - mmbert_prob  — P(high_risk) from finetuned mmBERT
  - llm_score    — LLM judge score in [0,1], or 0 when --skip_llm

Required columns (names configurable):
  - text column (default: Text) — full prompt/conversation string passed to scorers
  - label column (default: label) — 0 = low_risk, 1 = high_risk

Example (from repo root):

  uv run python project/scripts/build_ensemble_cascade_features.py \\
    --input_csv datasets/seed_validation_set.csv \\
    --output_csv datasets/ensemble_cascade_features.csv

Then train the meta-learner (paths relative to project/ or absolute):

  cd project && uv run python scripts/train_ensemble_cascade_meta.py \\
    --features_csv ../datasets/ensemble_cascade_features.csv \\
    --output_json models/ensemble_cascade_meta.json \\
    --threshold 0.3
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
    parser = argparse.ArgumentParser(description="Build ensemble cascade feature CSV for meta training.")
    parser.add_argument("--input_csv", required=True, help="Labeled CSV (e.g. seed_validation_set.csv)")
    parser.add_argument(
        "--output_csv",
        default="",
        help="Output path (default: <input_basename>_ensemble_features.csv next to input)",
    )
    parser.add_argument("--text_column", default="Text", help="Column with full text for scoring")
    parser.add_argument("--label_column", default="label", help="Binary label column (0/1)")
    parser.add_argument(
        "--mmbert_path",
        default=str(_PROJECT / "models" / "mmbert_guardrail_demo"),
        help="Finetuned mmBERT directory (save_pretrained output)",
    )
    parser.add_argument(
        "--skip_llm",
        action="store_true",
        help="Set llm_score=0 for every row (fast; train meta on filter+mmBERT only)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Process at most N rows (0 = all)")
    parser.add_argument("--start", type=int, default=0, help="Skip first N rows")
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
        f"{in_path.stem}_ensemble_features.csv"
    )

    df = pd.read_csv(in_path)
    tc = args.text_column
    lc = args.label_column
    if tc not in df.columns:
        print(f"Missing text column {tc!r}. Columns: {list(df.columns)}", file=sys.stderr)
        return 1
    if lc not in df.columns:
        print(f"Missing label column {lc!r}. Columns: {list(df.columns)}", file=sys.stderr)
        return 1

    mmbert_dir = Path(args.mmbert_path).expanduser()
    if not mmbert_dir.is_dir():
        print(
            f"mmBERT directory not found: {mmbert_dir}\n"
            "Train first: python -m src.guardrails.train_classifier_guardrail ...",
            file=sys.stderr,
        )
        return 1

    from providers.openai_provider import OpenAIProvider
    from src.guardrails.base import (
        EvaluationType,
        GuardrailConfig,
        GuardrailStatus,
    )
    from src.guardrails.classifier import load_classifier_guardrail
    from src.guardrails.llm_judge import LLMJudgeGuardrail
    from src.prompt_templates.few_shot_guardrail_prompt import FEW_SHOT_USER_INPUT_PROMPT
    from src.submission._runtime_config import resolve_device_from_hackathon
    from src.submission.submission_ensemble_cascade import word_filter_feature

    device = resolve_device_from_hackathon(_PROJECT)
    mmbert_gr = load_classifier_guardrail(
        model_path=str(mmbert_dir),
        name="feature_extract_mmbert",
        description="Inner mmBERT for feature CSV",
        threshold=1.01,
        device=device,
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

    rows_out = []
    eval_type = EvaluationType.USER_INPUT
    total = len(sub)
    for i, (_, row) in enumerate(sub.iterrows()):
        text = row[tc]
        if text is None or (isinstance(text, float) and str(text) == "nan"):
            text = ""
        else:
            text = str(text)
        try:
            y = _parse_label(row[lc])
        except ValueError as e:
            print(f"Row {i}: bad label: {e}", file=sys.stderr)
            return 1

        wf = word_filter_feature(text, mode="binary")
        m_out = mmbert_gr.evaluate(text, evaluation_type=eval_type)
        m_prob = float(m_out.score) if m_out.score is not None else (
            1.0 if m_out.status == GuardrailStatus.FAIL else 0.0
        )

        if llm_gr is not None:
            l_out = llm_gr.evaluate(text, evaluation_type=eval_type)
            if l_out.score is not None:
                llm_s = float(l_out.score)
            else:
                llm_s = 1.0 if l_out.status == GuardrailStatus.FAIL else 0.0
        else:
            llm_s = 0.0

        rows_out.append(
            {
                "word_filter": wf,
                "mmbert_prob": m_prob,
                "llm_score": llm_s,
                "label": y,
            }
        )
        if (i + 1) % max(1, total // 10) == 0 or (i + 1) == total:
            print(f"Processed {i + 1}/{total}", flush=True)

    out_df = pd.DataFrame(rows_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out_df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
