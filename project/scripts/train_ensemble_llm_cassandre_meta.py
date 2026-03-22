#!/usr/bin/env python3
"""
Fit a 2-feature logistic meta-learner for `submission_ensemble_llm_cassandre`.

CSV columns:
  - cassandre_score  (float [0,1] from CassandreMultiTurnGuardrail.evaluate().score)
  - llm_score        (float [0,1] from LLMJudgeGuardrail)
  - label            (0 = low_risk, 1 = high_risk)

Generate features from a labeled CSV:

  uv run python project/scripts/build_ensemble_llm_cassandre_features.py \\
    --input_csv datasets/seed_validation_set.csv \\
    --output_csv datasets/ensemble_llm_cassandre_features.csv

Example:

  cd project && uv run python scripts/train_ensemble_llm_cassandre_meta.py \\
    --features_csv ../datasets/ensemble_llm_cassandre_features.csv \\
    --output_json models/ensemble_llm_cassandre_meta.json \\
    --threshold 0.4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Train 2-feature meta-learner (Cassandre + LLM).")
    parser.add_argument("--features_csv", required=True)
    parser.add_argument(
        "--output_json",
        default=str(_PROJECT / "models" / "ensemble_llm_cassandre_meta.json"),
    )
    parser.add_argument("--threshold", type=float, default=0.4)
    args = parser.parse_args()

    try:
        import pandas as pd
        from sklearn.linear_model import LogisticRegression
    except ImportError as e:
        print("Requires pandas and scikit-learn.", e, file=sys.stderr)
        return 1

    path = Path(args.features_csv).expanduser()
    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    df = pd.read_csv(path)
    required = ("cassandre_score", "llm_score", "label")
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Missing columns {missing}. Have: {list(df.columns)}", file=sys.stderr)
        return 1

    X = df[list(required[:-1])].astype(float).values
    y = df["label"].astype(int).values
    if set(y.tolist()) - {0, 1}:
        print("label must be 0 or 1.", file=sys.stderr)
        return 1

    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(X, y)

    coef = clf.coef_.ravel().tolist()
    intercept = float(clf.intercept_.ravel()[0])
    out_path = Path(args.output_json)
    if not out_path.is_absolute():
        out_path = (_PROJECT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "coef": coef,
        "intercept": intercept,
        "threshold": args.threshold,
        "feature_order": ["cassandre_score", "llm_score"],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {out_path}")
    print(f"  coef={coef} intercept={intercept:.6f} threshold={args.threshold}")
    print(f"  training accuracy: {clf.score(X, y):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
