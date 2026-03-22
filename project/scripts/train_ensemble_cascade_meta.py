#!/usr/bin/env python3
"""
Fit a logistic regression meta-learner for `submission_ensemble_cascade`.

Expects a CSV with one row per validation example and columns:
  - word_filter   (0/1 or float in [0,1])
  - mmbert_prob   (float in [0,1])
  - llm_score     (float in [0,1])
  - label         (0 = low_risk, 1 = high_risk)

Generate the feature table from a labeled CSV (e.g. ``datasets/seed_validation_set.csv``) with:

  uv run python project/scripts/build_ensemble_cascade_features.py \\
    --input_csv datasets/seed_validation_set.csv \\
    --output_csv datasets/ensemble_cascade_features.csv

Then fit the meta-learner (from ``project/`` so ``models/`` resolves to ``project/models/``):

  cd project && uv run python scripts/train_ensemble_cascade_meta.py \\
    --features_csv ../datasets/ensemble_cascade_features.csv \\
    --output_json models/ensemble_cascade_meta.json \\
    --threshold 0.3

The JSON is loaded automatically by `get_guardrails()` in `src/submission/submission_ensemble_cascade.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Train logistic meta-learner for ensemble cascade.")
    parser.add_argument(
        "--features_csv",
        required=True,
        help="CSV with columns word_filter, mmbert_prob, llm_score, label",
    )
    parser.add_argument(
        "--output_json",
        default=str(_PROJECT / "models" / "ensemble_cascade_meta.json"),
        help="Where to write coef + intercept + threshold",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Recommended risk-averse decision threshold on meta probability",
    )
    args = parser.parse_args()

    try:
        import pandas as pd
        from sklearn.linear_model import LogisticRegression
    except ImportError as e:
        print("Requires pandas and scikit-learn.", e, file=sys.stderr)
        return 1

    path = Path(args.features_csv)
    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    df = pd.read_csv(path)
    required = ("word_filter", "mmbert_prob", "llm_score", "label")
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"CSV missing columns {missing}. Have: {list(df.columns)}", file=sys.stderr)
        return 1

    X = df[list(required[:-1])].astype(float).values
    y = df["label"].astype(int).values
    if set(y.tolist()) - {0, 1}:
        print("label must be 0 or 1 only.", file=sys.stderr)
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
        "feature_order": ["word_filter", "mmbert_prob", "llm_score"],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {out_path}")
    print(f"  coef={coef} intercept={intercept:.6f} threshold={args.threshold}")
    train_acc = clf.score(X, y)
    print(f"  training accuracy (in-sample): {train_acc:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
