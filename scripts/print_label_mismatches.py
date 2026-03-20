#!/usr/bin/env python3
"""Print rows where `label` differs from `cassandre_grand_composite`."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a predictions CSV and print cases where "
            "`label` != `cassandre_grand_composite`."
        )
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="results/predictions_convokit_v2.7.csv",
        help="Path to CSV file (default: results/predictions_convokit_v2.7.csv)",
    )
    return parser.parse_args()


def to_int(value: str) -> int:
    return int(str(value).strip())


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    total_rows = 0
    mismatch_count = 0

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        required_columns = {"content", "label", "cassandre_grand_composite"}
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Missing required columns: {', '.join(sorted(missing))}"
            )

        for row_idx, row in enumerate(reader, start=1):
            total_rows += 1
            label = to_int(row["label"])
            pred = to_int(row["cassandre_grand_composite"])

            if label != pred:
                mismatch_count += 1
                content_preview = " ".join((row.get("content") or "").split())
                if len(content_preview) > 220:
                    content_preview = content_preview[:217] + "..."

                print(f"[{mismatch_count}] row={row_idx} label={label} pred={pred}")
                print(f"content: {content_preview}")
                print("-" * 80)

    print(f"Done. mismatches={mismatch_count} total_rows={total_rows}")


if __name__ == "__main__":
    main()
