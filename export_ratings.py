"""Export anonymous, labeled feedback from the local app database."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

from matcher import FEATURE_COLUMNS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Where to write the ratings CSV")
    parser.add_argument(
        "--database",
        type=Path,
        default=Path(__file__).parent / "instance" / "matcher.sqlite3",
        help="App SQLite database path",
    )
    args = parser.parse_args()
    if not args.database.exists():
        raise SystemExit(f"No app database found at {args.database}")

    with sqlite3.connect(args.database) as db:
        frame = pd.read_sql_query(
            """SELECT s.respondent_id, s.feature_json, f.good_fit
               FROM feedback AS f
               JOIN shown_matches AS s
                 ON s.request_id = f.request_id AND s.tutor_id = f.tutor_id
               WHERE f.good_fit IS NOT NULL
               ORDER BY f.created_at""",
            db,
        )
    if frame.empty:
        raise SystemExit("No positive or negative feedback yet; neutral ratings are excluded from training.")

    features = pd.json_normalize(frame.pop("feature_json").map(__import__("json").loads))
    output = pd.concat([frame.reset_index(drop=True), features], axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output[["respondent_id", *FEATURE_COLUMNS, "good_fit"]].to_csv(args.output, index=False)
    print(f"Exported {len(output)} labeled pairs from {output['respondent_id'].nunique()} anonymous respondents to {args.output}")


if __name__ == "__main__":
    main()
