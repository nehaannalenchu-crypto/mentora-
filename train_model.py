"""Evaluate and train the logistic-regression tutor-fit model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from matcher import FEATURE_COLUMNS, MODEL_PATH


REQUIRED_COLUMNS = ["respondent_id", *FEATURE_COLUMNS, "good_fit"]


def evaluate(y_true, y_pred) -> dict[str, float | list]:
    return {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 3),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 3),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 3),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="CSV exported from real consented survey feedback")
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise SystemExit(f"Missing required CSV columns: {', '.join(missing)}")
    frame = frame[REQUIRED_COLUMNS].dropna().copy()
    frame[FEATURE_COLUMNS] = frame[FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise")
    frame["good_fit"] = pd.to_numeric(frame["good_fit"], errors="raise").astype(int)
    frame["respondent_id"] = frame["respondent_id"].astype(str)

    if not set(frame["good_fit"].unique()).issubset({0, 1}):
        raise SystemExit("good_fit must contain only 0 (not a fit) and 1 (good fit).")
    if len(frame) < 20 or frame["respondent_id"].nunique() < 5:
        raise SystemExit(
            "This pilot needs at least 20 labeled pairs from at least 5 respondents to attempt a grouped split. "
            "More independent respondents are preferable; these minimums do not make the results reliable."
        )
    if frame["good_fit"].nunique() < 2:
        raise SystemExit("Training data must include both positive and negative fit labels.")
    if ((frame["availability_overlap"] < 0) | (frame["availability_overlap"] > 1)).any():
        raise SystemExit("availability_overlap must be a proportion from 0 to 1.")
    if not frame[["subject_match", "language_match", "teaching_style_match"]].isin([0, 1]).all().all():
        raise SystemExit("subject_match, language_match and teaching_style_match must be 0 or 1.")

    X = frame[FEATURE_COLUMNS]
    y = frame["good_fit"]
    groups = frame["respondent_id"]
    split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_index, test_index = next(split.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise SystemExit(
            "The grouped split left one side with only one label. Collect more respondents or use a documented alternative split."
        )

    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42),
    )
    pipeline.fit(X_train, y_train)
    model_predictions = pipeline.predict(X_test)

    # A transparent baseline: good fit only when subject and at least one
    # available time both match. Use the same held-out respondents.
    baseline_predictions = (
        (X_test["subject_match"] == 1) & (X_test["availability_overlap"] > 0)
    ).astype(int)

    print(f"Labeled pairs: {len(frame)} from {groups.nunique()} respondents")
    print(f"Held-out pairs: {len(test_index)} from {groups.iloc[test_index].nunique()} respondents")
    print("Logistic regression:", evaluate(y_test, model_predictions))
    print("Rule-based baseline:", evaluate(y_test, baseline_predictions))
    print("Interpret metrics cautiously: this is a small survey pilot, not proof of general performance.")

    # Save the final serving model, refit on all collected rows only after the
    # held-out evaluation has been recorded above.
    final_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42),
    )
    final_model.fit(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final_model,
            "feature_columns": FEATURE_COLUMNS,
            "training_rows": int(len(frame)),
            "respondents": int(groups.nunique()),
            "data_source": str(args.csv),
        },
        MODEL_PATH,
    )
    print(f"Saved serving model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
