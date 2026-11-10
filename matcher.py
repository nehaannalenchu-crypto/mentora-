"""Explainable tutor ranking with optional Logistic Regression ranking scores."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


FEATURE_COLUMNS = [
    "subject_match",
    "availability_overlap",
    "language_match",
    "teaching_style_match",
]
MODEL_PATH = Path(__file__).parent / "models" / "matcher.joblib"


def build_pair_features(student: dict[str, Any], tutor: dict[str, Any]) -> dict[str, float]:
    student_slots = set(student.get("availability", []))
    tutor_slots = set(tutor.get("availability", []))
    preferred_languages = set(student.get("languages", [])) - {"Other"}
    tutor_languages = set(tutor.get("languages", [])) - {"Other"}
    student_styles = set(student.get("learning_styles", []))
    tutor_style = tutor.get("teaching_style", "")
    overlap = len(student_slots & tutor_slots) / len(student_slots) if student_slots else 0.0
    subject_match = bool(set(student.get("subjects", [])) & set(tutor.get("subjects", [])))
    return {
        "subject_match": float(subject_match),
        "availability_overlap": overlap,
        "language_match": float(not preferred_languages or bool(preferred_languages & tutor_languages)),
        "teaching_style_match": float(not student_styles or tutor_style in student_styles),
    }


def load_trained_model() -> dict[str, Any] | None:
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def rank_tutors(
    student: dict[str, Any],
    tutors: list[dict[str, Any]],
    model_bundle: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Filter essential constraints, then rank by match features."""
    ranked: list[dict[str, Any]] = []
    student_subjects = set(student.get("subjects", []))
    student_level = student.get("study_level", "")
    student_gender = student.get("teacher_gender", "any")
    student_formats = set(student.get("formats", []))
    student_modes = set(student.get("delivery_modes", []))
    budget = float(student.get("budget_max", 0) or 0)
    student_slots = set(student.get("availability", []))

    for tutor in tutors:
        tutor_subjects = set(tutor.get("subjects", []))
        tutor_levels = set(tutor.get("grades", []))
        tutor_formats = set(tutor.get("formats", []))
        tutor_modes = set(tutor.get("delivery_modes", []))
        tutor_slots = set(tutor.get("availability", []))
        tutor_rate = float(tutor.get("hourly_rate", 0) or 0)
        tutor_gender = tutor.get("gender", "")

        if not student_subjects.intersection(tutor_subjects):
            continue
        if budget and tutor_rate > budget:
            continue
        if student_level and tutor_levels and student_level not in tutor_levels:
            continue
        if student_gender in {"female", "male"} and tutor_gender in {"female", "male"} and tutor_gender != student_gender:
            continue
        if student_slots and not student_slots.intersection(tutor_slots):
            continue
        if student_formats and tutor_formats and not student_formats.intersection(tutor_formats):
            continue
        if student_modes and tutor_modes and not student_modes.intersection(tutor_modes):
            continue

        features = build_pair_features(student, tutor)
        if model_bundle:
            columns = model_bundle.get("feature_columns", FEATURE_COLUMNS)
            row = pd.DataFrame([[features[column] for column in columns]], columns=columns)
            score = float(model_bundle["model"].predict_proba(row)[0][1])
            mode = "model"
        else:
            # Transparent cold-start score; feedback is needed before ML training.
            score = (
                0.35 * features["subject_match"]
                + 0.30 * features["availability_overlap"]
                + 0.20 * features["language_match"]
                + 0.15 * features["teaching_style_match"]
            )
            mode = "baseline"

        reasons = ["Subject and level fit"]
        if features["availability_overlap"] > 0:
            reasons.append("Available at a time you selected")
        if features["language_match"]:
            reasons.append("Language preference fits")
        if features["teaching_style_match"]:
            reasons.append("Learning style fits")
        if budget:
            reasons.append("Within your budget")
        ranked.append({**tutor, "features": features, "score": score, "scoring_mode": mode, "reasons": reasons})

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked
