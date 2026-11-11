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
    """Rank approved profiles by closeness; preferences are not exclusion rules."""
    ranked = []
    requested_subjects = set(student.get("subjects", [])) - {"Other"}
    level = student.get("study_level", "")
    gender = student.get("teacher_gender", "any")
    budget = float(student.get("budget_max", 0) or 0)
    slots = set(student.get("availability", []))
    formats = set(student.get("formats", []))
    modes = set(student.get("delivery_modes", []))
    for tutor in tutors:
        features = build_pair_features(student, tutor)
        covered = requested_subjects & set(tutor.get("subjects", []))
        features["subject_match"] = len(covered) / len(requested_subjects) if requested_subjects else 1.0
        if not slots:
            features["availability_overlap"] = 1.0
        rate = float(tutor.get("hourly_rate", 0) or 0)
        level_fit = not level or level in tutor.get("grades", [])
        gender_fit = gender == "any" or gender == tutor.get("gender")
        format_fit = not formats or bool(formats & set(tutor.get("formats", [])))
        mode_fit = not modes or bool(modes & set(tutor.get("delivery_modes", [])))
        budget_fit = min(1.0, budget / rate) if budget and rate else 1.0
        preference_score = (
            .35 * features["subject_match"] + .20 * features["availability_overlap"]
            + .10 * features["language_match"] + .10 * features["teaching_style_match"]
            + .07 * level_fit + .08 * budget_fit + .04 * gender_fit
            + .03 * format_fit + .03 * mode_fit
        )
        score, mode = preference_score, "baseline"
        if model_bundle:
            columns = model_bundle.get("feature_columns", FEATURE_COLUMNS)
            row = pd.DataFrame([[features[column] for column in columns]], columns=columns)
            prediction = float(model_bundle["model"].predict_proba(row)[0][1])
            score, mode = .7 * prediction + .3 * preference_score, "model"
        reasons, tradeoffs = [], []
        if covered:
            reasons.append("Teaches " + ", ".join(sorted(covered)))
        if requested_subjects - covered:
            tradeoffs.append("Does not list " + ", ".join(sorted(requested_subjects - covered)))
        overlap = slots & set(tutor.get("availability", []))
        if slots and overlap:
            reasons.append(f"{len(overlap)} of {len(slots)} preferred time slots overlap")
        elif slots:
            tradeoffs.append("No shared time slots; ask about another time")
        if budget and rate > budget:
            tradeoffs.append(f"MUR {rate - budget:,.0f}/hour above your budget")
        elif budget:
            reasons.append("Within your budget")
        if not level_fit:
            tradeoffs.append("Your study level is not listed")
        if not gender_fit:
            tradeoffs.append("Different from your teacher gender preference")
        if features["language_match"]:
            reasons.append("Language preference fits")
        else:
            tradeoffs.append("No preferred language listed")
        if features["teaching_style_match"]:
            reasons.append("Learning style fits")
        if not format_fit:
            tradeoffs.append("Different class format")
        if not mode_fit:
            tradeoffs.append("Different delivery mode")
        ranked.append({**tutor, "features": features, "score": score,
                       "scoring_mode": mode, "reasons": reasons, "tradeoffs": tradeoffs,
                       "availability_matches": bool(overlap) if slots else True})
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked
