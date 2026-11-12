"""Shared feature extraction, a transparent baseline, and trained ML ranking."""
from __future__ import annotations

import logging
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn
from sklearn.exceptions import InconsistentVersionWarning

FEATURE_COLUMNS = [
    "subject_match", "availability_overlap", "language_match", "teaching_style_match",
    "level_match", "budget_fit", "gender_preference_match", "class_format_match", "delivery_mode_match",
]
FEATURE_LABELS = {
    "subject_match": "Share of requested subjects the tutor teaches",
    "availability_overlap": "Share of preferred time slots that overlap",
    "language_match": "At least one preferred language is shared",
    "teaching_style_match": "The teaching approach matches a preferred style",
    "level_match": "The tutor lists the student's study level",
    "budget_fit": "Budget divided by hourly rate, capped at 1",
    "gender_preference_match": "The tutor matches the optional gender preference",
    "class_format_match": "A preferred class format is offered",
    "delivery_mode_match": "Online or in-person preference is offered",
}
BASELINE_WEIGHTS = dict(zip(FEATURE_COLUMNS, [.35, .20, .10, .10, .07, .08, .04, .03, .03]))
MODEL_PATH = Path(__file__).parent / "models" / "matcher.joblib"
REPORT_PATH = Path(__file__).parent / "models" / "evaluation.json"
SCHEMA_VERSION = 2


def build_pair_features(student: dict[str, Any], tutor: dict[str, Any]) -> dict[str, float]:
    """The same nine features are used for generation, training, export and serving."""
    subjects = set(student.get("subjects", [])) - {"Other"}
    slots = set(student.get("availability", []))
    languages = set(student.get("languages", [])) - {"Other"}
    styles = set(student.get("learning_styles", []))
    level = student.get("study_level", "")
    budget = float(student.get("budget_max", 0) or 0)
    rate = float(tutor.get("hourly_rate", 0) or 0)
    gender = student.get("teacher_gender", "any")
    formats = set(student.get("formats", []))
    modes = set(student.get("delivery_modes", []))
    return {
        "subject_match": len(subjects & set(tutor.get("subjects", []))) / len(subjects) if subjects else 1.0,
        "availability_overlap": len(slots & set(tutor.get("availability", []))) / len(slots) if slots else 1.0,
        "language_match": float(not languages or bool(languages & set(tutor.get("languages", [])))),
        "teaching_style_match": float(not styles or tutor.get("teaching_style") in styles),
        "level_match": float(not level or level in tutor.get("grades", [])),
        "budget_fit": min(1.0, budget / rate) if budget and rate else 1.0,
        "gender_preference_match": float(gender == "any" or gender == tutor.get("gender")),
        "class_format_match": float(not formats or bool(formats & set(tutor.get("formats", [])))),
        "delivery_mode_match": float(not modes or bool(modes & set(tutor.get("delivery_modes", [])))),
    }


def baseline_score(features: dict[str, float]) -> float:
    return sum(BASELINE_WEIGHTS[name] * features[name] for name in FEATURE_COLUMNS)


@lru_cache(maxsize=2)
def _load_model(path: str, modified: int) -> dict[str, Any] | None:
    try:
        # Only load this project's trusted, locally trained model artifact.
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            bundle = joblib.load(path)
        if bundle.get("schema_version") != SCHEMA_VERSION or bundle.get("feature_columns") != FEATURE_COLUMNS:
            raise ValueError("Model feature schema differs; retrain using train_model.py.")
        if bundle.get("sklearn_version") != sklearn.__version__:
            raise ValueError("Model and scikit-learn versions differ; retrain in this environment.")
        if bundle.get("training_source") not in {"simulated", "human_feedback"}:
            raise ValueError("Model training source is not documented.")
        return bundle
    except Exception as error:
        logging.getLogger(__name__).warning("ML model unavailable; using preference baseline: %s", error)
        return None


def load_trained_model() -> dict[str, Any] | None:
    if not MODEL_PATH.exists():
        return None
    return _load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns)


def rank_tutors(student: dict[str, Any], tutors: list[dict[str, Any]], model_bundle: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Rank every supplied approved tutor; partial matches remain visible."""
    ranked = []
    requested_subjects = set(student.get("subjects", [])) - {"Other"}
    budget = float(student.get("budget_max", 0) or 0)
    slots = set(student.get("availability", []))
    predictions = None
    if model_bundle and tutors:
        frame = pd.DataFrame([build_pair_features(student, tutor) for tutor in tutors], columns=FEATURE_COLUMNS)
        predictions = model_bundle["model"].predict_proba(frame)[:, 1]
    for index, tutor in enumerate(tutors):
        features = build_pair_features(student, tutor)
        covered = requested_subjects & set(tutor.get("subjects", []))
        rate = float(tutor.get("hourly_rate", 0) or 0)
        level_fit = bool(features["level_match"])
        gender_fit = bool(features["gender_preference_match"])
        format_fit = bool(features["class_format_match"])
        mode_fit = bool(features["delivery_mode_match"])
        score = float(predictions[index]) if predictions is not None else baseline_score(features)
        mode = "model" if predictions is not None else "baseline"
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
