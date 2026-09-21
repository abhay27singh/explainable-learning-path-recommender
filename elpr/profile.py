"""The learner background form, and how its answers reach the model.

Only attributes the knowledge tracing model was trained on appear here. Every value is
one of OULAD's own categories (the ``feature_vocab`` table of the DuckDB build), listed
in code order, so a form answer maps to exactly the code the model saw in training.

Anything a student cannot know about themselves (their clustered early-engagement
profile, click statistics, registration day) stays at the dataset average. That is the
honest "no information" value, and it is what every registered learner received before
this form existed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from elpr.data.columns import CATEGORICAL, NUMERIC

# form field -> (matrix column, OULAD values in code order, readable labels)
CATEGORICAL_FIELDS: dict[str, tuple[str, list[str], dict[str, str]]] = {
    "age_band": ("age_code", ["0-35", "35-55", "55<="],
                 {"0-35": "Under 35", "35-55": "35 to 55", "55<=": "55 and over"}),
    "gender": ("gender_code", ["F", "M"], {"F": "Female", "M": "Male"}),
    # The dataset's British qualification names, shown as their nearest Indian
    # equivalent. The value sent to the model is unchanged; only the label differs.
    "highest_education": ("education_code", [
        "A Level or Equivalent", "HE Qualification", "Lower Than A Level",
        "No Formal quals", "Post Graduate Qualification"], {
        "No Formal quals": "No formal qualification",
        "Lower Than A Level": "Below class 12",
        "A Level or Equivalent": "Class 12 (higher secondary)",
        "HE Qualification": "Bachelor's degree or diploma",
        "Post Graduate Qualification": "Postgraduate degree"}),
    "region": ("region_code", [
        "East Anglian Region", "East Midlands Region", "Ireland", "London Region",
        "North Region", "North Western Region", "Scotland", "South East Region",
        "South Region", "South West Region", "Wales", "West Midlands Region",
        "Yorkshire Region"], {}),
    "imd_band": ("imd_code", [
        "0-10%", "10-20", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%",
        "70-80%", "80-90%", "90-100%", "unknown"], {
        "0-10%": "0-10% (most deprived)", "10-20": "10-20%",
        "90-100%": "90-100% (least deprived)", "unknown": "Prefer not to say"}),
    "disability": ("disability_code", ["N", "Y"], {"N": "No", "Y": "Yes"}),
}

# form field -> (min, max, step), clamped to the range present in OULAD
NUMERIC_FIELDS: dict[str, tuple[int, int, int]] = {
    "num_of_prev_attempts": (0, 6, 1),
    "studied_credits": (30, 360, 15),
}

FIELD_LABELS = {
    "age_band": "Age",
    "gender": "Gender",
    "highest_education": "Highest education",
    "region": "Region",
    "imd_band": "Deprivation band (UK index)",
    "disability": "Declared disability",
    "num_of_prev_attempts": "Previous attempts at this course",
    "studied_credits": "Credits you are studying",
}

MODULE_CODES = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]

# Questions only a British student could answer. The model was trained on a UK dataset,
# but the people using this site are in India: a UK region, a UK deprivation decile and
# a UK credit load are unanswerable here, and a guess would be worse than a blank. They
# stay at the dataset average, the same honest no-information value used for everything
# a learner cannot know about themselves.
UK_ONLY: tuple[str, ...] = ("region", "imd_band", "studied_credits")
UK_ONLY_NOTE = ("Three questions from the original UK study (region, deprivation band and "
                "credit load) are not asked here, because they have no Indian equivalent. "
                "The model uses the dataset average for them.")


def clean(profile: dict | None) -> dict:
    """Keep only recognised fields with valid values. Unknown keys are dropped."""
    out: dict = {}
    for field, (_, values, _) in CATEGORICAL_FIELDS.items():
        v = (profile or {}).get(field)
        if v in values:
            out[field] = v
    for field, (lo, hi, _) in NUMERIC_FIELDS.items():
        v = (profile or {}).get(field)
        if v is None or v == "":
            continue
        try:
            out[field] = int(min(max(float(v), lo), hi))
        except (TypeError, ValueError):
            continue
    return out


def options(include_uk_only: bool = False) -> list[dict]:
    """Form definition for the front end, without the UK-only questions by default."""
    fields = []
    for field, (_, values, labels) in CATEGORICAL_FIELDS.items():
        if field in UK_ONLY and not include_uk_only:
            continue
        fields.append({
            "field": field, "label": FIELD_LABELS[field], "type": "choice",
            "options": [{"value": v, "label": labels.get(v, v)} for v in values],
        })
    for field, (lo, hi, step) in NUMERIC_FIELDS.items():
        if field in UK_ONLY and not include_uk_only:
            continue
        fields.append({"field": field, "label": FIELD_LABELS[field], "type": "range",
                       "min": lo, "max": hi, "step": step})
    return fields


def _offsets() -> tuple[dict[str, tuple[int, int]], int]:
    """Where each one-hot block sits in a matrix row, and where numerics begin.

    Mirrors the concatenation order of ``build_student_matrix``.
    """
    offsets, pos = {}, 0
    for col, size in CATEGORICAL.items():
        offsets[col] = (pos, size)
        pos += size
    return offsets, pos


class ProfileEncoder:
    """Encode form answers exactly as ``build_student_matrix`` encoded training rows."""

    def __init__(self, features: pd.DataFrame, default: np.ndarray):
        self.default = np.asarray(default, dtype=np.float32)
        self.offsets, self.numeric_start = _offsets()
        numeric = features[NUMERIC].fillna(0).to_numpy(dtype=np.float32)
        numeric[:, 3:] = np.log1p(np.clip(numeric[:, 3:], 0, None))
        self.mean = numeric.mean(axis=0)
        std = numeric.std(axis=0)
        self.std = np.where(std > 0, std, 1.0)

    def _set_onehot(self, row: np.ndarray, column: str, code: int) -> None:
        start, size = self.offsets[column]
        row[start:start + size] = 0.0
        row[start + code] = 1.0

    def encode(self, profile: dict | None, module: str | None = None) -> np.ndarray:
        row = self.default.copy()
        answers = clean(profile)
        for field, (column, values, _) in CATEGORICAL_FIELDS.items():
            if field in answers:
                self._set_onehot(row, column, values.index(answers[field]))
        if module in MODULE_CODES:
            self._set_onehot(row, "module_code", MODULE_CODES.index(module))
        for field in NUMERIC_FIELDS:
            if field in answers:
                i = NUMERIC.index(field)
                # these two columns are standardised without log1p, as in training
                row[self.numeric_start + i] = (answers[field] - self.mean[i]) / self.std[i]
        return row.astype(np.float32)
