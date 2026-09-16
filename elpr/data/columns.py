"""Column names and response codes shared by the dataset, the model and the web API.

Kept apart from `dataset.py` on purpose: that module imports PyTorch, and the web API
needs only these names. Importing it at start-up delayed the port opening by more than
a second, so anything that is just a name lives here.
"""
from __future__ import annotations

CATEGORICAL: dict[str, int] = {
    "gender_code": 2,
    "region_code": 13,
    "education_code": 5,
    "imd_code": 11,
    "age_code": 3,
    "disability_code": 2,
    "module_code": 7,
    "learner_profile": 3,
}

NUMERIC: list[str] = [
    "num_of_prev_attempts",
    "studied_credits",
    "date_registration",
    "early_sessions",
    "early_clicks",
    "early_clicks_per_session",
    "early_span",
    "early_mean_gap",
    "early_gap_std",
    "early_active_days",
]

RESPONSE_INCORRECT, RESPONSE_CORRECT, RESPONSE_UNLABELLED = 0, 1, 2
