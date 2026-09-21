"""Background form.

What matters is not that answers are saved, but that they reach the model encoded
exactly as the training rows were, and that they genuinely change the prediction. A form
whose answers the model ignores would be decoration.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from elpr.data.dataset import NUMERIC, build_student_matrix
from elpr.db.app_store import AppStore
from elpr.profile import (
    CATEGORICAL_FIELDS,
    MODULE_CODES,
    NUMERIC_FIELDS,
    ProfileEncoder,
    _offsets,
    clean,
)


def test_uk_only_questions_are_not_asked_of_indian_students():
    """The model was trained in the UK; the students here are in India. A UK region,
    deprivation decile or credit load cannot be answered honestly, so they are not
    asked and stay at the dataset average."""
    from elpr.profile import UK_ONLY, options as profile_options

    asked = {f["field"] for f in profile_options()}
    assert asked.isdisjoint(UK_ONLY), "no UK-only question reaches a student"
    assert {"age_band", "gender", "highest_education", "disability"} <= asked
    assert {f["field"] for f in profile_options(include_uk_only=True)} >= set(UK_ONLY)


def test_education_options_read_as_indian_qualifications():
    from elpr.profile import options as profile_options

    labels = {o["label"] for f in profile_options()
              if f["field"] == "highest_education" for o in f["options"]}
    assert labels == {"No formal qualification", "Below class 12",
                      "Class 12 (higher secondary)", "Bachelor's degree or diploma",
                      "Postgraduate degree"}
    assert not any("A Level" in l or "HE Qualification" in l for l in labels)


def test_clean_keeps_only_valid_answers():
    out = clean({"gender": "F", "age_band": "not a band", "injected": "x",
                 "studied_credits": "9999", "num_of_prev_attempts": "abc"})
    assert out == {"gender": "F", "studied_credits": 360}


@pytest.fixture(scope="module")
def training():
    from api.service import PROCESSED

    features = pd.read_parquet(PROCESSED / "student_features.parquet")
    matrix, _ = build_student_matrix(features)
    return features, matrix


def test_encoder_reproduces_training_encoding_for_real_learners(training):
    """Encode real learners' answers and compare block by block with the matrix the
    model was trained on. Any mismatch means the form feeds the model wrong numbers."""
    features, matrix = training
    encoder = ProfileEncoder(features, matrix.mean(axis=0))
    offsets, numeric_start = _offsets()

    columns = [c for c, _, _ in CATEGORICAL_FIELDS.values()] + ["module_code"]
    usable = features[
        features[columns].notna().all(axis=1)
        & (features.studied_credits <= NUMERIC_FIELDS["studied_credits"][1])
    ]
    rng = np.random.default_rng(0)
    for i in rng.choice(usable.index.to_numpy(), size=25, replace=False):
        r = features.loc[i]
        profile = {field: values[int(r[column])]
                   for field, (column, values, _) in CATEGORICAL_FIELDS.items()}
        profile.update({field: int(r[field]) for field in NUMERIC_FIELDS})
        row = encoder.encode(profile, MODULE_CODES[int(r["module_code"])])

        for field, (column, _, _) in CATEGORICAL_FIELDS.items():
            s, n = offsets[column]
            assert np.array_equal(row[s:s + n], matrix[i, s:s + n]), f"learner {i}: {field}"
        s, n = offsets["module_code"]
        assert np.array_equal(row[s:s + n], matrix[i, s:s + n]), f"learner {i}: module"
        for field in NUMERIC_FIELDS:
            j = numeric_start + NUMERIC.index(field)
            assert row[j] == pytest.approx(matrix[i, j], abs=1e-5), f"learner {i}: {field}"


def test_unanswered_fields_stay_at_the_dataset_average(training):
    features, matrix = training
    default = matrix.mean(axis=0).astype(np.float32)
    encoder = ProfileEncoder(features, default)

    assert np.array_equal(encoder.encode({}, None), default)

    row = encoder.encode({"gender": "F"}, None)
    s, n = _offsets()[0]["gender_code"]
    changed = set(np.flatnonzero(row != default).tolist())
    assert changed <= set(range(s, s + n)), "only the gender block may move"


def test_profile_is_stored_updated_and_removed_with_the_account(tmp_path):
    store = AppStore(tmp_path / "app.db")
    user = store.register("learner", "passw0rd", "L", "student", module="DDD")
    assert store.get_profile(user.id) == {}

    store.set_profile(user.id, {"gender": "F"})
    store.set_profile(user.id, {"gender": "M", "age_band": "0-35"})
    assert store.get_profile(user.id) == {"gender": "M", "age_band": "0-35"}

    store.delete_account("learner")
    remaining = sqlite3.connect(tmp_path / "app.db").execute(
        "SELECT COUNT(*) FROM learner_profiles").fetchone()[0]
    assert remaining == 0


def test_answers_actually_change_the_prediction(tmp_path):
    """The point of the form. If this fails, the form is decoration."""
    from api.service import Service

    service = Service()
    service.store = AppStore(tmp_path / "app.db")
    user = service.store.register("learner", "passw0rd", "L", "student", module="DDD")

    before = service.engine.mastery(service.registered_sequence(user))
    service.store.set_profile(user.id, {
        "highest_education": "No Formal quals",
        "num_of_prev_attempts": 3,
        "imd_band": "0-10%",
    })
    after = service.engine.mastery(service.registered_sequence(user))

    assert not np.allclose(before, after), "background answers must reach the model"
