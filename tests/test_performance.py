"""Speed work, kept honest by tests rather than by feel.

The model is the expensive part: one run took about 60 ms, and building a five-step
path ran it once per step. These tests pin the caching that fixed that, and check the
cache is dropped the moment a student records something new.
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def service():
    from api.service import Service

    return Service()


@pytest.fixture
def student(service, tmp_path):
    from elpr.db.app_store import AppStore

    service.store = AppStore(tmp_path / "app.db")
    return service.store.register("speedy", "pass", "S", "student", module="CCC", stage="ug")


def counting(service, monkeypatch):
    """Count how many times the model actually runs."""
    calls = []
    real = service.engine.mastery
    monkeypatch.setattr(service.engine, "mastery",
                        lambda seq: (calls.append(1), real(seq))[1])
    return calls


def test_a_five_step_path_runs_the_model_once(service, student, monkeypatch):
    calls = counting(service, monkeypatch)
    path = service.course_path(student, 5)
    assert len(path["steps"]) == 5
    assert len(calls) == 1, "one run per learner, not one per step"


def test_recording_activity_invalidates_the_cache(service, student, monkeypatch):
    calls = counting(service, monkeypatch)
    service.registered_state(student)
    service.registered_state(student)
    assert len(calls) == 1

    weeks = [n["concept"] for n in service.module_graph("CCC")["nodes"]]
    service.store.add_event(student.id, weeks[0], "study")
    service.registered_state(student)
    assert len(calls) == 2, "a new event must produce a fresh mastery estimate"

    service.store.set_profile(student.id, {"age_band": "35-55"})
    service.registered_state(student)
    assert len(calls) == 3, "changed background answers must produce a fresh estimate"


def test_adviser_overrides_reuse_one_model_run(service, monkeypatch):
    learner = 599577
    service.state_for(learner, None, 1.0, ())          # warm, outside the count
    calls = counting(service, monkeypatch)
    _, plain = service.state_for(learner, None, 1.0, ())
    _, known = service.state_for(learner, None, 1.0, (5,))
    assert not calls, "marking weeks as known must not re-run the model"
    assert known.mastery[5] == pytest.approx(0.999)
    assert plain.mastery[5] != pytest.approx(0.999), "the cached vector must not be mutated"
