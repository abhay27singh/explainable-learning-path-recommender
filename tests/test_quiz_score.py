"""Entering a quiz score instead of just saying "I passed".

A typed mark decides pass or fail on the same threshold the training data used
(sql/05_events.sql binarises at 40), so what the model sees from a student means the
same as what it learned from. The mark itself is kept, for the student to look back on.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def store(tmp_path):
    from elpr.db.app_store import AppStore

    return AppStore(tmp_path / "app.db")


@pytest.fixture
def student(store):
    return store.register("scorer", "pass", "S", "student", module="CCC", stage="ug")


def test_a_score_at_or_above_the_pass_mark_counts_as_a_pass(store, student):
    store.add_event(student.id, 3, "assessment", score=40)
    store.add_event(student.id, 4, "assessment", score=39.5)
    store.add_event(student.id, 5, "assessment", score=91)
    events = store.events(student.id)
    assert [e["correct"] for e in events] == [1, 0, 1]
    assert [e["score"] for e in events] == [40, 39.5, 91]
    assert store.PASS_MARK == 40


def test_a_score_overrides_a_mistaken_pass_or_fail(store, student):
    store.add_event(student.id, 3, "assessment", correct=True, score=12)
    assert store.events(student.id)[0]["correct"] == 0, "the mark decides, not the claim"


def test_a_score_outside_0_to_100_is_clamped(store, student):
    store.add_event(student.id, 3, "assessment", score=150)
    store.add_event(student.id, 4, "assessment", score=-20)
    assert [e["score"] for e in store.events(student.id)] == [100, 0]


def test_recording_without_a_score_still_works(store, student):
    store.add_event(student.id, 3, "study")
    store.add_event(student.id, 4, "assessment", correct=True)
    events = store.events(student.id)
    assert [e["score"] for e in events] == [None, None]
    assert [e["correct"] for e in events] == [None, 1]


def test_a_score_moves_the_path_on_like_any_pass(tmp_path):
    from api.service import Service
    from elpr.db.app_store import AppStore

    service = Service()
    service.store = AppStore(tmp_path / "app.db")
    user = service.store.register("mover", "pass", "M", "student", module="DDD", stage="ug")
    weeks = [n["concept"] for n in service.module_graph("DDD")["nodes"]]

    service.store.add_event(user.id, weeks[0], "assessment", score=75)
    assert service.done_concepts(user) == (weeks[0],)
    assert service.course_path(user, 1)["steps"][0]["concept"] == weeks[1]

    service.store.add_event(user.id, weeks[1], "assessment", score=20)
    assert service.done_concepts(user) == (weeks[0],), "a failed quiz is not done"
    assert service.course_path(user, 1)["steps"][0]["concept"] == weeks[1]


@pytest.fixture(scope="module")
def main(tmp_path_factory):
    import api.main as main
    from api.service import Service
    from elpr.db.app_store import AppStore

    main.service = Service()
    main.service.store = AppStore(tmp_path_factory.mktemp("api") / "app.db")
    return main


def test_the_api_takes_a_score_and_refuses_a_silly_one(main):
    from fastapi import HTTPException, Response
    from pydantic import ValidationError

    store = main.service.store
    main.register(main.Registration(username="apiscore", password="pass", role="student",
                                    stage="ug", module="CCC"), Response())
    token = store.create_session(store.authenticate("apiscore", "pass").id)

    out = main.record_study(main.StudyEvent(concept_id=3, kind="assessment", score=68), token)
    assert out["ok"] and out["pass_mark"] == 40
    assert store.events(store.authenticate("apiscore", "pass").id)[0]["score"] == 68

    with pytest.raises(ValidationError):
        main.StudyEvent(concept_id=3, kind="assessment", score=140)
    with pytest.raises(HTTPException) as bad:
        main.record_study(main.StudyEvent(concept_id=3, kind="study", score=50), token)
    assert bad.value.status_code == 400, "a score belongs to a test, not to reading"


def test_the_score_is_written_to_the_support_log(main):
    store = main.service.store
    user = store.authenticate("apiscore", "pass")
    token = store.create_session(user.id)
    main.record_study(main.StudyEvent(concept_id=4, kind="assessment", score=55), token)
    assert any("scored 55 out of 100" in e["detail"] for e in store.logs("apiscore"))
