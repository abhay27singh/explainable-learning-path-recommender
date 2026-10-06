"""An adviser opening a student who is still at school.

School students have no course, so the model has no weeks of theirs to rank. Its state
used to fall back to the first course in the dataset, and the adviser saw a confident
Psychology path ("Week 11, revise Week 4 first") for a class 12 student who had never
studied Psychology at all.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException


@pytest.fixture
def main(tmp_path):
    from api import main
    from api.service import Service
    from elpr.db.app_store import AppStore

    main.service = Service()
    main.service.store = AppStore(tmp_path / "app.db")
    store = main.service.store
    ravi = store.register("ravi", "passw0rd", "Ravi", "student", stage="class_12", stream="pcb")
    asha = store.register("asha", "passw0rd", "Asha", "student", module="CCC", stage="ug")
    meera = store.register("meera", "passw0rd", "Meera", "adviser")
    store.approve_adviser("meera")
    for student in (ravi, asha):
        store.join_class(student.id, store.class_code(meera.id))
    return main


def _token(main, name):
    store = main.service.store
    return store.create_session(store.authenticate(name, "passw0rd").id)


def test_an_adviser_is_told_a_school_student_has_no_course_path(main):
    adviser = _token(main, "meera")
    ravi = main.service.store.user_by_username("ravi").student_id
    for call in (lambda: main.learner_path(ravi, 5, None, adviser),
                 lambda: main.mastery(ravi, None, 1.0, None, adviser),
                 lambda: main.recommend(ravi, 3, "greedy", None, 1.0, None, adviser)):
        with pytest.raises(HTTPException) as refused:
            call()
        assert refused.value.status_code == 409
        assert "Class 12" in refused.value.detail and "no weekly path" in refused.value.detail


def test_a_student_on_a_course_still_gets_their_path(main):
    adviser = _token(main, "meera")
    asha = main.service.store.user_by_username("asha").student_id
    record = main.mastery(asha, None, 1.0, None, adviser)
    assert record["module"] == "CCC"
    assert record["display_name"] == "Asha", "the adviser page is titled with the name, not the id"
    assert main.learner_path(asha, 5, None, adviser)["steps"]


def test_the_learner_list_says_where_a_school_student_is(main):
    rows = main.students("", 40, _token(main, "meera"))["registered"]
    ravi = next(r for r in rows if r["display_name"] == "Ravi")
    assert ravi["stage"] == "class_12"
