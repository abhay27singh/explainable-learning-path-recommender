"""The shortlist: courses a student saves from the Course Finder.

School students have no course and no weekly path, so the shortlist is what their
dashboard is for. Saving must be idempotent, ordered newest first, and must not
survive the account being deleted.
"""
from __future__ import annotations

import pytest

from elpr import course_finder as cf


@pytest.fixture
def store(tmp_path):
    from elpr.db.app_store import AppStore

    return AppStore(tmp_path / "app.db")


def test_saving_is_idempotent_and_newest_first(store):
    user = store.register("saver", "pass", "S", "student", stage="class_12")
    store.save_course(user.id, "btech_cse")
    store.save_course(user.id, "bsc_cs")
    store.save_course(user.id, "btech_cse")           # saving twice changes nothing
    assert store.saved_courses(user.id) == ["bsc_cs", "btech_cse"]

    assert store.remove_course(user.id, "bsc_cs") is True
    assert store.remove_course(user.id, "bsc_cs") is False
    assert store.saved_courses(user.id) == ["btech_cse"]


def test_shortlists_are_private_to_each_student(store):
    one = store.register("one", "pass", "O", "student", stage="class_12")
    two = store.register("two", "pass", "T", "student", stage="class_10")
    store.save_course(one.id, "btech_cse")
    assert store.saved_courses(two.id) == []


def test_deleting_an_account_deletes_its_shortlist(store):
    user = store.register("leaver", "pass", "L", "student", stage="ug", module="CCC")
    store.save_course(user.id, "mba")
    assert store.delete_account("leaver") is True
    assert store.saved_courses(user.id) == []


def test_every_saved_key_resolves_to_a_full_course():
    for key in ("btech_cse", "mba", "dip_mech", "ba_psych"):
        course = cf.course(key)
        assert course["name"] and course["level_label"] and course["eligibility"]
        assert course["years"] and course["years"][0]["subjects"]
    with pytest.raises(KeyError):
        cf.course("not-a-course")


@pytest.fixture(scope="module")
def main(tmp_path_factory):
    import api.main as main
    from api.service import Service
    from elpr.db.app_store import AppStore

    main.service = Service()
    main.service.store = AppStore(tmp_path_factory.mktemp("api") / "app.db")
    return main


def _status(call) -> int:
    from fastapi import HTTPException

    try:
        call()
    except HTTPException as exc:
        return exc.status_code
    return 200


def test_api_saves_reads_and_removes(main):
    store = main.service.store
    student = store.register("apisaver", "pass", "A", "student", stage="class_12")
    adviser = store.register("apiadviser", "pass", "V", "adviser")
    token = store.create_session(student.id)

    assert main.save_course(main.SaveCourseBody(key="btech_cse"), token) == {"saved": "btech_cse"}
    listed = main.my_saved_courses(token)["courses"]
    assert [c["key"] for c in listed] == ["btech_cse"]
    assert listed[0]["level_label"] == cf.LEVELS["ug"]

    assert _status(lambda: main.save_course(main.SaveCourseBody(key="nope"), token)) == 404
    assert _status(lambda: main.my_saved_courses(store.create_session(adviser.id))) == 400
    assert _status(lambda: main.my_saved_courses(None)) == 401

    assert main.unsave_course("btech_cse", token) == {"removed": True}
    assert main.my_saved_courses(token)["courses"] == []
