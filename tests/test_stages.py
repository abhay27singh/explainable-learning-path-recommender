"""Study levels: class 10 to postgraduate.

The level decides what the Course Finder offers (only the next level up) and whether a
student has a course with a week-by-week path. Research results are for the admin only.
"""
from __future__ import annotations

import sqlite3

import pytest

from elpr import course_finder as cf


def test_finder_offers_only_the_next_level_from_each_stage():
    o = cf.options()
    assert [s["value"] for s in o["stages"]] == list(cf.STAGES)
    # every rung also reaches the skill path, which is what NEP 2020 asks for
    assert o["next_levels"]["class_10"] == ["diploma_10", "skill"]
    assert o["next_levels"]["class_12"] == ["diploma_12", "ug", "skill"]
    assert o["next_levels"]["ug"] == ["pg", "skill"]
    assert o["next_levels"]["pg"] == ["skill"]
    assert set(cf.NEXT_LEVELS) == set(cf.STAGES)
    assert {lv for v in cf.NEXT_LEVELS.values() for lv in v} <= set(cf.LEVELS)
    assert {s["value"] for s in o["stages"] if s["has_course"]} == set(cf.COURSE_STAGES)


def test_stage_is_added_to_old_databases_stored_and_changed(tmp_path):
    from elpr.db.app_store import AppStore

    path = tmp_path / "app.db"
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL,"
        " display_name TEXT NOT NULL,"
        " role TEXT NOT NULL CHECK (role IN ('student','adviser','admin')),"
        " salt BLOB NOT NULL, password_hash BLOB NOT NULL, module TEXT,"
        " student_id INTEGER UNIQUE, created_at REAL NOT NULL)")
    con.commit()
    con.close()

    store = AppStore(path)
    user = store.register("school", "passw0rd", "S", "student", stage="class_12")
    signed_in = store.authenticate("school", "passw0rd")
    assert (signed_in.stage, signed_in.module) == ("class_12", None)
    store.set_studies(user.id, "ug", "CCC")
    again = store.authenticate("school", "passw0rd")
    assert (again.stage, again.module) == ("ug", "CCC")


@pytest.fixture(scope="module")
def main(tmp_path_factory):
    """The API module with a throwaway account store. Route functions are called
    directly with a session token, the way FastAPI calls them."""
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


def _register(main, username, **extra):
    from fastapi import Response

    body = main.Registration(username=username, password="passw0rd", role="student", **extra)
    return main.register(body, Response())


def _token(main, username):
    user = main.service.store.authenticate(username, "passw0rd")
    return main.service.store.create_session(user.id)


def test_registration_needs_a_level_and_a_course_only_for_course_levels(main):
    assert _status(lambda: _register(main, "nolevel", module="CCC")) == 400
    assert _status(lambda: _register(main, "nocourse", stage="ug")) == 400
    school = _register(main, "schooler", stage="class_12", module="CCC")
    assert school["stage"] == "class_12" and school["module"] is None
    assert _status(lambda: main.my_path(5, None, _token(main, "schooler"))) == 400


def test_finder_rejects_levels_that_are_not_next_for_the_student(main):
    _register(main, "undergrad", stage="ug", module="CCC")
    token = _token(main, "undergrad")
    same = main.FinderBody(level="ug", stream="pcm", interests=["coding"])
    assert _status(lambda: main.course_finder_suggest(same, token)) == 400
    nxt = main.FinderBody(level="pg", degree="bcom", interests=["business"])
    assert main.course_finder_suggest(nxt, token)["courses"]

    moved = main.update_studies(main.StudiesBody(stage="pg", module="CCC"), token)
    assert moved["stage"] == "pg"
    guest = main.FinderBody(level="diploma_10", interests=["coding"])
    assert main.course_finder_suggest(guest, None)["courses"]


def test_research_results_are_for_the_admin_only(main):
    assert _status(lambda: main.metrics(None)) == 401
    _register(main, "curious", stage="ug", module="CCC")
    assert _status(lambda: main.metrics(_token(main, "curious"))) == 403
    main.service.store.create_admin("boss", "passw0rd")
    assert _status(lambda: main.metrics(_token(main, "boss"))) == 200
