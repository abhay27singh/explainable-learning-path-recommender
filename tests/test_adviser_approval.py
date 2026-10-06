"""An adviser account sees students only once an admin has approved it.

Regression: anyone could choose "Academic adviser" on the sign-up page and at once read
every registered student's name, level, study history and path, and the notes advisers
had written about them. Many of those students are under 18. A new adviser now waits.
"""
from __future__ import annotations

import sqlite3

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
    store.register("asha", "passw0rd", "Asha", "student", module="CCC", stage="ug")
    store.register("newadv", "passw0rd", "New Adviser", "adviser")
    store.create_admin("boss", "passw0rd", "Boss")
    return main


def _token(main, name):
    store = main.service.store
    return store.create_session(store.authenticate(name, "passw0rd").id)


def _status(call):
    try:
        call()
    except HTTPException as exc:
        return exc.status_code
    return 200


def test_a_new_adviser_can_see_no_student_until_approved(main):
    token = _token(main, "newadv")
    asha = main.service.store.user_by_username("asha").student_id
    calls = {
        "list": lambda: main.students("", 40, token),
        "overview": lambda: main.overview(token),
        "path": lambda: main.learner_path(asha, 5, None, token),
        "record": lambda: main.mastery(asha, None, 1.0, None, token),
        "recommend": lambda: main.recommend(asha, 3, "greedy", None, 1.0, None, token),
        "notes": lambda: main.learner_notes(asha, token),
        "write a note": lambda: main.add_learner_note(asha, main.NoteBody(text="hi"), token),
    }
    assert {name: _status(call) for name, call in calls.items()} == dict.fromkeys(calls, 403)
    assert main.me(token)["approved"] is False, "the page needs this to explain the wait"

    main.approve_adviser("newadv", _token(main, "boss"))
    assert {name: _status(call) for name, call in calls.items()} == dict.fromkeys(calls, 200)


def test_only_an_admin_approves_and_only_advisers_can_be_approved(main):
    assert _status(lambda: main.approve_adviser("newadv", _token(main, "asha"))) == 403
    assert _status(lambda: main.approve_adviser("newadv", _token(main, "newadv"))) == 403
    assert _status(lambda: main.approve_adviser("asha", _token(main, "boss"))) == 404
    assert _status(lambda: main.approve_adviser("nobody", _token(main, "boss"))) == 404


def test_the_approval_is_in_both_logs(main):
    main.approve_adviser("newadv", _token(main, "boss"))
    store = main.service.store
    adviser_log = [e["kind"] for e in store.logs("newadv")]
    admin_log = [e["kind"] for e in store.logs("boss")]
    assert "approved as an adviser" in adviser_log and "approved an adviser" in admin_log


def test_students_and_admins_never_wait(main):
    store = main.service.store
    assert store.user_by_username("asha").approved
    assert store.user_by_username("boss").approved
    accounts = {a["username"]: a["approved"] for a in store.list_accounts()}
    assert accounts == {"asha": 1, "newadv": 0, "boss": 1}, "the admin page shows who waits"


def test_advisers_from_before_approval_existed_keep_their_access(tmp_path):
    """The column is added to an existing database as approved, so nobody already using
    the site is locked out by the update."""
    from elpr.db.app_store import AppStore

    path = tmp_path / "old.db"
    AppStore(path).register("oldadv", "passw0rd", "Old", "adviser")
    con = sqlite3.connect(path)
    con.execute("ALTER TABLE users DROP COLUMN approved")
    con.commit()
    con.close()
    assert AppStore(path).user_by_username("oldadv").approved
