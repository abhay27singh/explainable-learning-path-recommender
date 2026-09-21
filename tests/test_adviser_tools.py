"""Adviser tools: notes on a learner, and a flag for students who have gone quiet.

Notes are the adviser's own record, so a student must never read them and an adviser
must not delete a colleague's. The quiet flag counts from real activity, never from an
estimate, and a student who signed up and never started counts from sign-up.
"""
from __future__ import annotations

import time

import pytest


@pytest.fixture
def store(tmp_path):
    from elpr.db.app_store import AppStore

    return AppStore(tmp_path / "app.db")


DAY = 86400


def test_notes_are_kept_newest_first_with_their_author(store):
    adviser = store.register("adv", "passw0rd", "Dr Adviser", "adviser")
    store.add_note(599577, adviser.id, "Called about missed assessments.")
    store.add_note(599577, adviser.id, "Agreed to restart at week 12.")
    notes = store.notes(599577)
    assert [n["text"] for n in notes] == ["Agreed to restart at week 12.",
                                          "Called about missed assessments."]
    assert notes[0]["author"] == "Dr Adviser"
    assert store.notes(123456) == [], "notes belong to one learner only"


def test_an_empty_note_is_refused_and_a_long_one_is_trimmed(store):
    adviser = store.register("adv2", "passw0rd", "A", "adviser")
    with pytest.raises(ValueError):
        store.add_note(1, adviser.id, "   ")
    store.add_note(1, adviser.id, "x" * 5000)
    assert len(store.notes(1)[0]["text"]) == store.NOTE_MAX


def test_an_adviser_can_only_delete_their_own_note(store):
    one = store.register("adv3", "passw0rd", "One", "adviser")
    two = store.register("adv4", "passw0rd", "Two", "adviser")
    note = store.add_note(42, one.id, "Mine.")
    assert store.delete_note(note, two.id) is False, "not yours to delete"
    assert store.delete_note(note, one.id) is True
    assert store.notes(42) == []


def test_a_student_is_quiet_after_a_week_without_activity(store):
    now = time.time()
    active = store.register("busy", "passw0rd", "Busy", "student", module="CCC", stage="ug")
    silent = store.register("silent", "passw0rd", "Silent", "student", module="CCC", stage="ug")
    store.add_event(active.id, 3, "study")
    store.add_event(silent.id, 3, "study")

    import sqlite3
    con = sqlite3.connect(store.path)
    con.execute("UPDATE study_events SET created_at = ? WHERE user_id = ?",
                (now - 30 * DAY, silent.id))
    con.commit()
    con.close()

    by_name = {r["username"]: r for r in store.registered_students(now=now)}
    assert by_name["busy"]["quiet"] is False and by_name["busy"]["quiet_days"] == 0
    assert by_name["silent"]["quiet"] is True and by_name["silent"]["quiet_days"] == 30
    assert by_name["silent"]["never_started"] is False


def test_a_student_who_never_started_is_counted_from_sign_up(store):
    now = time.time()
    user = store.register("newbie", "passw0rd", "New", "student", module="CCC", stage="ug")
    import sqlite3
    con = sqlite3.connect(store.path)
    con.execute("UPDATE users SET created_at = ? WHERE id = ?", (now - 10 * DAY, user.id))
    con.commit()
    con.close()

    row = store.registered_students(now=now)[0]
    assert row["never_started"] is True and row["quiet"] is True and row["quiet_days"] == 10


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


def test_notes_are_for_advisers_and_admins_only(main):
    from fastapi import Response

    store = main.service.store
    adviser = store.register("apiadv", "passw0rd", "Dr API", "adviser")
    adviser_token = store.create_session(adviser.id)
    main.register(main.Registration(username="pupil", password="passw0rd", role="student",
                                    stage="ug", module="CCC"), Response())
    pupil_token = store.create_session(store.authenticate("pupil", "passw0rd").id)

    out = main.add_learner_note(599577, main.NoteBody(text="Follow up next week"), adviser_token)
    assert out["notes"][0]["text"] == "Follow up next week"
    assert main.learner_notes(599577, adviser_token)["notes"]

    assert _status(lambda: main.learner_notes(599577, pupil_token)) == 403
    assert _status(lambda: main.add_learner_note(
        599577, main.NoteBody(text="sneaky"), pupil_token)) == 403
    assert _status(lambda: main.learner_notes(599577, None)) == 401

    assert _status(lambda: main.delete_learner_note(599577, 999, adviser_token)) == 404
    assert main.delete_learner_note(599577, out["id"], adviser_token)["notes"] == []
