"""Invites and classes: an adviser sees a student only if both sides chose it.

The admin chooses the adviser: an adviser can only sign up with a one-time invite the
admin made for them, and then waits for the admin's approval. The student chooses the
adviser: they join the adviser's class with its code, and can leave it. An approved
adviser approved by mistake therefore sees a handful of students, never all of them.
"""
from __future__ import annotations

import sqlite3

import pytest
from fastapi import HTTPException, Response


@pytest.fixture
def main(tmp_path):
    from api import main
    from api.service import Service
    from elpr.db.app_store import AppStore

    main.service = Service()
    main.service.store = AppStore(tmp_path / "app.db")
    main.service.store.create_admin("boss", "passw0rd", "Boss")
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


def _sign_up(main, username, role, **extra):
    body = main.Registration(username=username, password="passw0rd", role=role,
                             display_name=username.title(), **extra)
    return main.register(body, Response())


def _adviser(main, username):
    """An adviser the way a real one arrives: invited, signed up, then approved."""
    code = main.create_invite(main.InviteBody(note=f"for {username}"), _token(main, "boss"))["code"]
    _sign_up(main, username, "adviser", invite=code)
    main.approve_adviser(username, _token(main, "boss"))
    return main.my_class(_token(main, username))["code"]


def test_an_adviser_cannot_sign_up_without_an_invite(main):
    assert _status(lambda: _sign_up(main, "walkin", "adviser")) == 400
    assert _status(lambda: _sign_up(main, "guesser", "adviser", invite="ABCD-EFGH-JKMN")) == 400
    assert main.service.store.user_by_username("walkin") is None
    assert main.service.store.user_by_username("guesser") is None


def test_an_invite_works_once_and_only_until_it_expires(main):
    import time

    store = main.service.store
    boss = store.user_by_username("boss")
    code = store.create_invite(boss.id, "Mrs Rao")["code"]
    assert _sign_up(main, "rao", "adviser", invite=code.lower())["approved"] is False
    assert _status(lambda: _sign_up(main, "rao2", "adviser", invite=code)) == 400

    old = store.create_invite(boss.id, "Mr Late", now=time.time() - 8 * 86400)["code"]
    assert _status(lambda: _sign_up(main, "late", "adviser", invite=old)) == 400
    states = {i["note"]: i["state"] for i in store.invites()}
    assert states == {"Mrs Rao": "used", "Mr Late": "expired"}


def test_the_database_never_holds_an_invite_code(main, tmp_path):
    """A copy of the database must not be a way to sign up as an adviser."""
    code = main.create_invite(main.InviteBody(note="Ms Iyer"), _token(main, "boss"))["code"]
    dump = "\n".join(sqlite3.connect(tmp_path / "app.db").iterdump())
    assert code not in dump and code.replace("-", "") not in dump


def test_only_an_admin_makes_or_withdraws_invites(main):
    _adviser(main, "rao")
    _sign_up(main, "kid", "student", stage="class_10")
    for name in ("rao", "kid"):
        assert _status(lambda: main.create_invite(main.InviteBody(note="x"), _token(main, name))) == 403
        assert _status(lambda: main.list_invites(_token(main, name))) == 403

    boss = _token(main, "boss")
    spare = main.create_invite(main.InviteBody(note="Mr Spare"), boss)
    assert main.revoke_invite(spare["id"], boss) == {"revoked": spare["id"]}
    assert _status(lambda: _sign_up(main, "spare", "adviser", invite=spare["code"])) == 400
    used = next(i for i in main.list_invites(boss)["invites"] if i["state"] == "used")
    assert _status(lambda: main.revoke_invite(used["id"], boss)) == 404, "a used invite is history"


def test_the_admin_sees_who_each_waiting_adviser_was_invited_as(main):
    code = main.create_invite(main.InviteBody(note="Mrs Rao, physics"), _token(main, "boss"))["code"]
    _sign_up(main, "rao", "adviser", invite=code)
    row = next(a for a in main.service.store.list_accounts() if a["username"] == "rao")
    assert row["invited_for"] == "Mrs Rao, physics" and row["approved"] == 0


def test_an_adviser_sees_only_their_own_class(main):
    rao_code, sen_code = _adviser(main, "rao"), _adviser(main, "sen")
    _sign_up(main, "asha", "student", stage="ug", module="CCC", class_code=rao_code)
    _sign_up(main, "ravi", "student", stage="ug", module="CCC", class_code=sen_code)
    store = main.service.store
    asha = store.user_by_username("asha").student_id
    ravi = store.user_by_username("ravi").student_id
    rao = _token(main, "rao")

    assert [r["id_student"] for r in main.students("", 40, rao)["registered"]] == [asha]
    overview = main.overview(rao)
    assert [r["student_id"] for r in overview["registered"]] == [asha]
    assert overview["stats"]["n_students"] == 1, "counts are of the class, not the site"
    assert _status(lambda: main.mastery(asha, None, 1.0, None, rao)) == 200
    for call in (lambda: main.mastery(ravi, None, 1.0, None, rao),
                 lambda: main.learner_path(ravi, 5, None, rao),
                 lambda: main.recommend(ravi, 3, "greedy", None, 1.0, None, rao),
                 lambda: main.learner_notes(ravi, rao),
                 lambda: main.add_learner_note(ravi, main.NoteBody(text="hi"), rao)):
        assert _status(call) == 403
    # The anonymised dataset learners are open to every approved adviser, as before.
    assert _status(lambda: main.mastery(599577, None, 1.0, None, rao)) == 200
    # The admin still sees everyone.
    assert len(main.students("", 40, _token(main, "boss"))["registered"]) == 2


def test_a_student_can_join_and_leave_a_class(main):
    code = _adviser(main, "rao")
    _sign_up(main, "asha", "student", stage="ug", module="CCC")
    store = main.service.store
    asha = store.user_by_username("asha")
    me, rao = _token(main, "asha"), _token(main, "rao")

    assert main.my_advisers(me)["advisers"] == []
    assert _status(lambda: main.mastery(asha.student_id, None, 1.0, None, rao)) == 403
    joined = main.join_class(main.JoinBody(code=code.lower().replace("-", " ")), me)
    assert joined["joined"] == "Rao" and [a["display_name"] for a in joined["advisers"]] == ["Rao"]
    assert _status(lambda: main.mastery(asha.student_id, None, 1.0, None, rao)) == 200

    adviser_id = joined["advisers"][0]["id"]
    assert main.leave_class(adviser_id, me) == {"advisers": []}
    assert _status(lambda: main.mastery(asha.student_id, None, 1.0, None, rao)) == 403
    assert _status(lambda: main.leave_class(adviser_id, me)) == 404


def test_a_wrong_class_code_at_sign_up_makes_no_account(main):
    """Better to say so than to sign the student up and quietly leave them unlinked."""
    assert _status(lambda: _sign_up(main, "asha", "student", stage="class_10",
                                    class_code="ZZZZ-ZZZZ")) == 400
    assert main.service.store.user_by_username("asha") is None


def test_a_new_class_code_keeps_the_class_and_retires_the_old_code(main):
    old = _adviser(main, "rao")
    _sign_up(main, "asha", "student", stage="ug", module="CCC", class_code=old)
    new = main.new_class_code(_token(main, "rao"))["code"]
    assert new != old
    _sign_up(main, "ravi", "student", stage="class_10")
    ravi = _token(main, "ravi")
    assert _status(lambda: main.join_class(main.JoinBody(code=old), ravi)) == 400
    assert _status(lambda: main.join_class(main.JoinBody(code=new), ravi)) == 200
    assert len(main.students("", 40, _token(main, "rao"))["registered"]) == 2


def test_a_waiting_adviser_has_no_class_to_join(main):
    code = main.create_invite(main.InviteBody(note="Mrs Rao"), _token(main, "boss"))["code"]
    _sign_up(main, "rao", "adviser", invite=code)
    assert _status(lambda: main.my_class(_token(main, "rao"))) == 403


def test_guessing_class_codes_is_rate_limited():
    from api.main import RATE_LIMITS
    assert any(m == "POST" and prefix == "/api/me/advisers" and limit.limit <= 10
               for m, prefix, limit, _ in RATE_LIMITS)


def test_deleting_an_account_removes_its_class_links(main):
    code = _adviser(main, "rao")
    _sign_up(main, "asha", "student", stage="ug", module="CCC", class_code=code)
    store = main.service.store
    assert store.delete_account("asha")
    assert main.students("", 40, _token(main, "rao"))["registered"] == []
    assert store.delete_account("rao")
    assert [i["state"] for i in store.invites()] == ["used"], "a deleted adviser's invite stays used"


def test_advisers_from_before_classes_keep_the_students_they_could_see(tmp_path):
    """The first start with classes links existing students to existing approved
    advisers, so nothing in use breaks. Later sign-ups have to join by code."""
    from elpr.db.app_store import AppStore

    path = tmp_path / "old.db"
    store = AppStore(path)
    adviser = store.register("oldadv", "passw0rd", "Old", "adviser")
    store.approve_adviser("oldadv")
    student = store.register("oldkid", "passw0rd", "Kid", "student", stage="class_10")
    con = sqlite3.connect(path)
    con.execute("DROP TABLE adviser_students")
    con.commit()
    con.close()

    store = AppStore(path)
    assert store.adviser_can_see(adviser.id, student.student_id)
    late = store.register("newkid", "passw0rd", "New", "student", stage="class_10")
    assert not store.adviser_can_see(adviser.id, late.student_id)
