"""Administrator role and its privilege boundary.

The point of these tests is not that admin works, but that the boundary holds: an
admin cannot be created through the web API, and a non-admin cannot reach admin routes.
"""
from __future__ import annotations

import pytest

from elpr.db.app_store import AppStore


@pytest.fixture()
def store(tmp_path):
    return AppStore(tmp_path / "app.db")


def test_registration_cannot_create_an_admin(store):
    """The escalation path that matters: public sign-up must refuse role='admin'."""
    with pytest.raises(ValueError):
        store.register("someone", "a-long-password", "Someone", "admin")


def test_admin_created_from_the_command_line(store):
    user = store.create_admin("root", "a-sufficiently-long-pw", "Root")
    assert user.role == "admin"
    assert user.student_id is None, "an admin is not a learner"
    assert store.authenticate("root", "a-sufficiently-long-pw") is not None


def test_password_minimum_is_the_same_for_every_role(store):
    """Deliberately low for a classroom demo. One constant governs every role."""
    from elpr.db.app_store import MIN_PASSWORD

    short = "x" * (MIN_PASSWORD - 1)
    ok = "x" * MIN_PASSWORD

    with pytest.raises(ValueError):
        store.register("learner", short, "Learner", "student")
    with pytest.raises(ValueError):
        store.create_admin("root", short, "Root")

    assert store.register("learner", ok, "Learner", "student").role == "student"
    assert store.create_admin("root", ok, "Root").role == "admin"


def test_listing_accounts_never_exposes_credentials(store):
    store.create_admin("root", "a-sufficiently-long-pw", "Root")
    store.register("learner", "passw0rd", "Learner", "student", module="DDD")
    for account in store.list_accounts():
        assert "salt" not in account
        assert "password_hash" not in account


def test_deleting_an_account_removes_its_study_history(store):
    user = store.register("learner", "passw0rd", "Learner", "student", module="DDD")
    store.add_event(user.id, concept_id=3, kind="study", correct=True)
    assert store.stats()["n_events"] == 1

    assert store.delete_account("learner") is True
    assert store.stats()["n_events"] == 0
    assert store.authenticate("learner", "passw0rd") is None
    assert store.delete_account("learner") is False, "second delete is a no-op"


def test_password_reset_invalidates_existing_sessions(store):
    user = store.register("learner", "passw0rd", "Learner", "student", module="DDD")
    token = store.create_session(user.id)
    assert store.user_for_session(token) is not None

    store.set_password("learner", "a-new-password")
    assert store.user_for_session(token) is None, "old session must not survive"
    assert store.authenticate("learner", "a-new-password") is not None


def test_role_migration_widens_the_check_constraint(tmp_path):
    """A database written before admin existed must accept admins after upgrade."""
    import sqlite3

    path = tmp_path / "legacy.db"
    con = sqlite3.connect(path)
    con.executescript(
        """CREATE TABLE users (
             id INTEGER PRIMARY KEY AUTOINCREMENT,
             username TEXT UNIQUE NOT NULL,
             display_name TEXT NOT NULL,
             role TEXT NOT NULL CHECK (role IN ('student', 'adviser')),
             salt BLOB NOT NULL, password_hash BLOB NOT NULL,
             module TEXT, student_id INTEGER UNIQUE, created_at REAL NOT NULL);"""
    )
    con.execute(
        "INSERT INTO users (username, display_name, role, salt, password_hash, created_at)"
        " VALUES ('old', 'Old', 'adviser', X'00', X'00', 0)"
    )
    con.commit()
    con.close()

    store = AppStore(path)                       # migration runs here
    store.create_admin("root", "a-sufficiently-long-pw", "Root")
    names = {a["username"] for a in store.list_accounts()}
    assert names == {"old", "root"}, "existing rows must survive the table rebuild"


def test_role_guards_separate_adviser_from_admin(tmp_path, monkeypatch):
    """The HTTP-layer boundary: guards are what stand between roles."""
    from fastapi import HTTPException

    import api.main as main

    store = AppStore(tmp_path / "app.db")
    store.register("adv", "passw0rd", "Adviser", "adviser")
    store.create_admin("root", "a-sufficiently-long-pw", "Root")
    monkeypatch.setattr(main, "_current", lambda token: store.user_for_session(token))

    adviser_token = store.create_session(store.authenticate("adv", "passw0rd").id)
    admin_token = store.create_session(
        store.authenticate("root", "a-sufficiently-long-pw").id)

    # an adviser is refused admin routes
    with pytest.raises(HTTPException) as exc:
        main._require_admin(adviser_token)
    assert exc.value.status_code == 403

    # an admin passes both, because admin is a superset of adviser
    assert main._require_admin(admin_token).role == "admin"
    assert main._require_adviser(admin_token).role == "admin"

    # and a student is refused both
    student = store.register("kid", "passw0rd", "Kid", "student", module="DDD")
    student_token = store.create_session(student.id)
    for guard in (main._require_admin, main._require_adviser):
        with pytest.raises(HTTPException):
            guard(student_token)

    # no session at all is a 401, not a 403
    with pytest.raises(HTTPException) as exc:
        main._require_admin(None)
    assert exc.value.status_code == 401


def test_staff_can_open_any_learner_record_but_students_only_their_own(monkeypatch):
    """Regression: admin could list learners but got 403 opening one, because two
    routes hard-coded role != 'adviser' instead of accepting admin."""
    from fastapi import HTTPException

    import api.main as main
    from elpr.db.app_store import User

    class StubService:
        """Any attribute access means the request got past the permission check."""
        def __getattr__(self, name):
            raise AttributeError("reached past the permission check")

    monkeypatch.setattr(main, "_service", lambda: StubService())

    other = 2364471  # a dataset learner the caller does not own
    calls = {
        "mastery": lambda: main.mastery(other, module=None, upto=1.0, known=None,
                                        elpr_session="t"),
        "recommend": lambda: main.recommend(other, k=3, planner="greedy",
                                            elpr_session="t"),
    }

    def gate(role, student_id=None):
        monkeypatch.setattr(main, "_current",
                            lambda token: User(1, "u", "U", role, None, student_id))
        out = {}
        for name, call in calls.items():
            try:
                call()
                out[name] = "allowed"
            except HTTPException as exc:
                out[name] = exc.status_code
            except Exception:
                out[name] = "allowed"
        return out

    assert gate("admin") == {"mastery": "allowed", "recommend": "allowed"}
    assert gate("adviser") == {"mastery": "allowed", "recommend": "allowed"}
    assert gate("student", student_id=90000099) == {"mastery": 403, "recommend": 403}


def test_a_weak_password_is_refused_at_the_new_floor(tmp_path):
    """Four characters was a classroom setting. Eight is the floor for a real site."""
    from elpr.db.app_store import MIN_PASSWORD, AppStore

    store = AppStore(tmp_path / "app.db")
    assert MIN_PASSWORD == 8
    with pytest.raises(ValueError):
        store.register("shorty", "pass", "S", "student", stage="ug", module="CCC")
    assert store.register("longer", "passw0rd", "L", "student", stage="ug", module="CCC")


def test_sign_in_attempts_are_throttled_per_username(tmp_path):
    """Without account recovery, guessing in a loop is the obvious attack."""
    from elpr.db.app_store import SIGNIN_MAX_FAILURES, AppStore

    store = AppStore(tmp_path / "app.db")
    store.register("target", "passw0rd", "T", "student", stage="ug", module="CCC")
    now = 1_000_000.0

    for _ in range(SIGNIN_MAX_FAILURES):
        assert store.authenticate("target", "wrongpass", now=now) is None
    assert store.locked_out("target", now) > 0
    # even the right password is refused while locked out
    assert store.authenticate("target", "passw0rd", now=now) is None

    later = now + 16 * 60                       # the window has passed
    assert store.locked_out("target", later) == 0
    assert store.authenticate("target", "passw0rd", now=later) is not None
    assert store.locked_out("target", later) == 0, "a good password clears the count"


def test_throttling_is_per_username_not_global(tmp_path):
    from elpr.db.app_store import SIGNIN_MAX_FAILURES, AppStore

    store = AppStore(tmp_path / "app.db")
    store.register("victim", "passw0rd", "V", "student", stage="ug", module="CCC")
    store.register("bystander", "passw0rd", "B", "student", stage="ug", module="CCC")
    now = 2_000_000.0
    for _ in range(SIGNIN_MAX_FAILURES + 2):
        store.authenticate("victim", "nope", now=now)
    assert store.locked_out("victim", now) > 0
    assert store.locked_out("bystander", now) == 0
    assert store.authenticate("bystander", "passw0rd", now=now) is not None
