"""The per-account log an admin reads when a student says "it is not working".

What matters is that it records enough to answer the question, never records a secret,
is admin-only, and does not grow without bound.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def store(tmp_path):
    from elpr.db.app_store import AppStore

    return AppStore(tmp_path / "app.db")


def kinds(entries):
    return [e["kind"] for e in entries]


def test_entries_come_back_newest_first(store):
    user = store.register("logger", "passw0rd", "L", "student", stage="class_10")
    store.log(user.id, "signed in")
    store.log(user.id, "recorded activity", "week concept 3, study")
    entries = store.logs("logger")
    assert kinds(entries) == ["recorded activity", "signed in"]
    assert entries[0]["detail"] == "week concept 3, study"
    assert entries[0]["at"] >= entries[1]["at"]


def test_only_the_last_two_hundred_entries_are_kept(store):
    user = store.register("chatty", "passw0rd", "C", "student", stage="class_10")
    for i in range(store.LOG_KEEP + 25):
        store.log(user.id, "recorded activity", f"event {i}")
    entries = store.logs("chatty", limit=200)
    assert len(entries) == store.LOG_KEEP
    assert entries[0]["detail"] == f"event {store.LOG_KEEP + 24}", "newest survives"


def test_a_long_detail_is_trimmed_rather_than_stored_whole(store):
    user = store.register("verbose", "passw0rd", "V", "student", stage="class_10")
    store.log(user.id, "error 400", "x" * 5000)
    assert len(store.logs("verbose")[0]["detail"]) == 300


def test_logs_are_private_to_each_account_and_die_with_it(store):
    one = store.register("one", "passw0rd", "O", "student", stage="class_10")
    store.register("two", "passw0rd", "T", "student", stage="class_10")
    store.log(one.id, "signed in")
    assert store.logs("two") == []
    assert store.delete_account("one") is True
    assert store.logs("one") == []


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


def test_signing_up_studying_and_failing_to_sign_in_are_all_recorded(main):
    from fastapi import Response

    secret = "hunter2-lemon-drift"
    body = main.Registration(username="student1", password=secret, role="student",
                             stage="ug", module="CCC")
    main.register(body, Response())
    store = main.service.store
    token = store.create_session(store.authenticate("student1", secret).id)

    main.record_study(main.StudyEvent(concept_id=3, kind="study", correct=None), token)
    main.undo(token)
    assert _status(lambda: main.login(
        main.Credentials(username="student1", password="wrong-one"), Response())) == 401

    logged = store.logs("student1")
    entries = kinds(logged)
    assert entries[:3] == ["sign-in failed", "undid last activity", "recorded activity"]
    assert "signed up" in entries
    text = " ".join(e["kind"] + " " + e["detail"] for e in logged)
    assert secret not in text and "wrong-one" not in text, "passwords must never be logged"


def test_the_log_is_admin_only(main):
    from fastapi import Response

    store = main.service.store
    main.register(main.Registration(username="nosy", password="passw0rd", role="student",
                                    stage="class_10"), Response())
    student = store.create_session(store.authenticate("nosy", "passw0rd").id)
    assert _status(lambda: main.admin_account_log("student1", 100, student)) == 403
    assert _status(lambda: main.admin_account_log("student1", 100, None)) == 401

    store.create_admin("boss", "passw0rd")
    admin = store.create_session(store.authenticate("boss", "passw0rd").id)
    assert main.admin_account_log("student1", 100, admin)["entries"]
    assert _status(lambda: main.admin_account_log("ghost", 100, admin)) == 404
