"""The replay of a student's own record, and the real learner shown to visitors.

The replay must agree with the path about what counts as done, or stepping to the end
would show a different week map from the one on the page. The demo must be a real
learner from the research data, the same for everyone, and open without an account.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException


@pytest.fixture(scope="module")
def service(tmp_path_factory):
    from api.service import Service
    from elpr.db.app_store import AppStore

    s = Service()
    s.store = AppStore(tmp_path_factory.mktemp("db") / "app.db")
    return s


def _weeks(service, module="CCC"):
    return [n["concept"] for n in service.module_graph(module)["nodes"]]


def test_the_replay_ends_where_the_path_is(service):
    user = service.store.register("replayer", "passw0rd", "R", "student", module="CCC", stage="ug")
    w = _weeks(service)
    service.store.add_event(user.id, w[0], "study")
    service.store.add_event(user.id, w[1], "assessment", True, 70)
    service.store.add_event(user.id, w[2], "assessment", False, 20)   # found hard: stays
    out = service.replay(user)
    steps = out["steps"]
    assert len(steps) == 4 and steps[0]["entry"] is None and steps[0]["done"] == []
    assert steps[0]["next"] == w[0], "before anything, the course starts at its first week"
    assert [len(s["done"]) for s in steps] == [0, 1, 2, 2]
    assert steps[-1]["done"] == list(service.done_concepts(user))
    assert steps[-1]["next"] == service.course_path(user, 1)["steps"][0]["concept"]
    assert steps[3]["entry"]["score"] == 20


def test_the_replay_carries_no_raw_model_figures(service):
    """For a student's own record the model's figures sit near zero for most weeks; a
    replay of "1% to 3%" would tell a student nothing true about how they are doing."""
    user = service.store.user_by_username("replayer")
    for step in service.replay(user)["steps"]:
        assert set(step) == {"entry", "done", "next"}


def test_a_long_record_replays_its_latest_entries(service, monkeypatch):
    user = service.store.register("longrecord", "passw0rd", "L", "student", module="CCC", stage="ug")
    for c in _weeks(service)[:5]:
        service.store.add_event(user.id, c, "study")
    monkeypatch.setattr(service, "REPLAY_MAX", 2)
    out = service.replay(user)
    assert out["skipped"] == 3 and len(out["steps"]) == 3
    assert len(out["steps"][0]["done"]) == 3, "earlier entries are in place at step 0"


def test_only_a_student_on_a_course_has_a_replay(service):
    from api import main

    main.service = service
    store = service.store
    kid = store.register("schoolkid", "passw0rd", "K", "student", stage="class_12", stream="pcm")
    adv = store.register("replayadv", "passw0rd", "A", "adviser")
    for user, code in ((kid, 409), (adv, 400)):
        with pytest.raises(HTTPException) as refused:
            main.my_replay(store.create_session(user.id))
        assert refused.value.status_code == code
    assert any(m == "GET" and p == "/api/me/replay" for m, p, _, _ in main.RATE_LIMITS)


def test_the_demo_is_a_real_learner_and_open_to_everyone(service):
    from api import main

    main.service = service
    d = main.demo()
    assert d["mastery"]["student"] == service.DEMO_LEARNER
    assert service.DEMO_LEARNER < 90_000_000, "a dataset learner, never a registered student"
    assert d["path"]["steps"] and d["upto"] == 0.5
    assert main.demo() is d, "fixed research data: computed once"
