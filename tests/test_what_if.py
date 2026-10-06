"""The student's What-if: the model run again as if they had passed some weeks.

It is a preview, so the one thing it must never do is write anything. It also must not
describe a course the student is not on, and its statuses have to follow the same rule
as the prerequisite graph: a week is ready when everything it builds on is done.
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
    return main


def _student(main, name="asha", **kw):
    store = main.service.store
    store.register(name, "passw0rd", name, "student", **({"module": "CCC", "stage": "ug"} | kw))
    user = store.authenticate(name, "passw0rd")
    return user, store.create_session(user.id)


def _weeks(main):
    return [n["concept"] for n in main.service.module_graph("CCC")["nodes"]]


def test_a_what_if_writes_nothing(main):
    user, token = _student(main)
    weeks = _weeks(main)
    main.service.store.add_event(user.id, weeks[0], "study")
    before = main.service.store.events(user.id)
    logs = len(main.service.store.logs("asha"))
    main.my_what_if(f"{weeks[1]},{weeks[2]}", token)
    assert main.service.store.events(user.id) == before
    assert len(main.service.store.logs("asha")) == logs, "nothing happened, so nothing is logged"


def test_the_picked_weeks_count_as_done_and_the_path_moves_on(main):
    _, token = _student(main)
    weeks = _weeks(main)
    out = main.my_what_if(f"{weeks[0]},{weeks[1]}", token)
    after = {r["concept"]: r["status_after"] for r in out["weeks"]}
    assert after[weeks[0]] == after[weeks[1]] == "done"
    assert out["next_before"] == weeks[0] and out["next_after"] == weeks[2]
    assert len(out["weeks"]) == len(weeks)


def test_ready_follows_the_graph_rule(main):
    """Newly ready means every week it builds on is now done, and it was not before."""
    _, token = _student(main)
    weeks = _weeks(main)
    out = main.my_what_if(str(weeks[0]), token)
    preds = {}
    for e in main.service.module_graph("CCC")["edges"]:
        preds.setdefault(e["dst"], set()).add(e["src"])
    for c in out["newly_ready"]:
        assert preds.get(c, set()) <= {weeks[0]}
        assert weeks[0] in preds.get(c, set())


def test_the_model_is_run_again_on_the_longer_record(main):
    _, token = _student(main)
    out = main.my_what_if(str(_weeks(main)[2]), token)
    assert out["average_after"] != out["average_before"]
    assert all(0 <= r["after"] <= 1 for r in out["weeks"])


def test_weeks_outside_the_course_or_too_many_are_refused(main):
    _, token = _student(main)
    other = main.service.module_graph("AAA")["nodes"][0]["concept"]
    with pytest.raises(HTTPException) as outside:
        main.my_what_if(str(other), token)
    assert outside.value.status_code == 400
    with pytest.raises(HTTPException) as many:
        main.my_what_if(",".join(str(c) for c in _weeks(main)[:7]), token)
    assert many.value.status_code == 400


def test_a_school_student_has_no_what_if(main):
    _, token = _student(main, "ravi", module=None, stage="class_12", stream="pcm")
    with pytest.raises(HTTPException) as refused:
        main.my_what_if(str(_weeks(main)[0]), token)
    assert refused.value.status_code == 409
