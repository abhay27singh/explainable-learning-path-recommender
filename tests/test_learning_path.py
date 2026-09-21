"""The step-by-step path shown instead of the arc roadmap.

Each step must be exactly what the planner would recommend once the earlier steps are
treated as known, so the path is real model output under a stated assumption.
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def service():
    from api.service import Service

    return Service()


def _top(service, state):
    action = service.planners["greedy"].score(state)[:1][0]
    return int(service.explainer.explain(state, action).to_dict()["concept"])


def test_each_step_is_the_top_pick_once_earlier_steps_are_known(service, tmp_path):
    from elpr.db.app_store import AppStore

    service.store = AppStore(tmp_path / "app.db")
    user = service.store.register("pathy", "passw0rd", "P", "student", module="DDD")
    path = service.learning_path(lambda ov: service.registered_state(user, ov), 5)
    ids = [s["concept"] for s in path["steps"]]

    assert 1 <= len(ids) <= 5 and len(set(ids)) == len(ids)
    course = {n["concept"] for n in service.module_graph("DDD")["nodes"]}
    assert set(ids) <= course
    for i, concept in enumerate(ids):
        _, state = service.registered_state(user, tuple(ids[:i]))
        assert _top(service, state) == concept, f"step {i + 1}"


def test_student_path_follows_the_course_from_its_first_week(service, tmp_path):
    """Regression: a new student was told to start at week 9 instead of week 1."""
    from elpr.db.app_store import AppStore

    service.store = AppStore(tmp_path / "app.db")
    user = service.store.register("orderly", "passw0rd", "O", "student", module="CCC", stage="ug")
    weeks = [n["concept"] for n in service.module_graph("CCC")["nodes"]]
    path = service.course_path(user, 5)
    assert [s["concept"] for s in path["steps"]] == weeks[:5]
    assert path["n_done"] == 0 and path["n_weeks"] == len(weeks)


def test_studied_or_passed_weeks_are_done_and_a_later_hard_mark_reopens_them(service, tmp_path):
    """Regression: a week marked done came back later in the path."""
    from elpr.db.app_store import AppStore

    service.store = AppStore(tmp_path / "app.db")
    user = service.store.register("doer", "passw0rd", "D", "student", module="DDD", stage="ug")
    weeks = [n["concept"] for n in service.module_graph("DDD")["nodes"]]
    service.store.add_event(user.id, weeks[0], "study")
    service.store.add_event(user.id, weeks[1], "assessment", True)
    assert service.done_concepts(user) == (weeks[0], weeks[1])
    path = service.course_path(user, 3)
    assert [s["concept"] for s in path["steps"]] == weeks[2:5] and path["n_done"] == 2

    service.store.add_event(user.id, weeks[0], "assessment", False)
    assert service.done_concepts(user) == (weeks[1],)
    assert service.course_path(user, 1)["steps"][0]["concept"] == weeks[0]


def test_weeks_already_known_never_appear_and_steps_are_capped(service):
    learner = 599577
    base = service.learning_path(
        lambda ov: (lambda rs: (rs[0].code_module, rs[1]))(service.state_for(learner, None, 1.0, ov)), 3)
    first = base["steps"][0]["concept"]
    skipped = service.learning_path(
        lambda ov: (lambda rs: (rs[0].code_module, rs[1]))(service.state_for(learner, None, 1.0, ov)),
        3, start=(first,))
    assert first not in [s["concept"] for s in skipped["steps"]]
    many = service.learning_path(
        lambda ov: (lambda rs: (rs[0].code_module, rs[1]))(service.state_for(learner, None, 1.0, ov)), 50)
    assert len(many["steps"]) <= 8
