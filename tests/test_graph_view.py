"""The prerequisite graph on the dashboard marks each week done, current, ready or
locked. Which weeks are done comes from the server, in the same response as the path,
so the graph and the path can never disagree about it.
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def service(tmp_path_factory):
    from api.service import Service
    from elpr.db.app_store import AppStore

    s = Service()
    s.store = AppStore(tmp_path_factory.mktemp("graph") / "app.db")
    return s


def test_the_path_lists_done_weeks_in_course_order(service):
    user = service.store.register("grapher", "passw0rd", "G", "student", module="CCC", stage="ug")
    weeks = [n["concept"] for n in service.module_graph("CCC")["nodes"]]
    service.store.add_event(user.id, weeks[2], "study")
    service.store.add_event(user.id, weeks[0], "assessment", score=70)
    service.store.add_event(user.id, weeks[1], "assessment", correct=False)
    path = service.course_path(user, steps=3)
    assert path["done"] == [weeks[0], weeks[2]], "course order, and a hard week is not done"
    assert path["n_done"] == len(path["done"])
    assert path["steps"][0]["concept"] == weeks[1], "the first unfinished week is next"


def test_every_graph_link_joins_two_weeks_of_the_same_course(service):
    g = service.module_graph("CCC")
    ids = {n["concept"] for n in g["nodes"]}
    assert g["edges"] and all(e["src"] in ids and e["dst"] in ids for e in g["edges"])
    assert all(e["src"] != e["dst"] for e in g["edges"])
