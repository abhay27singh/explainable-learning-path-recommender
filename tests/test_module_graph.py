"""Roadmap graph data: every course's weeks and prerequisite links.

The roadmap draws exactly what this returns, so the checks are that nothing leaks
between courses, no link points backwards in time, and the per-course links add up
to the 724 prerequisite links reported in the paper.
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def service():
    from api.service import Service

    return Service()


def test_every_link_stays_inside_its_course_and_runs_forward(service):
    for module in service.modules:
        g = service.module_graph(module)
        week = {n["concept"]: n["week"] for n in g["nodes"]}
        assert g["nodes"], module
        for e in g["edges"]:
            assert e["src"] in week and e["dst"] in week, (module, e)
            assert week[e["src"]] <= week[e["dst"]], (module, e)
            assert e["kinds"], (module, e)


def test_links_add_up_to_the_published_total(service):
    total = sum(len(service.module_graph(m)["edges"]) for m in service.modules)
    assert total == 724
    nodes = sum(len(service.module_graph(m)["nodes"]) for m in service.modules)
    assert nodes == 237


def test_unknown_course_is_rejected(service):
    with pytest.raises(KeyError):
        service.module_graph("ZZZ")
