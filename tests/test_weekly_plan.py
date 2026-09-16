"""Spreading a course over the number of weeks a student chooses.

The load must be as even as the subject list allows, no week may be empty, and the
plan must never quietly drop or duplicate a subject.
"""
from __future__ import annotations

import pytest

from elpr import course_finder as cf


def flat(plan):
    return [s["name"] for w in plan["weeks"] for s in w["subjects"]]


def test_every_subject_appears_once_in_order():
    plan = cf.weekly_plan("btech_cse", 24)
    course = [n for _, names in cf._BY_KEY["btech_cse"].years for n in names
              if not cf._NOT_A_TOPIC.search(n)]
    assert flat(plan) == course, "subjects keep their order and none are lost"
    assert plan["n_subjects"] == len(course)
    assert [w["week"] for w in plan["weeks"]] == list(range(1, plan["n_weeks"] + 1))


def test_the_load_is_spread_as_evenly_as_the_list_allows():
    for weeks in (4, 8, 12, 24, 36):
        plan = cf.weekly_plan("btech_cse", weeks)
        sizes = [len(w["subjects"]) for w in plan["weeks"]]
        assert min(sizes) >= 1, "no empty weeks"
        assert max(sizes) - min(sizes) <= 1, f"uneven load at {weeks} weeks: {sizes}"
        assert sum(sizes) == plan["n_subjects"]


def test_asking_for_more_weeks_than_subjects_is_capped():
    plan = cf.weekly_plan("dca", 200)
    assert plan["n_weeks"] == plan["n_subjects"]
    assert all(len(w["subjects"]) == 1 for w in plan["weeks"])


def test_placements_are_not_planned_as_study_weeks():
    plan = cf.weekly_plan("btech_cse", 12)
    assert not any(cf._NOT_A_TOPIC.search(n) for n in flat(plan))


def test_every_subject_carries_college_study_links():
    plan = cf.weekly_plan("mba", 8)
    for week in plan["weeks"]:
        for subject in week["subjects"]:
            assert set(subject["links"]) == {"nptel", "swayam", "youtube"}


def test_bad_input_is_refused():
    with pytest.raises(KeyError):
        cf.weekly_plan("not-a-course", 12)
    with pytest.raises(ValueError):
        cf.weekly_plan("btech_cse", 0)


def test_the_plan_says_it_is_not_the_official_timetable():
    plan = cf.weekly_plan("bcom", 12)
    assert "not the official course timetable" in plan["note"]
    assert plan["options"] and all(w <= plan["n_subjects"] for w in plan["options"])


@pytest.fixture(scope="module")
def main(tmp_path_factory):
    import api.main as main
    from api.service import Service
    from elpr.db.app_store import AppStore

    main.service = Service()
    main.service.store = AppStore(tmp_path_factory.mktemp("api") / "app.db")
    return main


def test_the_api_serves_the_plan_to_anyone(main):
    from fastapi import HTTPException

    plan = main.course_finder_plan("btech_cse", 16)
    assert plan["n_weeks"] == 16 and plan["name"].startswith("B.Tech")
    with pytest.raises(HTTPException) as bad:
        main.course_finder_plan("nope", 16)
    assert bad.value.status_code == 404
