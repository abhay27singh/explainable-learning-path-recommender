"""The skill path, added to match NEP 2020.

The policy removes the hard separation between academic and vocational study, so skill
courses are reachable from every rung rather than being a consolation prize after
class 10. Degrees also carry the policy's exit points.
"""
from __future__ import annotations

import pytest

from elpr import course_finder as cf


def skills():
    return [c for c in cf.COURSES if c.level == "skill"]


def test_skill_courses_are_open_from_every_rung_of_the_ladder():
    for stage in cf.STAGES:
        assert "skill" in cf.NEXT_LEVELS[stage], f"{stage} should reach skill courses"
    assert [l["value"] for l in cf.next_after("pg")["levels"]] == ["skill"]


def test_a_skill_course_needs_no_stream_or_degree():
    for course in skills():
        assert course.eligible(set(), None), course.key
        assert not course.needs_all and not course.needs_any and course.degrees is None
    for stream in cf.STREAMS:
        out = cf.recommend("skill", ["coding", "machines"], stream=stream)
        assert out["courses"] and not out["also_consider"]
    assert cf.recommend("skill", ["health"], degree="bcom")["courses"]


def test_skill_subjects_link_to_skill_india_not_to_universities():
    links = cf.study_links("Arc Welding", track="skill")
    assert set(links) == {"skillindia", "nsdc", "youtube"}
    assert "site%3Askillindia.gov.in" in links["skillindia"]
    assert "nptel" not in str(links) and "ncert" not in str(links)
    for course in skills():
        for year in course.to_dict()["years"]:
            for subject in year["subjects"]:
                assert set(subject["links"]) == {"skillindia", "nsdc", "youtube"}


def test_each_skill_course_says_what_it_is_and_who_awards_it():
    for course in skills():
        d = course.to_dict()
        assert d["track"] == "skill" and d["track_label"] == "Skill path"
        assert "NSQF" in d["framework"]
        assert d["duration"] and d["eligibility"] and d["category"]
        assert d["exits"] == [], "exit points belong to degrees, not skill courses"


def test_degrees_carry_the_nep_exit_points():
    degree = cf.course("btech_cse")
    assert degree["exits"] == list(cf.NEP_UG_EXITS)
    assert "certificate" in degree["exits"][0] and "research" in degree["exits"][-1]
    for other in ("dip_cs", "mba", "iti_fitter"):
        assert cf.course(other)["exits"] == []


def test_the_skill_path_is_labelled_as_nsqf_and_not_a_replacement_for_school():
    assert "NSQF" in cf.SKILL_NOTE
    assert "alongside" in cf.SKILL_NOTE
    assert set(cf.TRACKS) == {"academic", "skill"}
    assert {c.track for c in cf.COURSES} == {"academic", "skill"}


def test_skill_courses_cover_trades_digital_care_and_business():
    categories = {c.category for c in skills()}
    assert len(skills()) >= 12
    for expected in ("Trades and technology", "Digital and office skills",
                     "Healthcare skills", "Business and service skills"):
        assert expected in categories
    assert any("ITI" in c.name for c in skills())


def test_the_finder_can_plan_a_skill_course_by_week():
    plan = cf.weekly_plan("skill_web_dev", 6)
    assert plan["n_weeks"] == 6 and plan["level_label"] == "Skill course"
    assert all(w["subjects"] for w in plan["weeks"])


@pytest.fixture(scope="module")
def main(tmp_path_factory):
    import api.main as main
    from api.service import Service
    from elpr.db.app_store import AppStore

    main.service = Service()
    main.service.store = AppStore(tmp_path_factory.mktemp("api") / "app.db")
    return main


def test_a_degree_student_can_still_take_a_skill_course(main):
    from fastapi import Response

    store = main.service.store
    main.register(main.Registration(username="ugskill", password="passw0rd", role="student",
                                    stage="ug", module="CCC"), Response())
    token = store.create_session(store.authenticate("ugskill", "passw0rd").id)
    out = main.course_finder_suggest(
        main.FinderBody(level="skill", interests=["coding"]), token)
    assert out["courses"], "NEP 2020 allows a degree student to take a skill course"
    assert all(c["track"] == "skill" for c in out["courses"])
