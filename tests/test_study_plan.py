"""The ladder: class 10, then class 12 or a diploma, then a degree, then postgraduate.

At each school rung a student first sees the subjects they are studying now, with free
study links, and then what they can study next. A class 12 stream decides both: PCM
leads to engineering, PCB to medicine and life sciences.
"""
from __future__ import annotations

import pytest

from elpr import course_finder as cf


def names(subjects):
    return [s["name"] for s in subjects]


def test_class_10_subjects_come_with_study_links():
    now = cf.subjects_now("class_10")
    assert names(now["subjects"]) == ["English", "Mathematics", "Science", "Social Science"]
    assert "Computer Applications" in names(now["optional"])
    for subject in now["subjects"]:
        links = subject["links"]
        assert set(links) == {"ncert", "khan", "youtube"}
        assert "site%3Ancert.nic.in" in links["ncert"] and "class+10" in links["youtube"]


def test_class_12_subjects_follow_the_stream():
    pcm = cf.subjects_now("class_12", "pcm")
    assert names(pcm["subjects"]) == ["English", "Physics", "Chemistry", "Mathematics"]
    assert pcm["label"] == "Class 12, " + cf.STREAMS["pcm"]
    assert names(cf.subjects_now("class_12", "pcb")["subjects"])[-1] == "Biology"
    with pytest.raises(ValueError):
        cf.subjects_now("class_12")                      # stream is needed
    with pytest.raises(ValueError):
        cf.subjects_now("not-a-stage")


def test_a_student_on_a_course_studies_that_course_not_school_subjects():
    for stage in cf.COURSE_STAGES:
        now = cf.subjects_now(stage)
        assert now["subjects"] == [] and now["label"] == cf.STAGES[stage]


def test_after_class_10_a_student_picks_a_stream_or_a_diploma():
    nxt = cf.next_after("class_10")
    assert [l["value"] for l in nxt["levels"]] == ["diploma_10", "skill"]
    assert [s["value"] for s in nxt["streams"]] == list(cf.STREAMS)
    leads = {s["value"]: s["leads_to"] for s in nxt["streams"]}
    assert "Engineering" in leads["pcm"] and "Medicine" in leads["pcb"]
    assert "Physics" in nxt["streams"][0]["core"]


def test_after_class_12_the_stream_decides_what_comes_next():
    nxt = cf.next_after("class_12", "pcb")
    assert [l["value"] for l in nxt["levels"]] == ["diploma_12", "ug", "skill"]
    assert "Medicine" in nxt["leads_to"]
    # NEP 2020 keeps skill courses open at every rung, including after a master's
    assert [l["value"] for l in cf.next_after("pg")["levels"]] == ["skill"]
    assert cf.next_after("pg")["streams"] == []

    pcm = [c["key"] for c in cf.recommend("ug", ["coding", "machines"], stream="pcm")["courses"]]
    pcb = [c["key"] for c in cf.recommend("ug", ["health", "biology"], stream="pcb")["courses"]]
    assert "btech_cse" in pcm and "mbbs" not in pcm
    assert "mbbs" in pcb


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


def test_stream_is_stored_at_signup_and_drives_the_study_plan(main):
    from fastapi import Response

    body = main.Registration(username="pcbstudent", password="passw0rd", role="student",
                             stage="class_12", stream="pcb")
    assert main.register(body, Response())["stream"] == "pcb"
    token = main.service.store.create_session(
        main.service.store.authenticate("pcbstudent", "passw0rd").id)

    plan = main.my_study_plan(token)
    assert names(plan["now"]["subjects"]) == ["English", "Physics", "Chemistry", "Biology"]
    assert "Medicine" in plan["next"]["leads_to"]
    assert [l["value"] for l in plan["next"]["levels"]] == ["diploma_12", "ug", "skill"]

    moved = main.update_studies(main.StudiesBody(stage="ug", module="CCC"), token)
    assert moved["module"] == "CCC" and moved["stream"] is None
    assert main.my_study_plan(token)["now"]["subjects"] == []


def test_a_bad_stream_is_rejected(main):
    from fastapi import Response

    body = main.Registration(username="badstream", password="passw0rd", role="student",
                             stage="class_12", stream="nope")
    assert _status(lambda: main.register(body, Response())) == 400
