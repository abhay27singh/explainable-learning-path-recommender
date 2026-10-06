"""Entrance exams: who may sit them, what the syllabus outline holds, and the plan.

The plan has to behave exactly like the course planner, because a student will use both
in the same week and a difference between them would read as a bug. The facts matter
more than the arithmetic: an exam listed for the wrong stream sends someone down two
years of the wrong subjects.
"""
from __future__ import annotations

import pytest

from elpr import exams


def flat(plan):
    return [u["name"] for w in plan["weeks"] for u in w["units"]]


def test_the_three_exams_are_there_with_their_official_source():
    assert set(exams.EXAMS) == {"jee_main", "neet_ug", "cuet_ug"}
    for exam in exams.EXAMS.values():
        assert exam.source.startswith("https://")
        assert exam.body == "National Testing Agency"
        assert exam.leads_to and exam.full_name


def test_an_exam_is_only_offered_to_streams_that_can_sit_it():
    """NEET is written on Biology and JEE on Mathematics. Offering either to the wrong
    stream would send a student after a syllabus they never studied."""
    assert [e.key for e in exams.for_stream("pcm")] == ["jee_main", "cuet_ug"]
    assert [e.key for e in exams.for_stream("pcb")] == ["neet_ug", "cuet_ug"]
    assert [e.key for e in exams.for_stream("pcmb")] == ["jee_main", "neet_ug", "cuet_ug"]
    for stream in ("commerce", "arts"):
        assert [e.key for e in exams.for_stream(stream)] == ["cuet_ug"]


def test_a_student_who_has_not_chosen_a_stream_sees_all_of_them():
    """Class 10. Seeing that NEET needs Biology is part of choosing the stream."""
    assert len(exams.for_stream(None)) == 3


def test_cuet_domain_papers_come_from_the_students_own_stream():
    commerce = dict(exams.sections_for("cuet_ug", "commerce"))
    assert "Accountancy" in commerce["Domain subjects"]
    assert "Physics" not in commerce["Domain subjects"]
    science = dict(exams.sections_for("cuet_ug", "pcm"))
    assert "Physics" in science["Domain subjects"]
    assert "English" not in science["Domain subjects"], "English is the Language section"
    assert "Domain subjects" not in dict(exams.sections_for("cuet_ug", None)), \
        "with no stream there is nothing true to put there"


def test_the_other_exams_ignore_the_stream():
    assert exams.sections_for("jee_main", "pcm") == exams.sections_for("jee_main", "pcmb")


def test_every_unit_appears_once_in_syllabus_order():
    plan = exams.exam_plan("neet_ug", 20)
    expected = [u for _, units in exams.sections_for("neet_ug") for u in units]
    assert flat(plan) == expected
    assert plan["n_units"] == len(expected)
    assert [w["week"] for w in plan["weeks"]] == list(range(1, plan["n_weeks"] + 1))


def test_the_load_is_spread_as_evenly_as_the_list_allows():
    for weeks in (4, 8, 12, 24, 36):
        plan = exams.exam_plan("jee_main", weeks)
        sizes = [len(w["units"]) for w in plan["weeks"]]
        assert min(sizes) >= 1, "no empty weeks"
        assert max(sizes) - min(sizes) <= 1, f"uneven load at {weeks} weeks: {sizes}"
        assert sum(sizes) == plan["n_units"]


def test_the_length_on_show_is_always_one_of_the_choices():
    for key in ("jee_main", "neet_ug", "cuet_ug"):
        for asked in (4, 24, 52, 500):
            plan = exams.exam_plan(key, asked, stream="pcmb")
            assert plan["n_weeks"] in plan["options"], (key, asked, plan["options"])
            assert all(w <= plan["n_units"] for w in plan["options"])


def test_dates_run_monday_to_sunday_from_the_monday_of_the_week_given():
    plan = exams.exam_plan("jee_main", 6, start="2026-03-04")   # a Wednesday
    first, last = plan["weeks"][0], plan["weeks"][-1]
    assert first["starts"] == "2026-03-02" and first["ends"] == "2026-03-08"
    assert plan["starts"] == first["starts"] and plan["ends"] == last["ends"]


def test_units_carry_school_study_links_not_university_ones():
    """These exams are written on the class 11 and 12 syllabus. NPTEL and SWAYAM carry
    nothing for it, which a student would find out only after clicking."""
    plan = exams.exam_plan("neet_ug", 8)
    for week in plan["weeks"]:
        for unit in week["units"]:
            assert set(unit["links"]) == {"ncert", "khan", "youtube"}


def test_the_plan_says_what_it_is_not():
    plan = exams.exam_plan("cuet_ug", 8, stream="arts")
    assert "not the official syllabus" in plan["note"]
    assert "check the official syllabus" in plan["note"].lower()
    assert plan["source"].startswith("https://")


def test_bad_input_is_refused():
    with pytest.raises(KeyError):
        exams.exam_plan("not-an-exam", 12)
    with pytest.raises(KeyError):
        exams.sections_for("not-an-exam")
    with pytest.raises(ValueError):
        exams.exam_plan("jee_main", 0)


def test_the_summary_counts_match_the_syllabus_behind_it():
    out = exams.summary("commerce")
    assert [e["key"] for e in out["exams"]] == ["cuet_ug"]
    cuet = out["exams"][0]
    assert cuet["n_units"] == sum(s["n_units"] for s in cuet["sections"])
    assert cuet["n_units"] == exams.exam_plan("cuet_ug", 4, stream="commerce")["n_units"]


def test_no_exam_date_cut_off_or_rank_is_promised_anywhere():
    """These change every year. A wrong one here would be worse than saying nothing."""
    words = ("cut-off", "cutoff", "rank", "percentile", "guarantee", "will clear")
    for exam in exams.EXAMS.values():
        text = " ".join([exam.leads_to, exam.note, exam.full_name]).lower()
        for word in words:
            assert word not in text, f"{exam.key} promises {word}"


@pytest.fixture(scope="module")
def main(tmp_path_factory):
    import api.main as main
    from api.service import Service
    from elpr.db.app_store import AppStore

    main.service = Service()
    main.service.store = AppStore(tmp_path_factory.mktemp("api") / "app.db")
    return main


def test_the_api_serves_exams_to_anyone(main):
    """A visitor choosing a stream needs these before they have an account."""
    out = main.exam_list("pcb")
    assert [e["key"] for e in out["exams"]] == ["neet_ug", "cuet_ug"]
    detail = main.exam_detail("neet_ug")
    assert detail["source"].startswith("https://")
    assert detail["sections"][0]["units"][0]["links"]["ncert"]


def test_the_api_refuses_an_exam_that_does_not_exist(main):
    from fastapi import HTTPException

    for call in (lambda: main.exam_detail("nope"), lambda: main.exam_plan("nope", 12)):
        with pytest.raises(HTTPException) as bad:
            call()
        assert bad.value.status_code == 404


def test_the_calendar_file_has_one_event_per_week(main):
    body = main.exam_plan_ics("jee_main", 6, "2026-03-02").body.decode()
    assert body.count("BEGIN:VEVENT") == 6
    assert "JEE Main revision" in body
    assert body.startswith("BEGIN:VCALENDAR\r\n")


def test_each_exam_links_to_its_own_official_nta_site():
    """Regression: CUET UG pointed at exams.nta.ac.in/CUET-UG/, which NTA retired and
    which now returns 404. Each exam has its own site on nta.nic.in."""
    sites = {key: exams.EXAMS[key].source for key in exams.EXAMS}
    assert sites == {"jee_main": "https://jeemain.nta.nic.in/", "neet_ug": "https://neet.nta.nic.in/",
                     "cuet_ug": "https://cuet.nta.nic.in/"}
