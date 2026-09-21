"""Dates on a plan, and the calendar file that carries it off this site.

A plan with no dates cannot be late, cannot be reminded about and cannot leave the
page. These tests hold the format to RFC 5545, because a file a calendar refuses to
open is worse than no file at all.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from elpr import course_finder as cf
from elpr.ics import calendar


def test_a_plan_without_a_start_date_has_no_dates():
    plan = cf.weekly_plan("btech_cse", 8)
    assert plan["starts"] is None and plan["ends"] is None
    assert all("starts" not in w for w in plan["weeks"])


def test_weeks_run_monday_to_sunday_from_the_week_the_student_picks():
    # 23 September 2026 is a Wednesday; the plan starts on that Monday, the 21st
    plan = cf.weekly_plan("btech_cse", 4, start="2026-09-23")
    assert [w["starts"] for w in plan["weeks"]] == ["2026-09-21", "2026-09-28",
                                                    "2026-10-05", "2026-10-12"]
    assert plan["weeks"][0]["ends"] == "2026-09-27", "a week ends on the Sunday"
    assert plan["starts"] == "2026-09-21" and plan["ends"] == "2026-10-18"
    for week in plan["weeks"]:
        assert date.fromisoformat(week["starts"]).weekday() == 0


def test_a_bad_start_date_is_refused():
    with pytest.raises(ValueError):
        cf.weekly_plan("btech_cse", 4, start="23-09-2026")
    with pytest.raises(ValueError):
        cf.weekly_plan("btech_cse", 4, start="not a date")


def test_the_calendar_file_is_shaped_the_way_calendars_expect():
    out = calendar("My plan", [{
        "uid": "w1@learning-path", "start": date(2026, 9, 21), "days": 7,
        "summary": "B.Tech · Week 1", "description": "Programming in C, Data Structures",
    }], now=datetime(2026, 9, 21, 8, 0, tzinfo=timezone.utc))

    assert out.startswith("BEGIN:VCALENDAR\r\n") and out.endswith("END:VCALENDAR\r\n")
    assert "\r\n" in out and "\n\n" not in out, "iCalendar lines end with CRLF"
    for required in ("VERSION:2.0", "BEGIN:VEVENT", "UID:w1@learning-path",
                     "DTSTAMP:20260921T080000Z", "DTSTART;VALUE=DATE:20260921",
                     "END:VEVENT"):
        assert required in out, required
    # DTEND is exclusive, so a seven-day week ends on the following Monday
    assert "DTEND;VALUE=DATE:20260928" in out
    # a comma inside a summary is a separator unless it is escaped
    assert "SUMMARY:B.Tech · Week 1" in out
    assert "Programming in C\\, Data Structures" in out


def test_long_lines_are_folded_without_breaking_a_character():
    out = calendar("Plan", [{
        "uid": "long@x", "start": date(2026, 1, 5), "days": 7,
        "summary": "Week 1", "description": "Engineering Mathematics · " * 12,
    }])
    for line in out.split("\r\n"):
        assert len(line.encode("utf-8")) <= 75, line[:40]
    assert "·" in out, "folding must not mangle multi-byte characters"


@pytest.fixture(scope="module")
def main(tmp_path_factory):
    import api.main as main
    from api.service import Service
    from elpr.db.app_store import AppStore

    main.service = Service()
    main.service.store = AppStore(tmp_path_factory.mktemp("api") / "app.db")
    return main


def test_the_course_plan_downloads_as_a_calendar(main):
    response = main.course_finder_plan_ics("dca", weeks=4, start="2026-09-23")
    body = response.body.decode()
    assert response.media_type.startswith("text/calendar")
    assert "attachment" in response.headers["Content-Disposition"]
    assert ".ics" in response.headers["Content-Disposition"]
    assert body.count("BEGIN:VEVENT") == 4
    assert "DTSTART;VALUE=DATE:20260921" in body


def test_a_students_own_path_downloads_as_a_calendar(main):
    from fastapi import HTTPException, Response

    store = main.service.store
    main.register(main.Registration(username="calstudent", password="passw0rd",
                                    role="student", stage="ug", module="CCC"), Response())
    token = store.create_session(store.authenticate("calstudent", "passw0rd").id)

    out = main.my_path_ics(weeks=5, start="2026-09-23", elpr_session=token).body.decode()
    assert out.count("BEGIN:VEVENT") == 5
    assert "DTSTART;VALUE=DATE:20260921" in out
    assert "Computer Science" in out

    main.register(main.Registration(username="calschool", password="passw0rd",
                                    role="student", stage="class_12", stream="pcm"), Response())
    school = store.create_session(store.authenticate("calschool", "passw0rd").id)
    with pytest.raises(HTTPException) as no_course:
        main.my_path_ics(weeks=5, start=None, elpr_session=school)
    assert no_course.value.status_code == 400
