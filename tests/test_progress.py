"""Streaks, calendar, badges and the weekly summary.

Streak rules are easy to get subtly wrong at day boundaries, so they are tested with
fixed dates. The weekly summary is tested end to end against recorded events, including
one backdated outside the seven-day window.
"""
from __future__ import annotations

from datetime import date, datetime

from elpr.progress import badges, calendar_days, level_of, streaks


def ts(y, m, d, h=12):
    return datetime(y, m, d, h).timestamp()


TODAY = date(2026, 9, 14)


def test_no_activity_means_no_streak():
    assert streaks([], TODAY) == {"current": 0, "longest": 0, "active_today": False,
                                  "last_active": None}


def test_current_streak_counts_back_from_today_and_ignores_repeat_days():
    s = streaks([ts(2026, 9, 12), ts(2026, 9, 13), ts(2026, 9, 14), ts(2026, 9, 14, 18)], TODAY)
    assert s["current"] == 3 and s["longest"] == 3 and s["active_today"]


def test_studying_yesterday_keeps_the_streak_alive_today():
    s = streaks([ts(2026, 9, 12), ts(2026, 9, 13)], TODAY)
    assert s["current"] == 2 and not s["active_today"]


def test_a_gap_breaks_the_current_streak_but_longest_is_kept():
    s = streaks([ts(2026, 9, d) for d in (1, 2, 3, 4)] + [ts(2026, 9, 10)], TODAY)
    assert s["current"] == 0 and s["longest"] == 4 and s["last_active"] == "2026-09-10"


def test_calendar_covers_whole_weeks_ending_today():
    cal = calendar_days([ts(2026, 9, 14), ts(2026, 9, 14, 9)], TODAY, weeks=12)
    assert cal[0]["weekday"] == 0
    assert cal[-1] == {"date": "2026-09-14", "weekday": TODAY.weekday(), "count": 2}
    assert len(cal) == 11 * 7 + TODAY.weekday() + 1


def test_level_of_matches_the_interface():
    assert [level_of(v, 0.9, 0.3) for v in (0.95, 0.9, 0.5, 0.3, 0.1)] == [2, 2, 1, 1, 0]


def test_badges_follow_real_counts():
    got = {b["name"]: b["earned"] for b in badges(n_events=10, longest=3, n_passed=1)}
    assert got == {"First step": True, "Three in a row": True, "Full week": False,
                   "Ten activities": True, "First pass": True}
    assert not any(b["earned"] for b in badges(0, 0, 0))


def test_weekly_summary_counts_only_the_last_seven_days(tmp_path):
    import sqlite3
    import time

    from api.service import Service
    from elpr.db.app_store import AppStore

    service = Service()
    service.store = AppStore(tmp_path / "app.db")
    user = service.store.register("learner", "pass", "L", "student", module="DDD")
    weeks = [n["concept"] for n in service.module_graph("DDD")["nodes"]]
    for concept in weeks[:3]:
        service.store.add_event(user.id, concept, "study")
    service.store.add_event(user.id, weeks[3], "assessment", True)

    con = sqlite3.connect(tmp_path / "app.db")
    con.execute("UPDATE study_events SET created_at = ? WHERE id = 1", (time.time() - 10 * 86400,))
    con.commit()
    con.close()

    p = service.progress_for(user)
    week = p["week"]
    assert week["activities"] == 3 and week["passed"] == 1 and week["hard"] == 0
    assert len(week["weeks_studied"]) == 3
    assert p["streak"]["current"] == 1 and p["streak"]["active_today"]
    assert p["streak"]["longest"] >= 1
    assert week["next_step"]
    text = " ".join(week["summary"])
    assert "3 activities" in text and "—" not in text
    assert {b["name"] for b in p["badges"] if b["earned"]} >= {"First step", "First pass"}
    # the backdated event is outside the 7-day summary but still inside the 12-week calendar
    assert sum(d["count"] for d in p["calendar"]) == 4
    # one event predates the window, so this is not a first week and a comparison is made
    assert week["first_week"] is False
    assert "first week" not in text


def test_first_week_makes_no_claims_about_change(tmp_path):
    """Regression: a new student was told "24 weeks moved down a level" after studying,
    because they were compared with the model's generic starting estimate and judged on
    relative levels. A first week has nothing real to compare with."""
    from api.service import Service
    from elpr.db.app_store import AppStore

    service = Service()
    service.store = AppStore(tmp_path / "app.db")
    user = service.store.register("newbie", "pass", "N", "student", module="DDD")
    weeks = [n["concept"] for n in service.module_graph("DDD")["nodes"]]
    service.store.add_event(user.id, weeks[0], "study")
    service.store.add_event(user.id, weeks[1], "assessment", True)
    service.store.add_event(user.id, weeks[2], "assessment", False)

    week = service.progress_for(user)["week"]
    assert week["first_week"] is True
    assert week["moved_up"] == [] and week["moved_down"] == []
    text = " ".join(week["summary"])
    assert "first week of activity" in text
    assert "moved" not in text and "fell" not in text and "rose" not in text


def test_new_student_gets_an_honest_empty_summary(tmp_path):
    from api.service import Service
    from elpr.db.app_store import AppStore

    service = Service()
    service.store = AppStore(tmp_path / "app.db")
    user = service.store.register("fresh", "pass", "F", "student", module="DDD")
    p = service.progress_for(user)
    assert p["streak"]["current"] == 0
    assert p["week"]["activities"] == 0
    assert p["week"]["summary"][0] == "No activity recorded in the last 7 days."
    assert not any(b["earned"] for b in p["badges"])
