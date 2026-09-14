"""Streaks, activity calendar and badges for registered students.

Pure functions over the timestamps of recorded study events, kept apart from the web layer
so they can be tested with fixed dates. Nothing here is estimated: every count comes from
events the student recorded.
"""
from __future__ import annotations

from collections import Counter
from datetime import date, timedelta


def _day_counts(stamps) -> Counter:
    return Counter(date.fromtimestamp(float(s)) for s in stamps)


def streaks(stamps, today: date) -> dict:
    """Current and longest run of consecutive days with at least one activity.

    A streak stays alive through today if the student was active yesterday, so it does
    not reset at midnight before they have had a chance to study."""
    days = set(_day_counts(stamps))
    if not days:
        return {"current": 0, "longest": 0, "active_today": False, "last_active": None}
    one = timedelta(days=1)
    day = today if today in days else today - one
    current = 0
    while day in days:
        current += 1
        day -= one
    longest = run = 0
    previous = None
    for day in sorted(days):
        run = run + 1 if previous is not None and day - previous == one else 1
        longest = max(longest, run)
        previous = day
    return {"current": current, "longest": longest, "active_today": today in days,
            "last_active": max(days).isoformat()}


def calendar_days(stamps, today: date, weeks: int = 12) -> list[dict]:
    """Activity count per day for whole weeks ending today, starting on a Monday."""
    counts = _day_counts(stamps)
    start = today - timedelta(days=today.weekday()) - timedelta(weeks=weeks - 1)
    out, day = [], start
    while day <= today:
        out.append({"date": day.isoformat(), "weekday": day.weekday(),
                    "count": counts.get(day, 0)})
        day += timedelta(days=1)
    return out


def level_of(mastery: float, mastered: float, ready: float) -> int:
    """0 needs work, 1 getting there, 2 strong. Mirrors the interface's LEVEL()."""
    return 2 if mastery >= mastered else 1 if mastery >= ready else 0


BADGES = (
    ("First step", "Record your first activity", lambda n, longest, passed: n >= 1),
    ("Three in a row", "Study on 3 days in a row", lambda n, longest, passed: longest >= 3),
    ("Full week", "Study on 7 days in a row", lambda n, longest, passed: longest >= 7),
    ("Ten activities", "Record 10 activities", lambda n, longest, passed: n >= 10),
    ("First pass", "Pass your first test", lambda n, longest, passed: passed >= 1),
)


def badges(n_events: int, longest: int, n_passed: int) -> list[dict]:
    return [{"name": name, "rule": rule, "earned": bool(test(n_events, longest, n_passed))}
            for name, rule, test in BADGES]
