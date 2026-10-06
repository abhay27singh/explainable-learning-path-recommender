"""The home page states a few figures before any data has loaded.

They are written into the page, so nothing updates them when the catalogue grows.
Regression: the page said 43 courses for weeks after the catalogue reached 64.
"""
from __future__ import annotations

import re
from pathlib import Path

from elpr import course_finder as cf

PAGE = (Path(__file__).resolve().parents[1] / "web" / "index.html").read_text()


def test_the_course_count_on_the_home_page_matches_the_catalogue():
    shown = re.search(r'id="fact-courses">(\d+)<', PAGE)
    assert shown, "the home page lost its course count"
    assert int(shown.group(1)) == len(cf.COURSES)


def test_the_home_page_names_every_level_a_student_can_pick():
    assert f"<b>{len(cf.STAGES)} levels</b>" in PAGE
