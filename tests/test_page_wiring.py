"""Markup the page draws but never wires up looks fine and does nothing.

The page is one file with no framework, so a block of HTML and the function that binds
its buttons are two separate calls, and nothing fails when the second is forgotten.
"""
from __future__ import annotations

import re
from pathlib import Path

PAGE = (Path(__file__).resolve().parents[1] / "web" / "index.html").read_text()
SCRIPT = re.findall(r"<script>(.*?)</script>", PAGE, re.S)[-1]


def functions() -> dict[str, str]:
    """Top-level function name to body, split at each top-level declaration."""
    parts = re.split(r"\n(?=(?:async )?function \w+\()", SCRIPT)
    out = {}
    for part in parts:
        m = re.match(r"(?:async )?function (\w+)\(", part)
        if m:
            out[m.group(1)] = part
    return out


def test_every_page_that_draws_exam_cards_also_binds_them():
    """Regression: a class 10 or class 12 student's My learning page drew the JEE, NEET
    and CUET cards without calling bindExams, so "What the syllabus covers" and the
    revision plan stayed on "Loading" forever."""
    # Helpers such as examsBlock only return markup; the functions that put it on the
    # page are the ones that must bind it.
    drawers = {name: body for name, body in functions().items()
               if "innerHTML" in body
               and any(k in body for k in ("${examsBlock(", "map(examCard)", "courseDetail("))}
    assert drawers, "no function draws exam cards any more; update this test"
    for name, body in drawers.items():
        assert "bindExams(" in body, f"{name} draws exam cards but never binds them"


def test_my_details_has_its_own_address():
    """Regression: My details was a tab on #/dashboard. Once opened, every link to My
    learning went to the same address, so the page kept showing the details until a
    refresh. It now has #/details, and only the router decides which one is showing."""
    fns = functions()
    assert 'name === "details"' in fns["show"]
    assert 'go("details", "My details", "details", "details")' in fns["studentRail"]
    setters = [name for name, body in fns.items() if 'studentTab = "details"' in body]
    assert setters == [], f"only the router may open My details: {setters}"
