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
    # Regression: the page still looked like My learning (its heading, its highlight in
    # the header, the quiz box on the right), so students thought the click had failed.
    assert 'pageHead("My details"' in fns["renderStudent"]
    assert 'details ? "details" : name' in fns["show"], "the header must not highlight My learning"
    assert 'class="railme" data-go="details"' in SCRIPT, "the profile card opens My details too"


def test_an_open_tab_can_tell_the_page_has_changed():
    """Regression: moving between pages never reloads the file, so a tab left open kept
    running its first copy after an update and still showed a bug that had been fixed.
    The page asks /api/version as it moves and reloads when the answer changes."""
    from fastapi import Response

    from api import main

    first = main.page_version(Response())["version"]
    assert first and first == main.page_version(Response())["version"]
    assert "pageChanged()" in SCRIPT and "location.reload()" in SCRIPT


def media_block_holding(css: str, needle: str) -> str:
    """The whole @media block that contains needle, found by matching braces."""
    at = css.index(needle)
    start = css.rindex("@media", 0, at)
    depth, i = 0, css.index("{", start)
    while True:
        depth += {"{": 1, "}": -1}.get(css[i], 0)
        if depth == 0:
            return css[start:i + 1]
        i += 1


def test_the_tab_bar_is_styled_at_every_width_it_shows():
    """Regression: the bar showed up to 860px but its button and icon styles sat in the
    520px block, so between the two (a tablet, a narrow laptop window) every icon drew
    unsized and filled black, as wide as a quarter of the screen."""
    css = re.search(r"<style>(.*?)</style>", PAGE, re.S).group(1)
    shown = media_block_holding(css, ".tabbar:not([hidden]){display:grid")
    for rule in (".tab{", ".tab svg{", ".tab.on{"):
        assert rule in shown, f"{rule} must apply wherever the tab bar shows"


def test_an_adviser_waiting_for_approval_is_told_why():
    """A waiting adviser's list of students is refused by the API. Without its own page
    that refusal would show as a bare error, which reads as a broken site."""
    fns = functions()
    assert "renderAdviserWaiting()" in fns["renderAdviser"]
    assert "/approve`" in fns["renderAdmin"], "the admin page needs the button that approves"
    assert 'id="adviser-note"' in PAGE, "the sign-up page says so before it happens"


def test_every_background_question_has_a_label_a_screen_reader_can_find():
    body = functions()["profileForm"]
    assert 'id="pf-${f.field}"' in body and '<label for="pf-${f.field}">' in body
