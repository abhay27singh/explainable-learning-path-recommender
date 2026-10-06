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


def test_the_invite_field_is_disabled_while_hidden():
    """It is required for an adviser. Visible or enabled for a student, it would block
    the student's sign-up without showing why, the way adviser sign-up once broke."""
    wrap = re.search(r'<div id="invite-wrap"[^>]*>.*?</div>', PAGE, re.S).group(0)
    assert 'style="display:none"' in wrap and "disabled" in wrap
    sync = functions()["syncSignup"]
    assert 'show("#invite-wrap", !student)' in sync and 'show("#class-wrap", student)' in sync


def test_both_kinds_of_student_can_see_and_change_who_sees_their_record():
    fns = functions()
    for name in ("renderSchoolStudent", "renderStudent"):
        assert "advisersBlock()" in fns[name] and "bindAdvisers()" in fns[name], name


def test_an_adviser_is_shown_their_class_code_and_an_admin_can_invite():
    fns = functions()
    assert "bindClassCode()" in fns["renderAdviser"]
    assert '"/api/admin/invites"' in fns["renderAdmin"] and "inv-revoke" in fns["renderAdmin"]


def test_a_school_students_page_uses_the_full_width_of_a_phone():
    """Regression: ".workspace.two" outranks ".workspace", so the one-column phone rule
    missed the school student's page, which kept a 236 pixel column on a 375 pixel phone."""
    css = re.search(r"<style>(.*?)</style>", PAGE, re.S).group(1)
    phone = media_block_holding(css, ".railside .quizbox{display:none}")
    assert ".workspace, .workspace.two{grid-template-columns:minmax(0,1fr)}" in phone


def test_footer_links_stay_one_per_line_on_a_touch_screen():
    """Regression: making them taller also made them inline, and they ran together as
    "Course FinderExplore coursesMy learning"."""
    css = re.search(r"<style>(.*?)</style>", PAGE, re.S).group(1)
    touch = media_block_holding(css, "(pointer: coarse)")
    rule = re.search(r"footer\.site \.cols a\{([^}]*)\}", touch).group(1)
    assert "inline" not in rule


def test_my_learning_says_where_the_student_is_once():
    """The page said "2 of 35 weeks done" four times: in the heading, a tile, the left
    rail and the band above the path. It is now one sentence, in the heading."""
    fns = functions()
    page = fns["renderStudent"]
    assert "courseLine(path, step, state.module)" in page
    assert 'class="tiles"' not in page and "cb-prog" not in page
    assert "coursePanel" not in PAGE, "the rail is for finding your way, not for numbers"
    assert "is next." in fns["courseLine"]


def test_the_right_column_stays_in_view_without_covering_the_strip():
    """Fixed like the left column on a wide screen. Its contents stick inside a column as
    tall as the rows beside the path, so they stop above the full-width strip below it;
    sticking the column itself made it slide over that strip."""
    css = re.search(r"<style>(.*?)</style>", PAGE, re.S).group(1)
    wide = media_block_holding(css, ".railside.stick{align-self:stretch}")
    assert ".railside.stick > .railside-in{position:sticky" in wide
    assert '<aside class="railside stick"><div class="railside-in">' in functions()["renderStudent"]


def test_progress_says_each_thing_once_in_sentences():
    """The heading said "Computer Science · 7 of 35 weeks done", the course card said it
    again as numbers, the activity card was a row of label and number pairs, and the
    streak appeared twice. Each is now one sentence, said once."""
    fns = functions()
    page = fns["renderProgress"]
    assert "You have finished ${path.n_done} of the ${path.n_weeks} weeks" in page
    assert "milestoneBlock(progress, {streak: false})" in page, "the activity card says the streak"
    assert "act-stats" not in PAGE and "cf-figs" not in PAGE and "railcourse" not in PAGE
    assert "-day streak" in fns["activityCard"]
    assert 'class="tiles"' not in fns["renderSchoolProgress"]


def test_progress_keeps_its_right_column_in_view_too():
    """Recent activity moved beside the calendar, which leaves the right column short
    enough to stay in view on a laptop screen."""
    page = functions()["renderProgress"]
    assert '<aside class="railside stick"><div class="railside-in">' in page
    side = page.split('<aside class="railside stick">')[1].split("</aside>")[0]
    assert "${recent}" not in side


def test_visitors_can_see_a_real_path_before_signing_up():
    fns = functions()
    assert '"try"' in re.search(r"const PUBLIC = new Set\(\[(.*?)\]\)", SCRIPT).group(1)
    assert 'id="view-try"' in PAGE and 'data-go="try"' in PAGE
    assert '"/api/demo"' in fns["renderTry"] and '"/api/demo"' in fns["fillShowcase"]
    # The demo is the model's own ranking, not course order, and the page must say so.
    assert "The model's next" in fns["pathList"]


def test_long_pages_get_a_section_bar_and_progress_gets_a_replay():
    fns = functions()
    assert '["privacy", "terms", "policies"].includes(name)' in fns["show"]
    assert "aria-current" in fns["markSection"]
    assert "startReplay(" in fns["renderProgress"] and '"/api/me/replay"' in fns["startReplay"]


def test_the_week_map_shows_the_level_not_the_raw_figure():
    """Regression: "Model: 1% chance of doing well" on a week the student had just
    studied. For a student's own record the raw figures sit near zero."""
    body = functions()["weekMap"]
    assert "chance of doing well" not in body and "LEVEL(c.mastery, t)" in body


def test_long_pages_use_a_wide_screen_without_overlong_lines():
    """The text stopped at about 70 characters a line, as it should, but sat on the left
    of a wide screen with nothing beside it. On a wide screen the section list now stands
    beside it and the two are centred; the line length stays readable."""
    css = re.search(r"<style>(.*?)</style>", PAGE, re.S).group(1)
    wide = media_block_holding(css, ".page.legal.has-bar{display:grid")
    assert "justify-content:center" in wide and "minmax(0,720px)" in wide
    assert 'page.classList.add("has-bar")' in functions()["sectionBar"]
    # Regression: the marker waited for an animation frame, which a hidden tab never
    # draws, and stopped following after one scroll in the background.
    assert 'addEventListener("scroll", markSection' in SCRIPT


def test_a_course_a_visitor_saves_is_kept_when_they_join():
    """Regression: "Save this course" sent a visitor to sign up and then forgot the
    course, at the moment they chose to join. Both sign-up and sign-in now keep it."""
    fns = functions()
    assert "save-later" in fns["bindSave"] and "PENDING_SAVE" in fns["bindSave"]
    login = re.search(r'\$\("#form-login"\)\.onsubmit = async e => \{(.*?)\n\};', SCRIPT, re.S).group(1)
    register = re.search(r'\$\("#form-register"\)\.onsubmit = async e => \{(.*?)\n\};', SCRIPT, re.S).group(1)
    for name, body in (("sign in", login), ("sign up", register)):
        assert "keepPendingSave()" in body, name
        assert body.index("keepPendingSave()") < body.index("nav("), f"{name}: save before the page loads"
