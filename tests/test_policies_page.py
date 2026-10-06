"""The Education policies page restates, in plain words, rules the code already applies.

The page is hand-written HTML and the rules live in elpr/course_finder.py and
elpr/exams.py, so the two can drift apart without anything failing: a syllabus revised in
exams.py would leave the page quoting last year's unit counts. These tests read the
numbers off the page and check each one against the data it describes.
"""
from __future__ import annotations

import re
from pathlib import Path

from elpr import course_finder as cf
from elpr.exams import CUET_UG, JEE_MAIN, NEET_UG

PAGE = (Path(__file__).resolve().parents[1] / "web" / "index.html").read_text()
SCRIPT = re.findall(r"<script>(.*?)</script>", PAGE, re.S)[-1]
VIEW = re.search(r'<div class="view" id="view-policies">(.*?)\n</div>\n', PAGE, re.S).group(1)
TEXT = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", VIEW))


def units(exam) -> dict[str, int]:
    return {name: len(topics) for name, topics in exam.sections}


def test_a_visitor_can_open_the_page_and_find_it():
    """Students read this before signing up, so it must not sit behind the sign-in."""
    assert '"policies"' in re.search(r"const PUBLIC = new Set\(\[(.*?)\]\)", SCRIPT).group(1)
    assert 'policies: "Education policies"' in SCRIPT
    footer = re.search(r'<footer class="site">(.*?)</footer>', PAGE, re.S).group(1)
    assert 'data-go="policies"' in footer
    privacy = re.search(r'<div class="view" id="view-privacy">(.*?)\n</div>\n', PAGE, re.S).group(1)
    assert 'data-go="policies"' in privacy


def test_exam_unit_counts_match_the_syllabi_the_plans_use():
    neet, jee = units(NEET_UG), units(JEE_MAIN)
    assert (f"{neet['Physics']} Physics units, {neet['Chemistry']} Chemistry units and "
            f"{neet['Biology']} Biology units") in TEXT
    assert (f"{jee['Physics']} Physics units, {jee['Chemistry']} Chemistry units and "
            f"{jee['Mathematics']} Mathematics units") in TEXT
    assert "General Aptitude Test" in units(CUET_UG) and "General Aptitude Test" in TEXT


def test_degree_exits_match_the_ugc_credits_the_course_pages_show():
    credits = [int(c) for c in re.findall(r"\((\d+) credits\)", " ".join(cf.NEP_UG_EXITS))]
    assert credits == [40, 80, 120, 160]
    for c in credits:
        assert f"({c} credits)" in TEXT
    assert "75 percent or more in the first six semesters" in TEXT
    assert "75 percent or more in the first six semesters" in cf.NEP_UG_EXITS[-1]
    assert "Ten courses" in TEXT and len(cf.NEP_EXIT_DEGREES) == 10
    for clause in ("within three years", "finish within seven"):
        assert clause in cf.NEP_EXIT_NOTE and clause in TEXT


def test_master_lengths_match_the_courses():
    assert "one year after a four-year honours degree" in TEXT
    assert cf.UGC_PG == "2 years, or 1 year after a four-year honours degree"
    by_key = {c.key: c for c in cf.COURSES}
    for key in ("mtech_cse", "mca", "mba"):
        assert by_key[key].duration == "2 years", key
    assert by_key["llb"].duration == "3 years"
    assert "itep" in by_key and "beled" not in by_key


def test_skill_course_counts_match_the_catalogue():
    skill = [c for c in cf.COURSES if c.level == "skill"]
    iti = sum(c.framework == cf.ITI for c in skill)
    sector = sum(c.framework == cf.NSDC for c in skill)
    assert (len(skill), iti, sector) == (16, 4, 12)
    assert "The 16 skill courses" in TEXT and "Four are ITI trades" in TEXT
    assert "Twelve are short job-role courses" in TEXT


def test_every_rule_names_an_official_source():
    """Each section points at the body that publishes the rule, never at a news site."""
    links = re.findall(r'href="(https://[^"]+)"', VIEW)
    assert len(links) >= 12
    hosts = {re.match(r"https://([^/]+)", u).group(1) for u in links}
    assert all(h.endswith((".gov.in", ".nic.in", ".org.in")) for h in hosts), hosts


def test_the_page_states_no_date_cut_off_or_rank_and_uses_no_em_dash():
    assert "\u2014" not in VIEW, "no em dashes in user-facing text"
    assert not re.search(r"\b(cut-off|rank)\s+(is|was|of)\b", TEXT, re.I)
