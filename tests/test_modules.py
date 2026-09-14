"""Course display names.

The real OULAD course titles are unknown; only each course's subject area is published.
The names shown in the interface are illustrative, so the one thing that must hold is
that no name contradicts the published area. A STEM subject on a Social Sciences course
would be a false statement about the data.
"""
from __future__ import annotations

from elpr.modules import (
    DISPLAY_NAME,
    SUBJECT_AREA,
    codes_matching,
    display_name,
    subject_area,
)

# Kuzilek, Hlosta, Zdrahal (2017), Scientific Data 4:170171, Table 1.
PUBLISHED_AREA = {
    "AAA": "Social Sciences", "BBB": "Social Sciences", "CCC": "STEM", "DDD": "STEM",
    "EEE": "STEM", "FFF": "STEM", "GGG": "Social Sciences",
}

SOCIAL_SCIENCE_SUBJECTS = {
    "Psychology", "Sociology", "Business Studies", "Economics", "Law", "History",
    "Politics", "Education", "Social Work",
}
STEM_SUBJECTS = {
    "Mathematics", "Computer Science", "Engineering", "Physics", "Chemistry", "Biology",
    "Statistics", "Environmental Science",
}


def test_subject_areas_match_the_published_paper():
    assert SUBJECT_AREA == PUBLISHED_AREA


def test_every_course_has_a_distinct_name():
    assert set(DISPLAY_NAME) == set(PUBLISHED_AREA)
    assert len(set(DISPLAY_NAME.values())) == len(DISPLAY_NAME), "names must not repeat"


def test_no_name_contradicts_its_published_subject_area():
    for code, name in DISPLAY_NAME.items():
        allowed = STEM_SUBJECTS if PUBLISHED_AREA[code] == "STEM" else SOCIAL_SCIENCE_SUBJECTS
        assert name in allowed, f"{code} is {PUBLISHED_AREA[code]} but is shown as {name}"


def test_lookups_and_search():
    assert display_name("DDD") == DISPLAY_NAME["DDD"]
    assert display_name("XYZ") == "XYZ" and display_name(None) == ""
    assert subject_area("AAA") == "Social Sciences"
    assert codes_matching("stem") == {"CCC", "DDD", "EEE", "FFF"}
    assert codes_matching("social") == {"AAA", "BBB", "GGG"}
    assert codes_matching(DISPLAY_NAME["DDD"][:4]) >= {"DDD"}
    assert codes_matching("x") == set()
