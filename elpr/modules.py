"""Human-readable names for the seven OULAD modules.

The Open University anonymised its course titles, so the real subjects are NOT known.
The only published attribute is each module's subject area (Social Sciences or STEM),
from Table 1 of

    J. Kuzilek, M. Hlosta, Z. Zdrahal, "Open University Learning Analytics dataset,"
    Scientific Data, vol. 4, 170171, 2017.

For readability the interface shows a familiar subject name for each course. These names
are ILLUSTRATIVE. Each is chosen only to fit its course's published subject area, and
nothing more is claimed. The interface says so where it matters (the signup form and the
Method page). Edit the names here and everything follows; tests/test_modules.py keeps
every name consistent with its published area.

The three-letter codes remain the identifiers everywhere internally (database, trained
model, API field values). Only what a person reads is changed.
"""
from __future__ import annotations

SUBJECT_AREA: dict[str, str] = {
    "AAA": "Social Sciences",
    "BBB": "Social Sciences",
    "CCC": "STEM",
    "DDD": "STEM",
    "EEE": "STEM",
    "FFF": "STEM",
    "GGG": "Social Sciences",
}

ILLUSTRATIVE_NAME: dict[str, str] = {
    "AAA": "Psychology",
    "BBB": "Business Studies",
    "CCC": "Computer Science",
    "DDD": "Mathematics",
    "EEE": "Engineering",
    "FFF": "Physics",
    "GGG": "Sociology",
}

DISPLAY_NAME: dict[str, str] = dict(ILLUSTRATIVE_NAME)


def display_name(code: str | None) -> str:
    """'DDD' -> 'Mathematics'. Unknown or empty codes are returned unchanged."""
    if not code:
        return code or ""
    return DISPLAY_NAME.get(code, code)


def subject_area(code: str | None) -> str:
    """The published subject area, the one attribute that is genuinely known."""
    return SUBJECT_AREA.get(code or "", "")


def module_display_map(codes) -> dict[str, str]:
    return {c: display_name(c) for c in codes}


def codes_matching(query: str) -> set[str]:
    """Codes whose name or subject area contains the query: 'math' and 'stem' both work."""
    q = (query or "").strip().lower()
    if len(q) < 2:
        return set()
    return {
        c for c in DISPLAY_NAME
        if q in DISPLAY_NAME[c].lower() or q in SUBJECT_AREA[c].lower()
    }
