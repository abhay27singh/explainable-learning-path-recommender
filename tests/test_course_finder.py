"""Course Finder rules.

Rule-based matching, so the tests check the rules hold: eligibility follows the class 12
subjects or bachelor's degree the student actually has, ranking follows interests, every
course is complete and labelled honestly, and study links only point where they should.
"""
from __future__ import annotations

import pytest

from elpr import course_finder as cf


def keys(items):
    return [c["key"] for c in items]


def test_options_describe_levels_streams_subjects_degrees_and_method():
    o = cf.options()
    assert [l["value"] for l in o["levels"]] == list(cf.LEVELS)
    assert {l["input"] for l in o["levels"]} == {"none", "stream", "degree"}
    for s in o["streams"]:
        assert s["core"] and set(s["core"]).isdisjoint(s["optional"]), s["value"]
    assert {d["value"] for d in o["degrees"]} == set(cf.DEGREES)
    assert "not the machine-learning model" in o["method"]


def test_biology_student_needs_mathematics_for_btech():
    pcb_core = cf.recommend("ug", ["coding", "electronics", "machines"], stream="pcb")
    assert not any(k.startswith("btech") for k in keys(pcb_core["courses"]))
    assert "btech_cse" in keys(pcb_core["also_consider"])

    with_maths = cf.recommend("ug", ["coding"], stream="pcb", subjects=["Mathematics"])
    assert "btech_cse" in keys(with_maths["courses"])
    assert "Mathematics" in with_maths["subjects"]


def test_subjects_outside_the_stream_are_ignored():
    out = cf.recommend("ug", ["health"], stream="pcm", subjects=["Biology", "Computer Science"])
    assert "Biology" not in out["subjects"] and "Computer Science" in out["subjects"]
    assert "mbbs" not in keys(out["courses"])
    assert "mbbs" in keys(out["also_consider"])


def test_every_suggestion_is_eligible_for_every_stream_and_subject_choice():
    for stream, subj in cf.STREAM_SUBJECTS.items():
        for extra in ([], subj["optional"]):
            taken = set(subj["core"]) | set(extra)
            for interest in cf.INTERESTS:
                for level in ("diploma_12", "ug"):
                    out = cf.recommend(level, [interest], stream=stream, subjects=extra)
                    for c in out["courses"]:
                        assert cf._BY_KEY[c["key"]].eligible(taken, None), (stream, c["key"])
                    for c in out["also_consider"]:
                        assert not cf._BY_KEY[c["key"]].eligible(taken, None)


def test_diploma_after_class_10_needs_no_stream_and_only_returns_that_level():
    out = cf.recommend("diploma_10", ["coding", "machines", "electronics"])
    assert out["courses"] and all(c["level"] == "diploma_10" for c in out["courses"])
    assert out["stream"] is None and out["subjects"] == []


def test_postgraduate_uses_the_bachelors_degree():
    out = cf.recommend("pg", ["business", "finance", "coding"], degree="bcom")
    got = keys(out["courses"])
    assert "mba" in got and "mcom" in got and "mca" in got
    assert "mtech_cse" in keys(out["also_consider"])
    assert out["degree_label"] == cf.DEGREES["bcom"]


def test_a_course_centred_on_the_interest_outranks_one_that_merely_touches_it():
    """Regression: with one interest picked, every course scored the same and the list
    fell back to alphabetical, putting B.A. Economics above B.Sc Computer Science for a
    student who chose data and statistics."""
    got = keys(cf.recommend("ug", ["data"], stream="pcm")["courses"])
    assert got.index("bsc_maths") < got.index("ba_econ")
    assert got.index("bsc_cs") < got.index("ba_econ")


def test_fit_labels_describe_the_interests_the_student_picked():
    one = cf.recommend("ug", ["coding"], stream="pcm")["courses"]
    assert {c["fit"] for c in one} <= {"Strong match", "Good match"}
    assert next(c["fit"] for c in one if c["key"] == "btech_cse") == "Strong match"
    two = cf.recommend("ug", ["coding", "law"], stream="pcm")["courses"]
    assert next(c["fit"] for c in two if c["key"] == "btech_cse") == "Partial match"


def test_ranking_follows_interests():
    out = cf.recommend("ug", ["coding", "maths", "data"], stream="pcm")
    assert keys(out["courses"])[0] == "btech_cse"
    scores = [c["score"] for c in out["courses"]]
    assert scores == sorted(scores, reverse=True)
    assert keys(cf.recommend("ug", ["finance"], stream="commerce")["courses"])[0] == "bcom"
    assert keys(cf.recommend("ug", ["people"], stream="arts")["courses"])[0] == "ba_psych"


def test_missing_or_bad_input_is_rejected():
    with pytest.raises(ValueError):
        cf.recommend("nope", ["coding"])
    with pytest.raises(ValueError):
        cf.recommend("ug", ["coding"])                      # stream needed
    with pytest.raises(ValueError):
        cf.recommend("pg", ["coding"])                      # degree needed
    with pytest.raises(ValueError):
        cf.recommend("ug", [], stream="pcm")
    with pytest.raises(ValueError):
        cf.recommend("ug", ["not-an-interest"], stream="pcm")


def test_explore_lists_every_course_without_filtering():
    everything = cf.explore()
    assert everything["count"] == len(cf.COURSES)
    listed = [c["key"] for lv in everything["levels"] for cat in lv["categories"]
              for c in cat["courses"]]
    assert sorted(listed) == sorted(c.key for c in cf.COURSES)
    pg = cf.explore("pg")
    assert pg["levels"] and all(l["level"] == "pg" for l in pg["levels"])


def test_college_subjects_link_to_nptel_swayam_and_youtube_and_skip_placements():
    links = cf.study_links("Data Structures")
    assert set(links) == {"nptel", "swayam", "youtube"}
    assert "site%3Anptel.ac.in" in links["nptel"] and "site%3Aswayam.gov.in" in links["swayam"]
    assert "%22Data+Structures%22" in links["nptel"]
    assert links["youtube"].startswith("https://www.youtube.com/results?search_query=")


def test_school_subjects_never_link_to_university_platforms():
    """Regression: class 10 and class 12 subjects linked to NPTEL and SWAYAM, which are
    university platforms and carry nothing for school students."""
    for schooling in ("class 10", "class 12"):
        links = cf.study_links("Science", schooling)
        assert set(links) == {"ncert", "khan", "youtube"}
        assert "nptel" not in str(links) and "swayam" not in str(links)
        assert "site%3Ancert.nic.in" in links["ncert"]
        assert "site%3Akhanacademy.org" in links["khan"]
        assert schooling.replace(" ", "+") in links["youtube"]

    school = cf.subjects_now("class_10")["subjects"] + cf.subjects_now("class_12", "pcm")["subjects"]
    assert school and all(set(s["links"]) == {"ncert", "khan", "youtube"} for s in school)
    college = cf.course("btech_cse")["years"][0]["subjects"]
    assert all(set(s["links"]) == {"nptel", "swayam", "youtube"} for s in college)
    for placement in ("Internship", "Major Project", "Dissertation", "Portfolio",
                      "Moot Court and Internship", "Industrial Training and Project"):
        assert cf.study_links(placement) == {}, placement


def test_catalogue_is_internally_consistent():
    seen = set()
    all_subjects = {s for v in cf.STREAM_SUBJECTS.values() for s in v["core"] + v["optional"]}
    for c in cf.COURSES:
        assert c.key not in seen, f"duplicate {c.key}"
        seen.add(c.key)
        assert c.level in cf.LEVELS and c.category and c.duration and c.framework, c.key
        assert c.interests and set(c.interests) <= set(cf.INTERESTS), c.key
        assert all(1 <= w <= 3 for w in c.interests.values()), c.key
        assert c.years and all(subjects for _, subjects in c.years), c.key
        assert set(c.needs_all) | set(c.needs_any) <= all_subjects, c.key
        if c.level == "pg":
            assert c.degrees is None or c.degrees <= set(cf.DEGREES), c.key
        else:
            assert c.degrees is None, c.key
        if c.level == "diploma_10":
            assert not c.needs_all and not c.needs_any, c.key
    assert {i for c in cf.COURSES for i in c.interests} == set(cf.INTERESTS)
    assert {c.level for c in cf.COURSES} == set(cf.LEVELS)


def test_no_user_facing_text_uses_em_dashes():
    texts = [cf.METHOD, cf.NOTE, cf.SUBJECTS_NOTE, *cf.LEVELS.values(), *cf.STREAMS.values(),
             *cf.DEGREES.values(), *cf.INTERESTS.values()]
    for v in cf.STREAM_SUBJECTS.values():
        texts += v["core"] + v["optional"]
    for c in cf.COURSES:
        texts += [c.name, c.category, c.duration, c.framework, c.eligibility]
        for label, subjects in c.years:
            texts += [label, *subjects]
    offenders = [t for t in texts if "—" in t]
    assert not offenders, offenders
