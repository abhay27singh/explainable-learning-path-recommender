"""Self-check questions on entrance exam units, graded on the server.

They replace a student's word with a real answer, so the bank has to be trustworthy:
one right answer per question, three distinct wrong ones, a reason for each, and every
unit on the JEE Main and NEET UG syllabus covered. The right answer must not sit in the
same place every time, or the check measures pattern spotting. And the results stay out
of the study events, because an exam unit is not a week of the course the model knows.
"""
from __future__ import annotations

from collections import Counter

import pytest

from elpr import exams
from elpr import selfcheck as sc


def answer_key(exam, section, unit):
    """The right option for each question, as shown on the page."""
    return {qid: sc._order(qid).index(0) for qid, _ in sc._questions(section, unit)}


def test_every_jee_and_neet_unit_has_a_check():
    for key in ("jee_main", "neet_ug"):
        for section, units in exams.sections_for(key):
            for unit in units:
                assert sc.has_check(section, unit), f"{key}: {section} / {unit} has none"


def test_every_topic_is_used_and_every_mapped_topic_exists():
    used = {t for topics in sc.UNIT_TOPICS.values() for t in topics}
    assert used == set(sc.BANK)


def test_every_mapped_unit_is_a_real_syllabus_unit():
    """A mapping for a unit name that no exam lists would be a check nobody can reach,
    usually a typo that leaves the real unit without one."""
    real = {(section, unit) for key in exams.EXAMS
            for section, units in exams.sections_for(key, "pcmb") for unit in units}
    assert set(sc.UNIT_TOPICS) <= real


def test_each_question_has_one_right_answer_three_distinct_wrong_ones_and_a_reason():
    for topic, questions in sc.BANK.items():
        assert len(questions) == 3, topic
        for q in questions:
            options = [q.right, *q.wrong]
            assert len(q.wrong) == 3, (topic, q.prompt)
            assert len(set(options)) == 4, f"repeated option in {topic}: {q.prompt}"
            assert q.prompt.strip() and q.why.strip(), (topic, q.prompt)


def test_no_em_dash_anywhere_in_the_bank():
    """House rule for anything a student reads."""
    for topic, questions in sc.BANK.items():
        for q in questions:
            for text in (q.prompt, q.right, *q.wrong, q.why):
                assert "—" not in text, (topic, text)


def test_the_right_answer_is_spread_across_all_four_places():
    slots = Counter(sc._order(f"{t}:{i}").index(0)
                    for t, qs in sc.BANK.items() for i, _ in enumerate(qs))
    total = sum(slots.values())
    assert set(slots) == {0, 1, 2, 3}
    assert min(slots.values()) / total > 0.15, slots


def test_the_order_is_the_same_every_time():
    """A reload, or a restart of the server, must not move the right answer under a
    student halfway through."""
    first = sc.questions_for("jee_main", "Physics", "Optics")
    again = sc.questions_for("jee_main", "Physics", "Optics")
    assert first == again
    assert sorted(sc._order("optics:0")) == [0, 1, 2, 3]


def test_the_questions_go_out_without_their_answers():
    out = sc.questions_for("neet_ug", "Biology: Botany", "Ecology and Environment")
    assert out["n"] == 3
    for q in out["questions"]:
        assert set(q) == {"id", "prompt", "options"}
        assert len(q["options"]) == 4


def test_a_unit_covering_two_topics_gets_both_sets():
    out = sc.questions_for("jee_main", "Chemistry", "Redox Reactions and Electrochemistry")
    assert out["n"] == 6


def test_the_same_unit_name_in_two_sections_is_two_different_checks():
    """NEET has a Thermodynamics unit in Physics and another in Chemistry."""
    physics = sc.questions_for("neet_ug", "Physics", "Thermodynamics")
    chemistry = sc.questions_for("neet_ug", "Chemistry", "Thermodynamics")
    assert physics["questions"][0]["prompt"] != chemistry["questions"][0]["prompt"]


def test_all_right_scores_100_and_all_wrong_scores_0():
    key = answer_key("jee_main", "Physics", "Kinematics")
    right = sc.grade("jee_main", "Physics", "Kinematics", key)
    assert (right["score"], right["n_correct"], right["passed"]) == (100, 3, True)
    wrong = sc.grade("jee_main", "Physics", "Kinematics",
                     {qid: (slot + 1) % 4 for qid, slot in key.items()})
    assert (wrong["score"], wrong["passed"]) == (0, False)
    assert all(r["answer_text"] and r["why"] for r in wrong["results"])


def test_two_of_three_passes_and_one_of_three_does_not():
    key = answer_key("neet_ug", "Chemistry", "Amines")
    ids = list(key)
    two = {**key, ids[0]: (key[ids[0]] + 1) % 4}
    one = {**two, ids[1]: (key[ids[1]] + 1) % 4}
    assert sc.grade("neet_ug", "Chemistry", "Amines", two)["passed"] is True
    assert sc.grade("neet_ug", "Chemistry", "Amines", one)["passed"] is False


def test_every_question_must_be_answered():
    key = answer_key("jee_main", "Mathematics", "Trigonometry")
    key.pop(next(iter(key)))
    with pytest.raises(ValueError):
        sc.grade("jee_main", "Mathematics", "Trigonometry", key)


def test_an_answer_outside_the_four_options_is_refused():
    key = answer_key("jee_main", "Mathematics", "Trigonometry")
    key[next(iter(key))] = 7
    with pytest.raises(ValueError):
        sc.grade("jee_main", "Mathematics", "Trigonometry", key)


def test_a_unit_must_belong_to_the_exam_it_is_checked_under():
    with pytest.raises(KeyError):
        sc.questions_for("jee_main", "Biology: Botany", "Ecology and Environment")
    with pytest.raises(KeyError):
        sc.questions_for("cuet_ug", "General Test", "Numerical Ability")
    with pytest.raises(KeyError):
        sc.questions_for("no_such_exam", "Physics", "Optics")


# ---------------------------------------------------------------------------- API
@pytest.fixture
def main(tmp_path):
    from api import main
    from api.service import Service
    from elpr.db.app_store import AppStore

    main.service = Service()
    main.service.store = AppStore(tmp_path / "app.db")
    return main


def _student(main):
    store = main.service.store
    store.register("checker", "passw0rd", "C", "student", stage="class_12", stream="pcm")
    user = store.authenticate("checker", "passw0rd")
    return user, store.create_session(user.id)


def test_a_signed_in_student_keeps_the_result_and_a_visitor_does_not(main):
    key = answer_key("jee_main", "Physics", "Optics")
    body = main.SelfCheckAnswers(exam="jee_main", section="Physics", unit="Optics",
                                 answers=key)
    assert main.selfcheck_grade(body, None)["saved"] is False

    _, token = _student(main)
    out = main.selfcheck_grade(body, token)
    assert out["saved"] is True and out["score"] == 100
    checks = main.my_selfchecks(token)["checks"]
    assert [(c["unit"], c["score"], c["tries"]) for c in checks] == [("Optics", 100, 1)]


def test_a_self_check_never_becomes_a_study_event(main):
    """The model reads study_events. An exam unit is not a week of the student's
    course, so a self-check result must not appear there."""
    user, token = _student(main)
    body = main.SelfCheckAnswers(exam="jee_main", section="Physics", unit="Optics",
                                 answers=answer_key("jee_main", "Physics", "Optics"))
    main.selfcheck_grade(body, token)
    assert main.service.store.events(user.id) == []


def test_the_exam_endpoints_say_which_units_have_a_check(main):
    listed = main.exam_list("pcm")
    jee = next(e for e in listed["exams"] if e["key"] == "jee_main")
    cuet = next(e for e in listed["exams"] if e["key"] == "cuet_ug")
    assert jee["n_checks"] == jee["n_units"] and cuet["n_checks"] == 0
    plan = main.exam_plan("neet_ug", 8)
    assert all(u["check"] for w in plan["weeks"] for u in w["units"])


def test_the_api_refuses_unknown_units_and_half_answered_checks(main):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as unknown:
        main.selfcheck_questions("jee_main", "Physics", "Astrology")
    assert unknown.value.status_code == 404
    half = main.SelfCheckAnswers(exam="jee_main", section="Physics", unit="Optics",
                                 answers={})
    with pytest.raises(HTTPException) as bad:
        main.selfcheck_grade(half, None)
    assert bad.value.status_code == 400


def test_deleting_an_account_removes_its_self_checks(main):
    _, token = _student(main)
    body = main.SelfCheckAnswers(exam="jee_main", section="Physics", unit="Optics",
                                 answers=answer_key("jee_main", "Physics", "Optics"))
    main.selfcheck_grade(body, token)
    assert main.service.store.delete_account("checker")
    con = main.service.store._connect()
    assert con.execute("SELECT COUNT(*) FROM self_checks").fetchone()[0] == 0
