"""Entrance exams a school student in India can aim at, and a week by week plan.

Three exams cover most of what a class 12 student applies through: JEE Main for
engineering, NEET UG for medicine, and CUET UG for central university degrees. They sit
beside the course ladder rather than inside it: an exam is not a qualification, it is the
gate in front of one.

What is listed here is the published unit outline of each syllabus, not the syllabus
itself. Units are dropped and added between years, so every exam carries the official
link and the page says to check it. Nothing here is a prediction of what will be asked,
and no exam date, cut-off or rank is stated anywhere, because those change every year and
a wrong one would be worse than none.

CUET is built differently from the other two. Its domain papers follow whatever the
student took in class 12, so the plan is assembled from their own stream rather than from
a fixed list.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from elpr.course_finder import (PLAN_WEEKS, STREAM_SUBJECTS, _monday_of, spread,
                                study_links)

SCHOOLING = "class 12"          # every exam here is written on the class 11 and 12 syllabus


@dataclass(frozen=True)
class Exam:
    key: str
    name: str
    full_name: str
    body: str                   # who conducts it
    streams: tuple              # class 12 streams it is open to
    leads_to: str
    source: str                 # the official syllabus page
    sections: tuple             # ((section, (unit, ...)), ...)
    note: str = ""


JEE_MAIN = Exam(
    key="jee_main",
    name="JEE Main",
    full_name="Joint Entrance Examination (Main), Paper 1",
    body="National Testing Agency",
    streams=("pcm", "pcmb"),
    leads_to="B.E. and B.Tech at NITs, IIITs and other centrally funded institutes, and "
             "the JEE Advanced attempt that leads to the IITs",
    source="https://jeemain.nta.nic.in/",
    note="Paper 1 only, the engineering paper. Papers 2A and 2B are for architecture and "
         "planning and have their own syllabus.",
    sections=(
        ("Physics", (
            "Units and Measurements", "Kinematics", "Laws of Motion",
            "Work, Energy and Power", "Rotational Motion", "Gravitation",
            "Properties of Solids and Liquids", "Thermodynamics",
            "Kinetic Theory of Gases", "Oscillations and Waves", "Electrostatics",
            "Current Electricity", "Magnetic Effects of Current and Magnetism",
            "Electromagnetic Induction and Alternating Currents", "Electromagnetic Waves",
            "Optics", "Dual Nature of Matter and Radiation", "Atoms and Nuclei",
            "Electronic Devices", "Experimental Skills")),
        ("Chemistry", (
            "Some Basic Concepts in Chemistry", "Atomic Structure",
            "Chemical Bonding and Molecular Structure", "Chemical Thermodynamics",
            "Solutions", "Equilibrium", "Redox Reactions and Electrochemistry",
            "Chemical Kinetics", "Classification of Elements and Periodicity in Properties",
            "p-Block Elements", "d- and f-Block Elements", "Co-ordination Compounds",
            "Purification and Characterisation of Organic Compounds",
            "Some Basic Principles of Organic Chemistry", "Hydrocarbons",
            "Organic Compounds Containing Halogens",
            "Organic Compounds Containing Oxygen",
            "Organic Compounds Containing Nitrogen", "Biomolecules",
            "Principles Related to Practical Chemistry")),
        ("Mathematics", (
            "Sets, Relations and Functions", "Complex Numbers and Quadratic Equations",
            "Matrices and Determinants", "Permutations and Combinations",
            "Binomial Theorem and its Simple Applications", "Sequence and Series",
            "Limit, Continuity and Differentiability", "Integral Calculus",
            "Differential Equations", "Co-ordinate Geometry",
            "Three Dimensional Geometry", "Vector Algebra", "Statistics and Probability",
            "Trigonometry")),
    ),
)

NEET_UG = Exam(
    key="neet_ug",
    name="NEET UG",
    full_name="National Eligibility cum Entrance Test (Undergraduate)",
    body="National Testing Agency",
    streams=("pcb", "pcmb"),
    leads_to="MBBS, BDS, BAMS, BHMS, veterinary and allied medical degrees across India",
    source="https://neet.nta.nic.in/",
    # The NMC syllabus for NEET (UG) 2026 (finalised 22 December 2025, unchanged since
    # the 2024 revision). Physics and Chemistry now share their unit names with JEE Main.
    note="Biology is half the paper and is set as two sections, Botany and Zoology. The "
         "units follow the NMC syllabus, which lists Biology as one subject.",
    sections=(
        ("Physics", (
            "Physics and Measurement", "Kinematics", "Laws of Motion",
            "Work, Energy and Power", "Rotational Motion", "Gravitation",
            "Properties of Solids and Liquids", "Thermodynamics", "Kinetic Theory of Gases",
            "Oscillations and Waves", "Electrostatics", "Current Electricity",
            "Magnetic Effects of Current and Magnetism",
            "Electromagnetic Induction and Alternating Currents", "Electromagnetic Waves",
            "Optics", "Dual Nature of Matter and Radiation", "Atoms and Nuclei",
            "Electronic Devices", "Experimental Skills")),
        ("Chemistry", (
            "Some Basic Concepts in Chemistry", "Atomic Structure",
            "Chemical Bonding and Molecular Structure", "Chemical Thermodynamics",
            "Solutions", "Equilibrium", "Redox Reactions and Electrochemistry",
            "Chemical Kinetics", "Classification of Elements and Periodicity in Properties",
            "p-Block Elements", "d- and f-Block Elements", "Coordination Compounds",
            "Purification and Characterisation of Organic Compounds",
            "Some Basic Principles of Organic Chemistry", "Hydrocarbons",
            "Organic Compounds Containing Halogens", "Organic Compounds Containing Oxygen",
            "Organic Compounds Containing Nitrogen", "Biomolecules",
            "Principles Related to Practical Chemistry")),
        ("Biology", (
            "Diversity in Living World", "Structural Organisation in Animals and Plants",
            "Cell Structure and Function", "Plant Physiology", "Human Physiology",
            "Reproduction", "Genetics and Evolution", "Biology and Human Welfare",
            "Biotechnology and Its Applications", "Ecology and Environment")),
    ),
)

# CUET's third section is the same for everyone; the domain papers are not, so they are
# filled in from the student's own stream when the plan is built.
CUET_COMMON = (
    ("Language", (
        "Reading Comprehension: factual passages", "Reading Comprehension: literary passages",
        "Reading Comprehension: narrative passages", "Verbal Ability",
        "Rearranging the Parts", "Choosing the Correct Word", "Synonyms and Antonyms",
        "Vocabulary in Context")),
    ("General Aptitude Test", (
        "General Knowledge", "Current Affairs", "General Mental Ability",
        "Numerical Ability", "Quantitative Reasoning",
        "Logical and Analytical Reasoning")),
)

CUET_UG = Exam(
    key="cuet_ug",
    name="CUET UG",
    full_name="Common University Entrance Test (Undergraduate)",
    body="National Testing Agency",
    streams=("pcm", "pcb", "pcmb", "commerce", "arts"),
    leads_to="undergraduate admission at central, state and participating private "
             "universities, including Delhi University, BHU and JNU",
    # NTA moved CUET UG to its own site; the old exams.nta.ac.in/CUET-UG/ page now 404s.
    source="https://cuet.nta.nic.in/",
    # Since 2025: 37 subjects (13 languages, 23 domain subjects and the General Aptitude
    # Test), at most five per candidate, chosen freely. Universities still set their own
    # subject rules, Delhi University counting only subjects studied in class 12.
    note="Choose up to five subjects in all, languages and the General Aptitude Test "
         "included. Since 2025 they need not match your class 12 subjects, but each "
         "university sets which subjects it accepts, and Delhi University counts only "
         "subjects studied in class 12, so this plan starts from your class 12 subjects.",
    sections=CUET_COMMON,
)

EXAMS: dict[str, Exam] = {e.key: e for e in (JEE_MAIN, NEET_UG, CUET_UG)}

PLAN_NOTE = ("A revision plan, not the official syllabus. Units keep the order the "
             "syllabus lists them in. Always check the official syllabus for the year "
             "being written.")


def for_stream(stream: str | None) -> list[Exam]:
    """Exams open to a class 12 stream, or all of them when no stream is chosen yet.

    A class 10 student has not picked a stream, so they get the whole list: seeing that
    NEET needs Biology is part of how the stream gets chosen."""
    if not stream:
        return list(EXAMS.values())
    return [e for e in EXAMS.values() if stream in e.streams]


def sections_for(key: str, stream: str | None = None) -> list[tuple[str, tuple]]:
    """The syllabus outline, with CUET's domain papers filled in from the stream."""
    exam = EXAMS.get(key)
    if exam is None:
        raise KeyError(key)
    sections = [(name, tuple(units)) for name, units in exam.sections]
    if exam.key == "cuet_ug":
        subjects = STREAM_SUBJECTS.get(stream or "", {}).get("core", [])
        domain = tuple(s for s in subjects if s != "English")
        if domain:
            sections.insert(0, ("Domain subjects", domain))
    return sections


def exam_plan(key: str, weeks: int = 24, start: str | None = None,
              stream: str | None = None) -> dict:
    """Spread one exam's syllabus over however many weeks are left before it.

    The same dealing as the course planner: units in syllabus order, the load as even as
    the list allows, no empty weeks, and real dates when a start is given."""
    exam = EXAMS.get(key)
    if exam is None:
        raise KeyError(key)
    if weeks < 1:
        raise ValueError("choose at least one week")
    units = [(section, unit) for section, topics in sections_for(key, stream)
             for unit in topics]
    weeks = min(int(weeks), len(units))

    begin = _monday_of(start) if start else None
    out = []
    for index, block in enumerate(spread(units, weeks)):
        week = {
            "week": index + 1,
            "sections": sorted({section for section, _ in block}),
            "units": [{"name": unit, "section": section,
                       "links": study_links(unit, SCHOOLING)} for section, unit in block],
        }
        if begin is not None:
            first = begin + timedelta(weeks=index)
            week["starts"] = first.isoformat()
            week["ends"] = (first + timedelta(days=6)).isoformat()
        out.append(week)
    return {
        "key": exam.key, "name": exam.name, "full_name": exam.full_name,
        "body": exam.body, "source": exam.source, "leads_to": exam.leads_to,
        "exam_note": exam.note, "stream": stream,
        "weeks": out, "n_weeks": weeks, "n_units": len(units),
        "starts": out[0].get("starts"), "ends": out[-1].get("ends"),
        "per_week": round(len(units) / weeks, 1),
        "options": sorted({w for w in PLAN_WEEKS if w <= len(units)} | {weeks}),
        "note": PLAN_NOTE,
    }


def summary(stream: str | None = None) -> dict:
    """Every exam the student can aim at, without the full syllabus behind each one."""
    return {
        "stream": stream,
        "exams": [{
            "key": e.key, "name": e.name, "full_name": e.full_name, "body": e.body,
            "leads_to": e.leads_to, "source": e.source, "note": e.note,
            "streams": list(e.streams),
            "sections": [{"name": name, "n_units": len(units)}
                         for name, units in sections_for(e.key, stream)],
            "n_units": sum(len(units) for _, units in sections_for(e.key, stream)),
        } for e in for_stream(stream)],
        "note": ("Exams are gates, not qualifications. Preparing for one does not replace "
                 "the class 12 course it is written on."),
    }
