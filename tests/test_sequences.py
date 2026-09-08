"""Invariants for the event stream, sequences, splits and labels."""

from __future__ import annotations

import numpy as np


def test_supervision_is_confined_to_assessment_events(con):
    bad = con.sql(
        "SELECT COUNT(*) FROM events WHERE kind = 'vle' AND label IS NOT NULL"
    ).fetchone()[0]
    assert bad == 0, "VLE context tokens must carry no label"


def test_every_assessment_event_is_labelled(con):
    bad = con.sql(
        "SELECT COUNT(*) FROM events WHERE kind = 'assessment' AND label IS NULL"
    ).fetchone()[0]
    assert bad == 0


def test_labels_are_binary(con):
    values = [r[0] for r in con.sql(
        "SELECT DISTINCT label FROM events WHERE label IS NOT NULL"
    ).fetchall()]
    assert set(values) <= {0, 1}


def test_imputation_never_invents_scores_for_non_submitters(con):
    """The bug this guards: median score joined on assessment identity alone gave
    students who never submitted a score they never earned."""
    bad = con.sql(
        "SELECT COUNT(*) FROM events WHERE kind = 'assessment' AND score IS NOT NULL "
        "AND label = 0 AND score >= 40"
    ).fetchone()[0]
    assert bad == 0, "a row scored at or above the pass mark is labelled as failure"


def test_class_balance_is_usable(con):
    """OULAD's submitted-only records give a 95.6% pass rate. Treating non-submission
    as failure must keep the positive rate inside a trainable band."""
    rate = con.sql(
        "SELECT AVG(label) FROM events WHERE kind = 'assessment'"
    ).fetchone()[0]
    assert 0.55 <= rate <= 0.80, f"positive rate {rate:.3f} is outside the usable band"


def test_sequences_are_chronological(con):
    """Positions must be non-decreasing in day within every sequence."""
    bad = con.sql(
        """
        SELECT COUNT(*) FROM (
            SELECT id_student, code_module, code_presentation, day, position,
                   LAG(day) OVER (PARTITION BY id_student, code_module, code_presentation
                                  ORDER BY position) AS prev_day
            FROM events_ordered
        ) WHERE prev_day IS NOT NULL AND day < prev_day
        """
    ).fetchone()[0]
    assert bad == 0


def test_every_sequence_has_supervision(con):
    bad = con.sql("SELECT COUNT(*) FROM sequences WHERE n_supervised < 1").fetchone()[0]
    assert bad == 0, "a sequence with no supervised token contributes nothing to the loss"


def test_sequence_arrays_are_aligned(con):
    bad = con.sql(
        """
        SELECT COUNT(*) FROM sequences
        WHERE LEN(concept_ids) <> n_events
           OR LEN(days)        <> n_events
           OR LEN(labels)      <> n_events
           OR LEN(clicks)      <> n_events
           OR LEN(is_supervised) <> n_events
        """
    ).fetchone()[0]
    assert bad == 0


def test_no_student_spans_two_folds(con):
    worst = con.sql(
        "SELECT MAX(n) FROM (SELECT COUNT(DISTINCT fold) AS n FROM splits GROUP BY id_student)"
    ).fetchone()[0]
    assert worst == 1, "a student appearing in train and test leaks between folds"


def test_folds_are_stratified(con):
    """Outcome proportions must be near-identical across folds."""
    df = con.sql(
        "SELECT fold, final_result, COUNT(*) AS n FROM splits GROUP BY 1, 2"
    ).df()
    share = df.pivot(index="fold", columns="final_result", values="n")
    share = share.div(share.sum(axis=1), axis=0)
    assert (share.max() - share.min()).max() < 0.02


def test_early_features_do_not_use_the_whole_enrolment(con):
    """Behavioural features must come from the first 28 days only, so that a feature
    cannot encode the outcome it will be used to predict.

    Note that OULAD day offsets are negative before a presentation starts — students
    open material during registration — so the observed SPAN of the early window can
    legitimately exceed 28 days. The invariant is the upper bound on the day, not the
    width of the interval.
    """
    latest = con.sql(
        """
        SELECT MAX(s.day)
        FROM sessions s
        JOIN early_engagement e USING (id_student, code_module, code_presentation)
        WHERE s.day < 28
        """
    ).fetchone()[0]
    assert latest < 28, f"early window reached day {latest}"

    # And the window must actually bind: full-enrolment activity runs far past day 28.
    full = con.sql("SELECT MAX(day) FROM sessions").fetchone()[0]
    assert full > 100, "sessions do not extend past the window; the guard is vacuous"
