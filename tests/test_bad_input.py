"""Odd values in an address are refused with a reason, never answered with a crash.

Each case here answered "500 Internal Server Error" or a silently wrong record when the
whole API was probed with out-of-range values before launch.
"""
from __future__ import annotations

from typing import Annotated

import pytest
from fastapi import HTTPException
from pydantic import TypeAdapter, ValidationError


@pytest.fixture(scope="module")
def main(tmp_path_factory):
    from api import main
    from api.service import Service
    from elpr.db.app_store import AppStore

    main.service = Service()
    main.service.store = AppStore(tmp_path_factory.mktemp("db") / "app.db")
    main.service.store.register("ravi", "passw0rd", "Ravi", "student",
                                stage="class_12", stream="pcm")
    return main


def test_week_ids_outside_the_course_list_are_dropped(main):
    """Regression: an id past the last week answered 500, and -1 read as the last week."""
    n = main.service.graph.n_concepts
    assert main._overrides(f"3,{n},{n + 5},-1,abc,3, 4 ,999999999999999999999") == (3, 4)
    assert main._overrides(None) == () and main._overrides("") == ()


def test_how_far_through_a_course_is_a_share_of_it():
    """Regression: upto=nan reached the dataset code and answered 500."""
    from api.main import UPTO

    share = TypeAdapter(Annotated[float, UPTO])
    assert share.validate_python(0.5) == 0.5
    for bad in (float("nan"), float("inf"), 1.5, -0.1):
        with pytest.raises(ValidationError):
            share.validate_python(bad)


def test_asking_for_no_recommendations_is_refused():
    """k=0 and k=-5 used to be accepted; a negative k sliced the list from the end."""
    import inspect

    from api import main
    for route in (main.recommend, main.my_recommendations):
        k = inspect.signature(route).parameters["k"].default
        assert any(getattr(m, "ge", None) == 1 for m in k.metadata), route.__name__


def test_a_school_student_has_no_course_record_to_send(main):
    """Regression: a class 12 student's own state, recommendations and progress came back
    as the first course's, an invented Psychology record."""
    store = main.service.store
    token = store.create_session(store.user_by_username("ravi").id)
    for call in (lambda: main.my_state(None, False, token),
                 lambda: main.my_recommendations(3, "greedy", None, token),
                 lambda: main.my_progress(token)):
        with pytest.raises(HTTPException) as refused:
            call()
        assert refused.value.status_code == 409


def test_a_plan_starts_on_a_date_the_calendar_can_reach():
    """Regression: 9999-12-31 is a real date, and adding the plan's weeks ran past the
    last date Python has."""
    from datetime import date

    from elpr.course_finder import _monday_of

    assert _monday_of("2026-10-08") == date(2026, 10, 5), "plans start on the Monday"
    for bad in ("9999-12-31", "0001-01-01", "2026-02-30", "soon"):
        with pytest.raises(ValueError):
            _monday_of(bad)


def test_a_missing_learner_is_named_without_quotes():
    """str(KeyError("no learner 1")) is "'no learner 1'", quotes and all."""
    from api.main import _missing

    assert _missing(KeyError("no learner 1")) == "no learner 1"
