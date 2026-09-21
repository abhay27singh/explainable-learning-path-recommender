"""FastAPI service: authentication, recommendations, adviser views, results.

Also the "RESTful API for LMS integration" the paper lists as future work — delivered
here rather than promised.

Access rules, enforced server-side rather than by hiding buttons:
  * a student may read and write only their own record;
  * an adviser may read any learner, including registered students;
  * nobody may act on another account by passing a different id.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import Cookie, FastAPI, HTTPException, Query, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse

from elpr.modules import module_display_map, subject_area
from elpr.profile import clean as clean_profile, options as profile_options
from elpr import course_finder, ics
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from elpr.db.app_store import User  # noqa: E402

if TYPE_CHECKING:                      # importing the service pulls in PyTorch, which
    from api.service import Service    # would delay the port opening by several seconds

app = FastAPI(title="Explainable Learning Path Recommender", version="1.0.0")
# The course list is 112 KB of JSON and the page itself 98 KB; both compress to about
# a tenth of that, which matters on a phone connection.
app.add_middleware(GZipMiddleware, minimum_size=1024)
service: Service | None = None
WEB = ROOT / "web"
COOKIE = "elpr_session"


@app.on_event("startup")
def load() -> None:
    """Warm the model in the background so the page is served straight away.

    Loading PyTorch, the checkpoint and the learner records takes about five seconds.
    Blocking startup on that left the port closed, so a refresh in those seconds showed
    "site can't be reached". Now the page loads immediately and the API answers 503 with
    a plain message until the model is ready; the interface waits and retries."""
    def warm() -> None:
        global service
        from api.service import Service

        start = time.perf_counter()
        loaded = Service()
        service = loaded
        print(f"loaded {loaded.checkpoint}, {loaded.graph.n_concepts} concepts, "
              f"in {time.perf_counter() - start:.1f}s")

    threading.Thread(target=warm, name="warm-model", daemon=True).start()


@app.middleware("http")
async def log_failed_requests(request, call_next):
    """Record failed API calls against the signed-in account.

    This is what makes a student's "it is not working" answerable: the admin can see
    the request that failed and when. Paths and status codes only, never form data."""
    response = await call_next(request)
    if response.status_code >= 400 and request.url.path.startswith("/api/") and service:
        try:
            user = _current(request.cookies.get(COOKIE))
            if user is not None:
                service.store.log(user.id, f"error {response.status_code}",
                                  f"{request.method} {request.url.path}")
        except Exception:                      # logging must never break a response
            pass
    return response


def _service() -> Service:
    if service is None:
        raise HTTPException(503, "starting up, this takes a few seconds")
    return service


def _current(token: str | None) -> User | None:
    return _service().store.user_for_session(token)


def _require(token: str | None) -> User:
    user = _current(token)
    if user is None:
        raise HTTPException(401, "sign in required")
    return user


def _require_adviser(token: str | None) -> User:
    """Adviser routes. An admin is a superset of an adviser, so it passes too."""
    user = _require(token)
    if user.role not in ("adviser", "admin"):
        raise HTTPException(403, "adviser role required")
    return user


def _require_admin(token: str | None) -> User:
    user = _require(token)
    if user.role != "admin":
        raise HTTPException(403, "admin role required")
    return user


def _overrides(value: str | None) -> tuple:
    if not value:
        return ()
    return tuple(int(v) for v in value.split(",") if v.strip().lstrip("-").isdigit())


# ---------------------------------------------------------------- auth
class Credentials(BaseModel):
    username: str = Field(min_length=3, max_length=40)
    # The store enforces the real minimum on registration, with a readable message.
    password: str = Field(min_length=1, max_length=200)


class Registration(Credentials):
    display_name: str = Field(default="", max_length=80)
    role: str = Field(default="student")
    module: str | None = None
    stage: str | None = None
    stream: str | None = None


def _studies(s: Service, stage: str | None, module: str | None,
             stream: str | None = None) -> tuple:
    """Validate a student's level, course and class 12 stream.

    Only diploma and degree students have a course; only class 12 students have a stream."""
    if stage not in course_finder.STAGES:
        raise HTTPException(400, "choose where you are in your studies")
    if stage == "class_12":
        if stream is not None and stream not in course_finder.STREAMS:
            raise HTTPException(400, "unknown class 12 stream")
        return stage, None, stream
    if stage not in course_finder.COURSE_STAGES:
        return stage, None, None
    if module not in s.modules:
        raise HTTPException(400, "choose the course you are studying")
    return stage, module, None


def _stage(user: User | None) -> str | None:
    """Accounts made before levels existed were all on a university course."""
    if user is None or user.role != "student":
        return None
    return user.stage or ("ug" if user.module else None)


@app.post("/api/auth/register")
def register(body: Registration, response: Response) -> dict:
    s = _service()
    stage = module = stream = None
    if body.role == "student":
        stage, module, stream = _studies(s, body.stage, body.module, body.stream)
    try:
        user = s.store.register(
            body.username, body.password, body.display_name, body.role, module, stage, stream
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    token = s.store.create_session(user.id)
    s.store.log(user.id, "signed up", f"{user.role}, {stage or 'no level'}")
    response.set_cookie(COOKIE, token, httponly=True, samesite="lax", max_age=12 * 3600)
    return _me(user)


@app.post("/api/auth/login")
def login(body: Credentials, response: Response) -> dict:
    s = _service()
    wait = s.store.locked_out(body.username)
    if wait:
        raise HTTPException(429, f"too many sign-in attempts, try again in {wait // 60 + 1} minutes")
    user = s.store.authenticate(body.username, body.password)
    if user is None:
        # Same message either way — never reveal whether the username exists.
        # The attempt is logged against the account when the username is real, which is
        # what lets an admin answer "I cannot sign in". The password is never recorded.
        known = s.store.user_by_username(body.username)
        if known is not None:
            s.store.log(known.id, "sign-in failed", "wrong password")
        raise HTTPException(401, "incorrect username or password")
    s.store.log(user.id, "signed in")
    token = s.store.create_session(user.id)
    response.set_cookie(COOKIE, token, httponly=True, samesite="lax", max_age=12 * 3600)
    return _me(user)


@app.post("/api/auth/logout")
def logout(response: Response, elpr_session: str | None = Cookie(None)) -> dict:
    if elpr_session:
        _service().store.end_session(elpr_session)
    response.delete_cookie(COOKIE)
    return {"ok": True}


def _me(user: User) -> dict:
    return {
        "username": user.username, "display_name": user.display_name,
        "role": user.role, "module": user.module, "student_id": user.student_id,
        "stage": _stage(user), "stream": user.stream,
    }


@app.get("/api/auth/me")
def me(elpr_session: str | None = Cookie(None)) -> dict:
    user = _current(elpr_session)
    return {"signed_in": user is not None, **(_me(user) if user else {})}


# ---------------------------------------------------------------- meta
@app.get("/api/health")
def health() -> dict:
    s = _service()
    return {
        "status": "ok", "checkpoint": s.checkpoint,
        "n_concepts": s.graph.n_concepts,
        "n_learners": int(s.sequences.id_student.nunique()),
        "modules": s.modules,
        "module_names": module_display_map(s.modules),
        "module_areas": {m: subject_area(m) for m in s.modules},
    }


# ---------------------------------------------------------------- my record
@app.get("/api/me/state")
def my_state(known: str | None = None, elpr_session: str | None = Cookie(None)) -> dict:
    user = _require(elpr_session)
    s = _service()
    if user.role != "student":
        raise HTTPException(400, "advisers have no learning record of their own")

    module, state = s.registered_state(user, _overrides(known))
    events = s.store.events(user.id)
    scope = state.scope()
    return {
        "student_id": user.student_id, "module": module,
        "n_events": len(events),
        "n_assessments": sum(1 for e in events if e["kind"] == "assessment"),
        "thresholds": {"mastered": round(state.thresholds.mastered, 4),
                       "prerequisite": round(state.thresholds.prerequisite, 4)},
        "concepts": [
            {"concept": int(c), "label": s.graph.label(int(c)),
             "week": int(s.graph.concepts.week.iloc[int(c)]),
             "mastery": round(float(state.mastery[c]), 4),
             "in_scope": bool(scope[c])}
            for c in range(s.graph.n_concepts) if scope[c]
        ],
        "history": events[-25:],
    }


@app.get("/api/me/recommend")
def my_recommendations(
    k: int = Query(3, le=10), planner: str = "greedy",
    known: str | None = None, elpr_session: str | None = Cookie(None),
) -> dict:
    user = _require(elpr_session)
    s = _service()
    if user.role != "student":
        raise HTTPException(400, "advisers have no learning record of their own")

    start = time.perf_counter()
    module, state = s.registered_state(user, _overrides(known))
    engine = s.planners.get(planner, s.planners["greedy"])
    actions = engine.score(state)[:k]
    payload = {
        "student_id": user.student_id, "module": module, "planner": planner,
        "n_eligible": int(len(engine.candidates(state)[0])),
        "recommendations": [s.explainer.explain(state, a).to_dict() for a in actions],
        "elapsed_ms": round((time.perf_counter() - start) * 1000, 1),
    }
    s.store.log_recommendation(user.id, user.student_id, planner, payload)
    return payload


@app.get("/api/me/path")
def my_path(steps: int = Query(5, ge=1, le=8), known: str | None = None,
            elpr_session: str | None = Cookie(None)) -> dict:
    """The student's course week by week from the first week not yet done.

    A week studied or passed counts as done, so pressing either button moves the path
    on; "I found it hard" keeps the week in place."""
    user = _require(elpr_session)
    s = _service()
    if user.role != "student":
        raise HTTPException(400, "advisers have no learning record of their own")
    if not user.module:
        raise HTTPException(400, "choose your course first")
    return s.course_path(user, steps, _overrides(known))


@app.get("/api/students/{student}/path")
def learner_path(student: int, steps: int = Query(5, ge=1, le=8), known: str | None = None,
                 elpr_session: str | None = Cookie(None)) -> dict:
    user = _require(elpr_session)
    s = _service()
    if user.role not in ("adviser", "admin") and user.student_id != student:
        raise HTTPException(403, "you may only view your own record")
    if s.is_registered(student):
        target = s.store.user_by_student_id(student)
        if target is None:
            raise HTTPException(404, "unknown learner")
        return s.learning_path(lambda ov: s.registered_state(target, ov), steps,
                               _overrides(known))
    try:
        return s.dataset_path(student, steps, _overrides(known))
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/api/me/progress")
def my_progress(elpr_session: str | None = Cookie(None)) -> dict:
    """Streak, activity calendar, badges and weekly summary from recorded study events."""
    user = _require(elpr_session)
    if user.role != "student":
        raise HTTPException(400, "only students have a progress record")
    return _service().progress_for(user)


class StudyEvent(BaseModel):
    concept_id: int
    kind: str = Field(default="study")
    correct: bool | None = None
    # A quiz mark out of 100. When given it decides pass or fail, on the same threshold
    # the model was trained with, so a typed score means the same as a dataset one.
    score: float | None = Field(default=None, ge=0, le=100)


@app.post("/api/me/study")
def record_study(body: StudyEvent, elpr_session: str | None = Cookie(None)) -> dict:
    user = _require(elpr_session)
    s = _service()
    if user.role != "student":
        raise HTTPException(400, "only students record study activity")
    if not 0 <= body.concept_id < s.graph.n_concepts:
        raise HTTPException(400, "unknown concept")
    if body.kind not in ("study", "assessment"):
        raise HTTPException(400, "kind must be study or assessment")
    if body.score is not None and body.kind != "assessment":
        raise HTTPException(400, "a score belongs to a test, not to reading")
    s.store.add_event(user.id, body.concept_id, body.kind, body.correct, body.score)
    outcome = ("" if body.score is not None else
               "" if body.correct is None else
               f", {'passed' if body.correct else 'found it hard'}")
    s.store.log(user.id, "recorded activity",
                f"week concept {body.concept_id}, {body.kind}" + outcome
                + ("" if body.score is None else f", scored {body.score:g} out of 100"))
    return {"ok": True, "pass_mark": s.store.PASS_MARK, "n_events": len(s.store.events(user.id))}


@app.post("/api/me/undo")
def undo(elpr_session: str | None = Cookie(None)) -> dict:
    user = _require(elpr_session)
    store = _service().store
    undone = store.undo_last_event(user.id)
    store.log(user.id, "undid last activity", "" if undone else "nothing to undo")
    return {"ok": undone}


# ---------------------------------------------------------------- course finder
@app.get("/api/course-finder/options")
def course_finder_options() -> dict:
    """Class 12 streams and interest areas. Public: a new student has no account yet."""
    return course_finder.options()


class FinderBody(BaseModel):
    level: str = "ug"
    stream: str | None = None
    subjects: list[str] = Field(default_factory=list)
    degree: str | None = None
    interests: list[str] = Field(default_factory=list)


@app.post("/api/course-finder")
def course_finder_suggest(body: FinderBody, elpr_session: str | None = Cookie(None)) -> dict:
    """Rule-based course suggestions. No personal data is stored.

    A signed-in student is offered only the level after their own: an undergraduate
    sees postgraduate courses, not diplomas after class 10."""
    stage = _stage(_current(elpr_session)) if elpr_session else None
    if stage and body.level not in course_finder.NEXT_LEVELS[stage]:
        raise HTTPException(400, f"that level is not the next step after {course_finder.STAGES[stage]}")
    try:
        return course_finder.recommend(level=body.level, interests=body.interests,
                                       stream=body.stream, subjects=body.subjects,
                                       degree=body.degree)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


class SaveCourseBody(BaseModel):
    key: str


@app.get("/api/me/courses")
def my_saved_courses(elpr_session: str | None = Cookie(None)) -> dict:
    """The student's shortlist, newest first, with subjects and study links."""
    user = _require(elpr_session)
    if user.role != "student":
        raise HTTPException(400, "only students have a shortlist")
    keys = _service().store.saved_courses(user.id)
    return {"courses": [course_finder.course(k) for k in keys if course_finder.exists(k)]}


@app.post("/api/me/courses")
def save_course(body: SaveCourseBody, elpr_session: str | None = Cookie(None)) -> dict:
    user = _require(elpr_session)
    if user.role != "student":
        raise HTTPException(400, "only students have a shortlist")
    if not course_finder.exists(body.key):
        raise HTTPException(404, "no such course")
    store = _service().store
    store.save_course(user.id, body.key)
    store.log(user.id, "saved a course", body.key)
    return {"saved": body.key}


@app.delete("/api/me/courses/{key}")
def unsave_course(key: str, elpr_session: str | None = Cookie(None)) -> dict:
    user = _require(elpr_session)
    if user.role != "student":
        raise HTTPException(400, "only students have a shortlist")
    return {"removed": _service().store.remove_course(user.id, key)}


@app.get("/api/course-finder/subjects")
def course_finder_subjects(stage: str, stream: str | None = None) -> dict:
    """What a student studies at this rung of the ladder, and what the next rung is.

    Public: the Course Finder walks class 10, class 12, diploma, degree, postgraduate
    for visitors as well as registered students."""
    try:
        now = (course_finder.subjects_now(stage, stream)
               if stage != "class_12" or stream else
               {"stage": stage, "label": course_finder.STAGES[stage], "subjects": [],
                "optional": [], "note": ""})
        return {"now": now, "next": course_finder.next_after(stage, stream)}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/course-finder/ahead")
def course_finder_ahead(stream: str, level: str = "ug") -> dict:
    """What a class 12 stream opens up later. Read-only, and public."""
    try:
        return course_finder.looking_ahead(stream, level)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/course-finder/plan")
def course_finder_plan(key: str, weeks: int = Query(24, ge=1, le=260),
                       start: str | None = None) -> dict:
    """One course's subjects spread over the number of weeks the student chooses.

    With a start date every week carries real dates, which is what makes the plan
    something a calendar can hold."""
    try:
        return course_finder.weekly_plan(key, weeks, start)
    except KeyError:
        raise HTTPException(404, "no such course")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/course-finder/plan.ics")
def course_finder_plan_ics(key: str, weeks: int = Query(24, ge=1, le=260),
                           start: str | None = None) -> Response:
    """The same plan as a calendar file: one all-day event per week."""
    from datetime import date as _date

    try:
        plan = course_finder.weekly_plan(key, weeks, start or _date.today().isoformat())
    except KeyError:
        raise HTTPException(404, "no such course")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    events = [{
        "uid": f"{plan['key']}-w{week['week']}@learning-path",
        "start": _date.fromisoformat(week["starts"]),
        "days": 7,
        "summary": f"{plan['name']} · Week {week['week']}",
        "description": "Study this week: "
                       + ", ".join(s["name"] for s in week["subjects"]),
    } for week in plan["weeks"]]
    body = ics.calendar(f"{plan['name']} study plan", events)
    return Response(content=body, media_type="text/calendar; charset=utf-8",
                    headers={"Content-Disposition":
                             f'attachment; filename="{plan["key"]}-study-plan.ics"'})


@app.get("/api/me/path.ics")
def my_path_ics(weeks: int = Query(8, ge=1, le=52), start: str | None = None,
                elpr_session: str | None = Cookie(None)) -> Response:
    """A student's own course path as a calendar: the next weeks, one event each."""
    from datetime import date as _date, timedelta

    user = _require(elpr_session)
    s = _service()
    if user.role != "student":
        raise HTTPException(400, "advisers have no learning record of their own")
    if not user.module:
        raise HTTPException(400, "choose your course first")
    try:
        begin = _date.fromisoformat(start) if start else _date.today()
    except ValueError:
        raise HTTPException(400, "give the start date as YYYY-MM-DD")
    begin -= timedelta(days=begin.weekday())         # weeks run Monday to Sunday

    path = s.course_path(user, min(weeks, 8))
    events = [{
        "uid": f"{user.student_id}-c{step['concept']}@learning-path",
        "start": begin + timedelta(weeks=index),
        "days": 7,
        "summary": f"{module_display_map(s.modules).get(path['module'], path['module'])} · "
                   f"{step['label'].split('·')[-1].strip().split(' (')[0]}",
        "description": step["text"],
    } for index, step in enumerate(path["steps"])]
    body = ics.calendar("My study path", events)
    return Response(content=body, media_type="text/calendar; charset=utf-8",
                    headers={"Content-Disposition": 'attachment; filename="my-study-path.ics"'})


@app.get("/api/course-finder/explore")
def course_finder_explore(level: str | None = None) -> dict:
    """Every course, grouped by level and category, with no eligibility filtering. Public."""
    try:
        return course_finder.explore(level)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


# ---------------------------------------------------------------- background form
@app.get("/api/graph/{module}")
def course_graph(module: str, elpr_session: str | None = Cookie(None)) -> dict:
    """Weeks of one course and their prerequisite links. Structure only, no personal data."""
    _require(elpr_session)
    try:
        return _service().module_graph(module)
    except KeyError:
        raise HTTPException(404, "unknown course")


@app.get("/api/profile/options")
def profile_form() -> dict:
    """The background questions and their allowed answers. Contains no personal data."""
    from elpr.profile import UK_ONLY_NOTE

    return {"fields": profile_options(), "note": UK_ONLY_NOTE}


class ProfileBody(BaseModel):
    profile: dict = Field(default_factory=dict)


@app.get("/api/me/profile")
def my_profile(elpr_session: str | None = Cookie(None)) -> dict:
    user = _require(elpr_session)
    if user.role != "student":
        raise HTTPException(400, "only students have a learning profile")
    return {"profile": _service().store.get_profile(user.id), "module": user.module}


@app.put("/api/me/profile")
def update_profile(body: ProfileBody, elpr_session: str | None = Cookie(None)) -> dict:
    """Save background answers. Anything unrecognised is dropped, never stored."""
    user = _require(elpr_session)
    if user.role != "student":
        raise HTTPException(400, "only students have a learning profile")
    store = _service().store
    cleaned = clean_profile(body.profile)
    store.set_profile(user.id, cleaned)
    store.log(user.id, "saved background details", f"{len(cleaned)} answers")
    return {"profile": cleaned}


class StudiesBody(BaseModel):
    stage: str
    module: str | None = None
    stream: str | None = None


@app.put("/api/me/studies")
def update_studies(body: StudiesBody, elpr_session: str | None = Cookie(None)) -> dict:
    """Where the student is now: level, course if they are on one, class 12 stream if not."""
    user = _require(elpr_session)
    s = _service()
    if user.role != "student":
        raise HTTPException(400, "only students have studies to set")
    stage, module, stream = _studies(s, body.stage, body.module, body.stream)
    s.store.set_studies(user.id, stage, module, stream)
    s.store.log(user.id, "changed studies",
                ", ".join(x for x in (stage, module, stream) if x))
    return _me(replace(user, stage=stage, module=module, stream=stream))


@app.get("/api/me/study-plan")
def my_study_plan(elpr_session: str | None = Cookie(None)) -> dict:
    """What a school student is studying now, with free study links, and what comes next.

    The ladder is class 10, then class 12 or a diploma, then a degree, then postgraduate
    study. A student already on a course has a weekly path instead."""
    user = _require(elpr_session)
    if user.role != "student":
        raise HTTPException(400, "only students have a study plan")
    stage = _stage(user)
    if stage is None:
        raise HTTPException(400, "choose where you are in your studies")
    now = {"stage": stage, "label": course_finder.STAGES[stage], "subjects": [],
           "optional": [], "note": ""}
    if stage == "class_10" or (stage == "class_12" and user.stream):
        now = course_finder.subjects_now(stage, user.stream)
    return {"stream": user.stream, "now": now, "next": course_finder.next_after(stage, user.stream)}


# ---------------------------------------------------------------- admin
@app.get("/api/admin/accounts/{username}/log")
def admin_account_log(username: str, limit: int = Query(100, ge=1, le=200),
                      elpr_session: str | None = Cookie(None)) -> dict:
    """What happened on one account, newest first, so an admin can answer a problem."""
    _require_admin(elpr_session)
    store = _service().store
    if store.user_by_username(username) is None:
        raise HTTPException(404, "no such account")
    return {"username": username.strip().lower(), "entries": store.logs(username, limit)}


@app.get("/api/admin/accounts")
def admin_accounts(elpr_session: str | None = Cookie(None)) -> dict:
    """Every registered account, with activity. Never returns salts or hashes."""
    _require_admin(elpr_session)
    store = _service().store
    return {"accounts": store.list_accounts(), "stats": store.stats()}


@app.delete("/api/admin/accounts/{username}")
def admin_delete_account(username: str,
                         elpr_session: str | None = Cookie(None)) -> dict:
    """Delete an account and its study history. An admin cannot delete itself."""
    me = _require_admin(elpr_session)
    if username.strip().lower() == me.username:
        raise HTTPException(400, "an admin cannot delete its own account")
    if not _service().store.delete_account(username):
        raise HTTPException(404, "no such account")
    return {"deleted": username}


# ---------------------------------------------------------------- adviser
@app.get("/api/students")
def students(q: str = "", limit: int = Query(40, le=200),
             elpr_session: str | None = Cookie(None)) -> dict:
    _require_adviser(elpr_session)
    s = _service()
    registered = [
        {"id_student": r["student_id"], "module": r["module"] or "n/a",
         "presentation": "registered", "n_events": r["n_events"] or 0,
         "n_assessments": r["n_assessments"] or 0,
         "outcome": "in progress", "display_name": r["display_name"],
         "registered": True, "quiet": r["quiet"], "quiet_days": r["quiet_days"],
         "never_started": r["never_started"]}
        for r in s.store.registered_students()
        if not q or str(r["student_id"]).startswith(q)
        or q.lower() in (r["display_name"] or "").lower()
    ]
    dataset = [{**r, "registered": False} for r in s.search(q, limit)]
    return {"registered": registered, "dataset": dataset}


@app.get("/api/students/{student}/mastery")
def mastery(student: int, module: str | None = None, upto: float = 1.0,
            known: str | None = None, elpr_session: str | None = Cookie(None)) -> dict:
    user = _require(elpr_session)
    s = _service()
    # A student may only look at themselves.
    if user.role not in ("adviser", "admin") and user.student_id != student:
        raise HTTPException(403, "you may only view your own record")

    if s.is_registered(student):
        target = s.store.user_by_student_id(student)
        if target is None:
            raise HTTPException(404, "unknown learner")
        mod, state = s.registered_state(target, _overrides(known))
        scope = state.scope()
        return {
            "student": student, "module": mod, "registered": True,
            "n_events": len(s.store.events(target.id)),
            "thresholds": {"mastered": round(state.thresholds.mastered, 4),
                           "prerequisite": round(state.thresholds.prerequisite, 4)},
            "concepts": [
                {"concept": int(c), "label": s.graph.label(int(c)),
                 "week": int(s.graph.concepts.week.iloc[int(c)]),
                 "module": str(s.graph.concepts.code_module.iloc[int(c)]),
                 "mastery": round(float(state.mastery[c]), 4), "in_scope": bool(scope[c])}
                for c in range(s.graph.n_concepts)
            ],
        }
    try:
        return {**s.mastery_for(student, module, upto, _overrides(known)),
                "registered": False}
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/api/recommend")
def recommend(student: int, k: int = Query(3, le=10), planner: str = "greedy",
              module: str | None = None, upto: float = 1.0, known: str | None = None,
              elpr_session: str | None = Cookie(None)) -> dict:
    user = _require(elpr_session)
    s = _service()
    if user.role not in ("adviser", "admin") and user.student_id != student:
        raise HTTPException(403, "you may only view your own record")

    start = time.perf_counter()
    if s.is_registered(student):
        target = s.store.user_by_student_id(student)
        if target is None:
            raise HTTPException(404, "unknown learner")
        mod, state = s.registered_state(target, _overrides(known))
        engine = s.planners.get(planner, s.planners["greedy"])
        actions = engine.score(state)[:k]
        payload = {
            "student": student, "module": mod, "presentation": "registered",
            "planner": planner, "registered": True,
            "n_eligible": int(len(engine.candidates(state)[0])),
            "thresholds": {"mastered": round(state.thresholds.mastered, 4),
                           "prerequisite": round(state.thresholds.prerequisite, 4)},
            "recommendations": [s.explainer.explain(state, a).to_dict() for a in actions],
        }
    else:
        try:
            payload = s.recommend(student, k, planner, module, upto, _overrides(known))
            payload["registered"] = False
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    payload["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 1)
    s.store.log_recommendation(user.id, student, planner, payload)
    return payload


class NoteBody(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


@app.get("/api/students/{student}/notes")
def learner_notes(student: int, elpr_session: str | None = Cookie(None)) -> dict:
    """Adviser notes on one learner. Advisers and admins only, never the student."""
    _require_adviser(elpr_session)
    return {"student": student, "notes": _service().store.notes(student)}


@app.post("/api/students/{student}/notes")
def add_learner_note(student: int, body: NoteBody,
                     elpr_session: str | None = Cookie(None)) -> dict:
    user = _require_adviser(elpr_session)
    s = _service()
    try:
        note_id = s.store.add_note(student, user.id, body.text)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"id": note_id, "notes": s.store.notes(student)}


@app.delete("/api/students/{student}/notes/{note_id}")
def delete_learner_note(student: int, note_id: int,
                        elpr_session: str | None = Cookie(None)) -> dict:
    """An adviser may delete their own notes; an admin may delete any."""
    user = _require_adviser(elpr_session)
    s = _service()
    removed = s.store.delete_note(note_id, user.id)
    if not removed and user.role == "admin":
        removed = s.store.delete_note_any(note_id)
    if not removed:
        raise HTTPException(404, "no note of yours with that id")
    return {"deleted": note_id, "notes": s.store.notes(student)}


@app.get("/api/adviser/overview")
def overview(elpr_session: str | None = Cookie(None)) -> dict:
    _require_adviser(elpr_session)
    s = _service()
    return {"registered": s.store.registered_students(), "stats": s.store.stats(),
            "quiet_after_days": s.store.QUIET_AFTER_DAYS}


# ---------------------------------------------------------------- results
@app.get("/api/graph")
def graph(module: str | None = None, elpr_session: str | None = Cookie(None)) -> dict:
    _require(elpr_session)
    return _service().graph_payload(module)


@app.get("/api/metrics")
def metrics(elpr_session: str | None = Cookie(None)) -> dict:
    """Study results, read live from results/ so the site cannot drift from the paper.
    Admin only: the research is kept as validation, not shown to students."""
    import json

    _require_admin(elpr_session)
    out = {}
    for name in ("table1", "table2", "explainability", "graph_stats", "dataset_stats"):
        path = ROOT / "results" / f"{name}.json"
        if path.exists():
            out[name] = json.loads(path.read_text())
    return out


if WEB.exists():
    app.mount("/static", StaticFiles(directory=WEB), name="static")

    @app.get("/")
    def index() -> FileResponse:
        # The page is the whole application, so a cached copy means a student keeps
        # seeing yesterday's version after an update. Never cache it.
        return FileResponse(WEB / "index.html",
                            headers={"Cache-Control": "no-store, must-revalidate"})
