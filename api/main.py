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
import time
from dataclasses import replace
from pathlib import Path

from fastapi import Cookie, FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse

from elpr.modules import module_display_map, subject_area
from elpr.profile import clean as clean_profile, options as profile_options
from elpr import course_finder
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.service import Service  # noqa: E402
from elpr.db.app_store import User  # noqa: E402

app = FastAPI(title="Explainable Learning Path Recommender", version="1.0.0")
service: Service | None = None
WEB = ROOT / "web"
COOKIE = "elpr_session"


@app.on_event("startup")
def load() -> None:
    global service
    start = time.perf_counter()
    service = Service()
    print(f"loaded {service.checkpoint}, {service.graph.n_concepts} concepts, "
          f"in {time.perf_counter() - start:.1f}s")


def _service() -> Service:
    if service is None:
        raise HTTPException(503, "model still loading")
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


def _studies(s: Service, stage: str | None, module: str | None) -> tuple:
    """Validate a student's level and course. Only diploma and degree students have a course."""
    if stage not in course_finder.STAGES:
        raise HTTPException(400, "choose where you are in your studies")
    if stage not in course_finder.COURSE_STAGES:
        return stage, None
    if module not in s.modules:
        raise HTTPException(400, "choose the course you are studying")
    return stage, module


def _stage(user: User | None) -> str | None:
    """Accounts made before levels existed were all on a university course."""
    if user is None or user.role != "student":
        return None
    return user.stage or ("ug" if user.module else None)


@app.post("/api/auth/register")
def register(body: Registration, response: Response) -> dict:
    s = _service()
    stage = module = None
    if body.role == "student":
        stage, module = _studies(s, body.stage, body.module)
    try:
        user = s.store.register(
            body.username, body.password, body.display_name, body.role, module, stage
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    token = s.store.create_session(user.id)
    response.set_cookie(COOKIE, token, httponly=True, samesite="lax", max_age=12 * 3600)
    return _me(user)


@app.post("/api/auth/login")
def login(body: Credentials, response: Response) -> dict:
    s = _service()
    user = s.store.authenticate(body.username, body.password)
    if user is None:
        # Same message either way — never reveal whether the username exists.
        raise HTTPException(401, "incorrect username or password")
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
        "stage": _stage(user),
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
        make = lambda ov: s.registered_state(target, ov)
    else:
        def make(ov):
            try:
                row, state = s.state_for(student, None, 1.0, ov)
            except KeyError as exc:
                raise HTTPException(404, str(exc)) from exc
            return row.code_module, state
    return s.learning_path(make, steps, _overrides(known))


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
    s.store.add_event(user.id, body.concept_id, body.kind, body.correct)
    return {"ok": True, "n_events": len(s.store.events(user.id))}


@app.post("/api/me/undo")
def undo(elpr_session: str | None = Cookie(None)) -> dict:
    user = _require(elpr_session)
    return {"ok": _service().store.undo_last_event(user.id)}


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
    return {"fields": profile_options()}


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
    cleaned = clean_profile(body.profile)
    _service().store.set_profile(user.id, cleaned)
    return {"profile": cleaned}


class StudiesBody(BaseModel):
    stage: str
    module: str | None = None


@app.put("/api/me/studies")
def update_studies(body: StudiesBody, elpr_session: str | None = Cookie(None)) -> dict:
    """Where the student is now and, for diploma and degree students, their course."""
    user = _require(elpr_session)
    s = _service()
    if user.role != "student":
        raise HTTPException(400, "only students have studies to set")
    stage, module = _studies(s, body.stage, body.module)
    s.store.set_studies(user.id, stage, module)
    return _me(replace(user, stage=stage, module=module))


# ---------------------------------------------------------------- admin
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
         "registered": True}
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


@app.get("/api/adviser/overview")
def overview(elpr_session: str | None = Cookie(None)) -> dict:
    _require_adviser(elpr_session)
    s = _service()
    return {"registered": s.store.registered_students(), "stats": s.store.stats()}


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
        return FileResponse(WEB / "index.html")
