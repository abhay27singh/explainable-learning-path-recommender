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
from pathlib import Path

from fastapi import Cookie, FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse
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
    user = _require(token)
    if user.role != "adviser":
        raise HTTPException(403, "adviser role required")
    return user


def _overrides(value: str | None) -> tuple:
    if not value:
        return ()
    return tuple(int(v) for v in value.split(",") if v.strip().lstrip("-").isdigit())


# ---------------------------------------------------------------- auth
class Credentials(BaseModel):
    username: str = Field(min_length=3, max_length=40)
    password: str = Field(min_length=8, max_length=200)


class Registration(Credentials):
    display_name: str = Field(default="", max_length=80)
    role: str = Field(default="student")
    module: str | None = None


@app.post("/api/auth/register")
def register(body: Registration, response: Response) -> dict:
    s = _service()
    if body.module and body.module not in s.modules:
        raise HTTPException(400, f"unknown module {body.module}")
    try:
        user = s.store.register(
            body.username, body.password, body.display_name, body.role, body.module
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


# ---------------------------------------------------------------- adviser
@app.get("/api/students")
def students(q: str = "", limit: int = Query(40, le=200),
             elpr_session: str | None = Cookie(None)) -> dict:
    _require_adviser(elpr_session)
    s = _service()
    registered = [
        {"id_student": r["student_id"], "module": r["module"] or "—",
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
    if user.role != "adviser" and user.student_id != student:
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
    if user.role != "adviser" and user.student_id != student:
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
def metrics() -> dict:
    """Study results, read live from results/ so the site cannot drift from the paper."""
    import json

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
