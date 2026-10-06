"""FastAPI service: authentication, recommendations, adviser views, results.

Also the "RESTful API for LMS integration" the paper lists as future work — delivered
here rather than promised.

Access rules, enforced server-side rather than by hiding buttons:
  * a student may read and write only their own record;
  * an adviser may read any learner, including registered students;
  * nobody may act on another account by passing a different id.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import Cookie, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse, PlainTextResponse,
                               RedirectResponse)
from starlette.exceptions import HTTPException as StarletteHTTPException

from elpr.modules import module_display_map, subject_area
from elpr.profile import clean as clean_profile, options as profile_options
from elpr import course_finder, exams, ics, selfcheck
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from elpr.db.app_store import INVITE_DAYS, User  # noqa: E402

if TYPE_CHECKING:                      # importing the service pulls in PyTorch, which
    from api.service import Service    # would delay the port opening by several seconds

# Deployment switches, read once at start. Both default to the safe setting.
#   ELPR_HTTPS=1     the site is served over HTTPS: session cookies are marked Secure and
#                    browsers are told to use HTTPS from then on (HSTS).
#   ELPR_API_DOCS=1  publish the generated API pages (/docs, /redoc, /openapi.json). Off
#                    by default, because they list every route to anyone who asks.
#   ELPR_TRUST_PROXY=1  behind a reverse proxy, take the client's address from
#                    X-Forwarded-For for rate limits. Never set it without a proxy, or a
#                    client could pick its own address.
#   ELPR_SITE_URL    the public address, such as https://example.in. Link previews,
#                    robots.txt and the sitemap use it; without it, the address the
#                    request arrived on, which behind a proxy is the internal one.
HTTPS = os.environ.get("ELPR_HTTPS") == "1"
SITE_URL = os.environ.get("ELPR_SITE_URL", "").rstrip("/")
API_DOCS = os.environ.get("ELPR_API_DOCS") == "1"
TRUST_PROXY = os.environ.get("ELPR_TRUST_PROXY") == "1"

app = FastAPI(title="GyanGraph", version="1.0.0",
              docs_url="/docs" if API_DOCS else None, redoc_url="/redoc" if API_DOCS else None,
              openapi_url="/openapi.json" if API_DOCS else None)
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


# Every response says what the browser may do with it. The page has inline script and
# style and loads nothing from anywhere else, so the policy can be this narrow.
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=()",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; font-src 'self'; connect-src 'self'; object-src 'none'; "
        "base-uri 'self'; form-action 'self'; frame-ancestors 'none'"),
}


class RateLimit:
    """At most `limit` requests per `window` seconds for each key, in memory.

    One process, one machine: enough for this site. Behind several workers each keeps
    its own count, so the real limit is that many times higher."""

    MAX_KEYS = 50_000

    def __init__(self, limit: int, window: float):
        self.limit, self.window = limit, window
        self.hits: dict[str, deque] = {}

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        q = self.hits.setdefault(key, deque())
        while q and now - q[0] >= self.window:
            q.popleft()
        if len(q) >= self.limit:
            return False
        q.append(now)
        if len(self.hits) > self.MAX_KEYS:
            # Never grow without end: forget keys with nothing recent, then the oldest half.
            for k in [k for k, v in self.hits.items() if not v]:
                del self.hits[k]
            for k in list(self.hits)[: max(0, len(self.hits) - self.MAX_KEYS // 2)]:
                del self.hits[k]
        return True


# What each limit protects: sign-up from account spam, the What-if from tying up the
# model, self-check marking from answer guessing by script, class codes from guessing
# one's way into an adviser's class, and everything else from plain flooding. Sign-in has its own per-username throttle in the store.
RATE_LIMITS = [
    ("POST", "/api/auth/register", RateLimit(5, 3600), "too many new accounts from here, try again in an hour"),
    ("GET", "/api/me/what-if", RateLimit(20, 60), "too many what-if checks in a minute, wait a moment"),
    ("POST", "/api/selfcheck", RateLimit(30, 60), "too many self-checks in a minute, wait a moment"),
    ("POST", "/api/me/advisers", RateLimit(10, 3600), "too many class codes tried, try again in an hour"),
    ("*", "/api/", RateLimit(300, 60), "too many requests, wait a moment"),
]


# The largest form on the site is a 1,000-character note. Nothing needs more than this,
# and a body is read into memory whole before it can be refused.
MAX_BODY = 64 * 1024


def client_key(request) -> str:
    if TRUST_PROXY:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            # The proxy appends the address it saw, so the last entry is the real one.
            # Regression: the first entry was used, and a visitor can send any first
            # entry they like, so a fresh made-up address each time escaped every limit.
            return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


@app.middleware("http")
async def guard(request, call_next):
    """Rate limits and a size limit before the work, security headers after it."""
    path, method, who = request.url.path, request.method, client_key(request)
    size = request.headers.get("content-length", "")
    for m, prefix, limiter, message in RATE_LIMITS:
        if (m == "*" or m == method) and path.startswith(prefix) and not limiter.allow(who):
            response = JSONResponse({"detail": message}, status_code=429)
            break
    else:
        if size.isdigit() and int(size) > MAX_BODY:
            response = JSONResponse({"detail": "that request is too large"}, status_code=413)
        else:
            response = await call_next(request)
    for name, value in SECURITY_HEADERS.items():
        response.headers.setdefault(name, value)
    # Answers from the API carry a student's own record, so no browser or proxy keeps one.
    if path.startswith("/api/"):
        response.headers.setdefault("Cache-Control", "no-store")
    if HTTPS:
        response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    return response


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
    if not user.approved:
        raise HTTPException(403, ADVISER_WAITING)
    return user


# Regression: anyone could sign up as an adviser and at once read every student's name,
# level, history and path, and the notes about them. An adviser now needs an invite from
# an admin, then the admin's approval, and even then sees only the students who joined
# their class.
ADVISER_WAITING = "an admin has not approved this adviser account yet"


def _may_view(s: Service, user: User, student: int) -> None:
    """Who may open a learner's record.

    A student: their own. An admin: anyone. An approved adviser: the anonymised dataset
    learners, and a registered student only once that student has joined their class."""
    if user.role == "student":
        if user.student_id != student:
            raise HTTPException(403, "you may only view your own record")
        return
    if not user.approved:
        raise HTTPException(403, ADVISER_WAITING)
    if user.role == "adviser" and s.is_registered(student) \
            and not s.store.adviser_can_see(user.id, student):
        raise HTTPException(403, "this student has not joined your class")


def _require_admin(token: str | None) -> User:
    user = _require(token)
    if user.role != "admin":
        raise HTTPException(403, "admin role required")
    return user


# How far through a dataset learner's course to look, as a share of it. Regression: "nan"
# got through and answered 500.
UPTO = Query(1.0, ge=0, le=1, allow_inf_nan=False)


def _overrides(value: str | None) -> tuple:
    """Week ids from a query string, keeping only ids the model has.

    Regression: an id past the last week answered 500, and "-1" read as the last week,
    because a negative index counts from the end."""
    if not value:
        return ()
    n = _service().graph.n_concepts
    ids = (v.strip() for v in value.split(",")[:MAX_OVERRIDES])
    return tuple(dict.fromkeys(int(v) for v in ids if v.isdigit() and int(v) < n))


MAX_OVERRIDES = 300


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
    invite: str | None = Field(default=None, max_length=40)       # advisers: from an admin
    class_code: str | None = Field(default=None, max_length=20)   # students: from an adviser


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


def _own_course(user: User) -> None:
    """The model's state needs a course. Regression: a class 10 or class 12 student was
    answered with the first course's state, an invented Psychology record."""
    if not user.module:
        raise HTTPException(409, "you are not on a course yet, so there is no weekly record")


def _course_student(s: Service, student: int) -> User:
    """A registered student's account, when they are on a course the model knows.

    A school student has no course, so there is no weekly path to show. The model's own
    state falls back to the first course, which put an invented Psychology path in front
    of an adviser who opened a class 12 student."""
    target = s.store.user_by_student_id(student)
    if target is None:
        raise HTTPException(404, "unknown learner")
    if not target.module:
        level = course_finder.STAGES.get(_stage(target) or "", "school")
        raise HTTPException(409, f"{target.display_name or 'This student'} is at {level} and "
                                 "not on a course yet, so there is no weekly path to show")
    return target


def _missing(exc: KeyError) -> str:
    """A KeyError's message without the quotes str() wraps round it."""
    return str(exc.args[0]) if exc.args else "not found"


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
    if body.role == "adviser" and not (body.invite or "").strip():
        raise HTTPException(400, "an adviser account needs an invite code from an admin")
    try:
        user = s.store.register(
            body.username, body.password, body.display_name, body.role, module, stage, stream,
            invite=body.invite if body.role == "adviser" else None,
            class_code=((body.class_code or "").strip() or None) if body.role == "student" else None,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    token = s.store.create_session(user.id)
    s.store.log(user.id, "signed up", f"{user.role}, {stage or 'no level'}"
                + ("" if user.approved else ", waiting for an admin"))
    response.set_cookie(COOKIE, token, httponly=True, samesite="lax", max_age=12 * 3600, secure=HTTPS)
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
    response.set_cookie(COOKIE, token, httponly=True, samesite="lax", max_age=12 * 3600, secure=HTTPS)
    return _me(user)


@app.post("/api/auth/logout")
def logout(response: Response, elpr_session: str | None = Cookie(None)) -> dict:
    if elpr_session:
        _service().store.end_session(elpr_session)
    response.delete_cookie(COOKIE, httponly=True, samesite="lax", secure=HTTPS)
    return {"ok": True}


def _me(user: User) -> dict:
    return {
        "username": user.username, "display_name": user.display_name,
        "role": user.role, "module": user.module, "student_id": user.student_id,
        "stage": _stage(user), "stream": user.stream, "approved": user.approved,
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
def my_state(known: str | None = None, full: bool = False,
             elpr_session: str | None = Cookie(None)) -> dict:
    user = _require(elpr_session)
    s = _service()
    if user.role != "student":
        raise HTTPException(400, "advisers have no learning record of their own")
    _own_course(user)

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
        # The record of study lists every finished week, so it asks for all of them.
        "history": events if full else events[-25:],
    }


@app.get("/api/me/recommend")
def my_recommendations(
    k: int = Query(3, ge=1, le=10), planner: str = "greedy",
    known: str | None = None, elpr_session: str | None = Cookie(None),
) -> dict:
    user = _require(elpr_session)
    s = _service()
    if user.role != "student":
        raise HTTPException(400, "advisers have no learning record of their own")
    _own_course(user)

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


@app.get("/api/me/what-if")
def my_what_if(weeks: str, elpr_session: str | None = Cookie(None)) -> dict:
    """A preview: the model run again as if the student had passed these weeks.

    Read-only, so a GET. Nothing is stored and nothing is logged, because nothing
    happened: the privacy page promises the log holds only what a student did."""
    user = _require(elpr_session)
    s = _service()
    if user.role != "student":
        raise HTTPException(400, "advisers have no learning record of their own")
    if not user.module:
        raise HTTPException(409, "a what-if needs a course, and you are not on one yet")
    try:
        return s.what_if(user, _overrides(weeks))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/students/{student}/path")
def learner_path(student: int, steps: int = Query(5, ge=1, le=8), known: str | None = None,
                 elpr_session: str | None = Cookie(None)) -> dict:
    user = _require(elpr_session)
    s = _service()
    _may_view(s, user, student)
    if s.is_registered(student):
        target = _course_student(s, student)
        return s.learning_path(lambda ov: s.registered_state(target, ov), steps,
                               _overrides(known))
    try:
        return s.dataset_path(student, steps, _overrides(known))
    except KeyError as exc:
        raise HTTPException(404, _missing(exc)) from exc


@app.get("/api/me/progress")
def my_progress(elpr_session: str | None = Cookie(None)) -> dict:
    """Streak, activity calendar, badges and weekly summary from recorded study events."""
    user = _require(elpr_session)
    if user.role != "student":
        raise HTTPException(400, "only students have a progress record")
    _own_course(user)
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
    if body.correct is not None and body.kind != "assessment":
        raise HTTPException(400, "reading a week has no pass or fail: record a quiz instead")
    # A pass is evidence the model learns from, so it needs a mark from a real quiz.
    # "I found it hard" stays a plain claim: it only ever holds a week in place.
    if body.kind == "assessment" and body.correct is True and body.score is None:
        raise HTTPException(400, "a pass needs your quiz mark: enter it out of 100")
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


@app.get("/api/exams")
def exam_list(stream: str | None = None) -> dict:
    """Entrance exams a student can aim at, narrowed by class 12 stream.

    Public, like the rest of the Course Finder: a visitor deciding on a stream needs to
    see which exams each one opens."""
    out = exams.summary(stream)
    for e in out["exams"]:
        e["n_checks"] = sum(selfcheck.has_check(name, u)
                            for name, units in exams.sections_for(e["key"], stream)
                            for u in units)
    out["check_note"] = selfcheck.COVERAGE_NOTE
    return out


@app.get("/api/exams/{key}")
def exam_detail(key: str, stream: str | None = None) -> dict:
    """One exam's published syllabus outline, section by section."""
    try:
        exam = exams.EXAMS[key]
        sections = exams.sections_for(key, stream)
    except KeyError:
        raise HTTPException(404, "no such exam")
    return {
        "key": exam.key, "name": exam.name, "full_name": exam.full_name,
        "body": exam.body, "leads_to": exam.leads_to, "source": exam.source,
        "note": exam.note, "streams": list(exam.streams), "stream": stream,
        "sections": [{"name": name,
                      "units": [{"name": u, "links": course_finder.study_links(u, exams.SCHOOLING),
                                 "check": selfcheck.has_check(name, u)}
                                for u in units]} for name, units in sections],
    }


@app.get("/api/exams/{key}/plan")
def exam_plan(key: str, weeks: int = Query(24, ge=1, le=260),
              start: str | None = None, stream: str | None = None) -> dict:
    """One exam's syllabus spread over the weeks left before it."""
    try:
        plan = exams.exam_plan(key, weeks, start, stream)
    except KeyError:
        raise HTTPException(404, "no such exam")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    for week in plan["weeks"]:
        for unit in week["units"]:
            unit["check"] = selfcheck.has_check(unit["section"], unit["name"])
    return plan


@app.get("/api/exams/{key}/plan.ics")
def exam_plan_ics(key: str, weeks: int = Query(24, ge=1, le=260),
                  start: str | None = None, stream: str | None = None) -> Response:
    """The same revision plan as a calendar file: one all-day event per week."""
    from datetime import date as _date

    try:
        plan = exams.exam_plan(key, weeks, start or _date.today().isoformat(), stream)
    except KeyError:
        raise HTTPException(404, "no such exam")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    events = [{
        "uid": f"{plan['key']}-w{week['week']}@learning-path",
        "start": _date.fromisoformat(week["starts"]),
        "days": 7,
        "summary": f"{plan['name']} revision · Week {week['week']}",
        "description": "Revise this week: "
                       + ", ".join(u["name"] for u in week["units"]),
    } for week in plan["weeks"]]
    body = ics.calendar(f"{plan['name']} revision plan", events)
    return Response(content=body, media_type="text/calendar; charset=utf-8",
                    headers={"Content-Disposition":
                             f'attachment; filename="{plan["key"]}-revision-plan.ics"'})


class SelfCheckAnswers(BaseModel):
    exam: str
    section: str
    unit: str
    answers: dict[str, int]


@app.get("/api/selfcheck")
def selfcheck_questions(exam: str, section: str, unit: str) -> dict:
    """Questions for one exam unit, without their answers. Public: anyone revising can
    check themselves, and only a signed-in student's result is kept."""
    try:
        return selfcheck.questions_for(exam, section, unit)
    except KeyError:
        raise HTTPException(404, "there is no self-check for that unit")


@app.post("/api/selfcheck")
def selfcheck_grade(body: SelfCheckAnswers, elpr_session: str | None = Cookie(None)) -> dict:
    """Mark the answers on the server, so the right answers never reach the page before
    the student has committed to theirs."""
    try:
        out = selfcheck.grade(body.exam, body.section, body.unit, body.answers)
    except KeyError:
        raise HTTPException(404, "there is no self-check for that unit")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    user = _current(elpr_session)
    out["saved"] = bool(user and user.role == "student")
    if out["saved"]:
        store = _service().store
        store.add_self_check(user.id, body.exam, body.section, body.unit,
                             out["n"], out["n_correct"], out["score"])
        store.log(user.id, "self-check",
                  f"{body.exam}, {body.unit}: {out['n_correct']} of {out['n']}")
    return out


@app.get("/api/me/selfchecks")
def my_selfchecks(elpr_session: str | None = Cookie(None)) -> dict:
    """The latest self-check result for each unit the student has checked."""
    user = _require(elpr_session)
    return {"checks": _service().store.self_checks(user.id),
            "pass_mark": selfcheck.PASS_MARK, "note": selfcheck.COVERAGE_NOTE}


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
        begin = course_finder._monday_of(start or _date.today().isoformat())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

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


@app.post("/api/admin/accounts/{username}/approve")
def approve_adviser(username: str, elpr_session: str | None = Cookie(None)) -> dict:
    """Let a waiting adviser account see students. Admins only."""
    admin = _require_admin(elpr_session)
    store = _service().store
    if not store.approve_adviser(username):
        raise HTTPException(404, "no adviser account with that username")
    target = store.user_by_username(username)
    store.log(target.id, "approved as an adviser", f"by {admin.username}")
    store.log(admin.id, "approved an adviser", target.username)
    return {"approved": target.username}


class InviteBody(BaseModel):
    note: str = Field(min_length=1, max_length=80)


@app.post("/api/admin/invites")
def create_invite(body: InviteBody, elpr_session: str | None = Cookie(None)) -> dict:
    """A one-time code for one named person to sign up as an adviser. Shown once."""
    admin = _require_admin(elpr_session)
    store = _service().store
    try:
        invite = store.create_invite(admin.id, body.note)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    store.log(admin.id, "made an adviser invite", invite["note"])
    return invite


@app.get("/api/admin/invites")
def list_invites(elpr_session: str | None = Cookie(None)) -> dict:
    _require_admin(elpr_session)
    return {"invites": _service().store.invites(), "days": INVITE_DAYS}


@app.delete("/api/admin/invites/{invite_id}")
def revoke_invite(invite_id: int, elpr_session: str | None = Cookie(None)) -> dict:
    admin = _require_admin(elpr_session)
    store = _service().store
    if not store.revoke_invite(invite_id):
        raise HTTPException(404, "no unused invite with that number")
    store.log(admin.id, "withdrew an adviser invite", str(invite_id))
    return {"revoked": invite_id}


@app.get("/api/adviser/class")
def my_class(elpr_session: str | None = Cookie(None)) -> dict:
    """The code an adviser gives students so they can join, and so be seen."""
    user = _require_adviser(elpr_session)
    if user.role != "adviser":
        raise HTTPException(400, "admins see every student and have no class")
    return {"code": _service().store.class_code(user.id)}


@app.post("/api/adviser/class/new")
def new_class_code(elpr_session: str | None = Cookie(None)) -> dict:
    """Replace a code that went further than it should. Students already in stay."""
    user = _require_adviser(elpr_session)
    if user.role != "adviser":
        raise HTTPException(400, "admins see every student and have no class")
    store = _service().store
    code = store.new_class_code(user.id)
    store.log(user.id, "made a new class code")
    return {"code": code}


class JoinBody(BaseModel):
    code: str = Field(min_length=1, max_length=20)


def _student(token: str | None) -> User:
    user = _require(token)
    if user.role != "student":
        raise HTTPException(400, "only students join an adviser's class")
    return user


@app.get("/api/me/advisers")
def my_advisers(elpr_session: str | None = Cookie(None)) -> dict:
    """The advisers who can see this student's record."""
    user = _student(elpr_session)
    return {"advisers": _service().store.advisers_of(user.id)}


@app.post("/api/me/advisers")
def join_class(body: JoinBody, elpr_session: str | None = Cookie(None)) -> dict:
    user = _student(elpr_session)
    store = _service().store
    try:
        adviser = store.join_class(user.id, body.code)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    store.log(user.id, "joined a class", adviser.display_name)
    store.log(adviser.id, "a student joined the class", user.display_name)
    return {"joined": adviser.display_name, "advisers": store.advisers_of(user.id)}


@app.delete("/api/me/advisers/{adviser_id}")
def leave_class(adviser_id: int, elpr_session: str | None = Cookie(None)) -> dict:
    """Stop an adviser seeing this student's record, and the notes written about it."""
    user = _student(elpr_session)
    store = _service().store
    if not store.leave_class(user.id, adviser_id):
        raise HTTPException(404, "you are not in that adviser's class")
    store.log(user.id, "left a class", str(adviser_id))
    return {"advisers": store.advisers_of(user.id)}


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
    user = _require_adviser(elpr_session)
    s = _service()
    registered = [
        {"id_student": r["student_id"], "module": r["module"] or "n/a", "stage": r["stage"],
         "presentation": "registered", "n_events": r["n_events"] or 0,
         "n_assessments": r["n_assessments"] or 0,
         "outcome": "in progress", "display_name": r["display_name"],
         "registered": True, "quiet": r["quiet"], "quiet_days": r["quiet_days"],
         "never_started": r["never_started"]}
        for r in s.store.registered_students(adviser_id=_class_of(user))
        if not q or str(r["student_id"]).startswith(q)
        or q.lower() in (r["display_name"] or "").lower()
    ]
    dataset = [{**r, "registered": False} for r in s.search(q, limit)]
    return {"registered": registered, "dataset": dataset}


@app.get("/api/students/{student}/mastery")
def mastery(student: int, module: str | None = None, upto: float = UPTO,
            known: str | None = None, elpr_session: str | None = Cookie(None)) -> dict:
    user = _require(elpr_session)
    s = _service()
    _may_view(s, user, student)

    if s.is_registered(student):
        target = _course_student(s, student)
        mod, state = s.registered_state(target, _overrides(known))
        scope = state.scope()
        return {
            "student": student, "module": mod, "registered": True,
            "display_name": target.display_name,
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
        raise HTTPException(404, _missing(exc)) from exc


@app.get("/api/recommend")
def recommend(student: int, k: int = Query(3, ge=1, le=10), planner: str = "greedy",
              module: str | None = None, upto: float = UPTO, known: str | None = None,
              elpr_session: str | None = Cookie(None)) -> dict:
    user = _require(elpr_session)
    s = _service()
    _may_view(s, user, student)

    start = time.perf_counter()
    if s.is_registered(student):
        target = _course_student(s, student)
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
            raise HTTPException(404, _missing(exc)) from exc

    payload["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 1)
    s.store.log_recommendation(user.id, student, planner, payload)
    return payload


class NoteBody(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


@app.get("/api/students/{student}/notes")
def learner_notes(student: int, elpr_session: str | None = Cookie(None)) -> dict:
    """Adviser notes on one learner. Advisers and admins only, never the student."""
    user = _require_adviser(elpr_session)
    _may_view(_service(), user, student)
    return {"student": student, "notes": _service().store.notes(student)}


@app.post("/api/students/{student}/notes")
def add_learner_note(student: int, body: NoteBody,
                     elpr_session: str | None = Cookie(None)) -> dict:
    user = _require_adviser(elpr_session)
    s = _service()
    _may_view(s, user, student)
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
    _may_view(s, user, student)
    removed = s.store.delete_note(note_id, user.id)
    if not removed and user.role == "admin":
        removed = s.store.delete_note_any(note_id)
    if not removed:
        raise HTTPException(404, "no note of yours with that id")
    return {"deleted": note_id, "notes": s.store.notes(student)}


@app.get("/api/adviser/overview")
def overview(elpr_session: str | None = Cookie(None)) -> dict:
    user = _require_adviser(elpr_session)
    s = _service()
    registered = s.store.registered_students(adviser_id=_class_of(user))
    # Counts over the adviser's own students: the site's totals are the admin's business.
    stats = {"n_students": len(registered),
             "n_events": sum(r["n_events"] or 0 for r in registered)}
    return {"registered": registered, "stats": stats,
            "quiet_after_days": s.store.QUIET_AFTER_DAYS}


def _class_of(user: User) -> int | None:
    """Whose class to list: an adviser's own, or everyone's for an admin."""
    return None if user.role == "admin" else user.id


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


def site_base(request: Request) -> str:
    return SITE_URL or str(request.base_url).rstrip("/")


NOT_FOUND_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Page not found · GyanGraph</title>
<meta name="robots" content="noindex"><link rel="icon" href="/favicon.ico">
<style>:root{color-scheme:light dark;--bg:#F3F9FE;--ink:#0F243D;--soft:#536B85;--accent:#0369A1}
@media (prefers-color-scheme: dark){:root{--bg:#0A1526;--ink:#F0F6FC;--soft:#C7E3F8;--accent:#38BDF8}}
body{margin:0;min-height:100vh;display:grid;place-items:center;background:var(--bg);color:var(--ink);
font:16px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif;padding:0 16px}
main{max-width:460px}h1{font-size:28px;margin:0 0 8px}p{color:var(--soft);margin:0 0 20px}
a{color:var(--accent);font-weight:600}</style></head>
<body><main><h1>That page is not here</h1><p>The address may be mistyped, or the page has moved.
Everything on this site starts from the home page.</p><a href="/">Go to the home page</a></main></body></html>"""


@app.exception_handler(StarletteHTTPException)
async def not_found(request: Request, exc: StarletteHTTPException):
    """A person who mistypes an address gets a page, not a line of JSON. The API keeps
    answering in JSON, because the page reads its errors from there."""
    if exc.status_code == 404 and not request.url.path.startswith("/api/"):
        return HTMLResponse(NOT_FOUND_PAGE, status_code=404)
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code,
                        headers=getattr(exc, "headers", None))


if WEB.exists():
    @app.get("/static/index.html", include_in_schema=False)
    def static_page() -> RedirectResponse:
        # Regression: the folder is served as it is, so the page had a second address
        # where its link-preview tags still read {{BASE_URL}}. Routes before the mount win.
        return RedirectResponse("/", status_code=301)

    app.mount("/static", StaticFiles(directory=WEB), name="static")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> FileResponse:
        # Browsers ask for this address whatever the page says, and it used to 404.
        return FileResponse(WEB / "icons" / "favicon.ico", media_type="image/x-icon")

    @app.get("/robots.txt", include_in_schema=False)
    def robots(request: Request) -> PlainTextResponse:
        base = site_base(request)
        return PlainTextResponse(f"User-agent: *\nAllow: /\nDisallow: /api/\nSitemap: {base}/sitemap.xml\n")

    @app.get("/sitemap.xml", include_in_schema=False)
    def sitemap(request: Request) -> Response:
        # The site is one page; its sections live after the # and search engines read
        # them from the page itself, so the sitemap is that one address.
        base = site_base(request)
        xml = ('<?xml version="1.0" encoding="UTF-8"?>\n'
               '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
               f"<url><loc>{base}/</loc></url></urlset>\n")
        return Response(xml, media_type="application/xml")

    _page_cache: dict = {}

    @app.get("/")
    def index(request: Request) -> HTMLResponse:
        # The page is the whole application, so a cached copy means a student keeps
        # seeing yesterday's version after an update. Never cache it.
        # Link previews (WhatsApp, Telegram) need full addresses for the page and its
        # picture, so the site's own address is filled in as the page is sent.
        path = WEB / "index.html"
        stamp = path.stat().st_mtime_ns
        if _page_cache.get("stamp") != stamp:
            _page_cache.update(stamp=stamp, text=path.read_text())
        base = site_base(request)
        return HTMLResponse(_page_cache["text"].replace("{{BASE_URL}}", base),
                            headers={"Cache-Control": "no-store, must-revalidate"})

    @app.get("/api/version")
    def page_version(response: Response) -> dict:
        """When the page file last changed. Moving between pages never reloads the
        file, so a tab left open kept running an old copy after an update, fixed bugs
        included. The page asks this as it moves and loads the new copy if it changed."""
        response.headers["Cache-Control"] = "no-store"
        stat = (WEB / "index.html").stat()
        return {"version": f"{stat.st_mtime_ns}-{stat.st_size}"}
