"""SQLite store for accounts, sessions and the study activity of registered learners.

Separate from the analytical DuckDB database on purpose: that one holds immutable
research data, this one holds mutable application state.

SECURITY POSTURE — stated plainly because it matters.

Passwords are hashed with scrypt and a per-user random salt; the plaintext is never
stored and never logged. Session tokens are 32 bytes from `secrets` and are held in
httpOnly cookies. That much is done properly.

What this is NOT: there is no TLS on localhost, no rate limiting, no account recovery,
no email verification, no audit log. It is adequate for a research demonstration on a
single machine and is not adequate for real student data. Anyone registering should be
told not to reuse a password they use elsewhere.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "app.db"

# Registered learners get ids from this range so they can never collide with an OULAD
# student id (those are all well below 10 million).
SYNTHETIC_ID_BASE = 90_000_000

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT UNIQUE NOT NULL,
    display_name  TEXT NOT NULL,
    role          TEXT NOT NULL CHECK (role IN ('student', 'adviser')),
    salt          BLOB NOT NULL,
    password_hash BLOB NOT NULL,
    module        TEXT,
    student_id    INTEGER UNIQUE,
    created_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    token      TEXT PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(id),
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL
);

-- Study activity for registered learners. Each row is one interaction, in the same
-- shape the model consumes: a concept, a day, and optionally an outcome.
CREATE TABLE IF NOT EXISTS study_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id),
    concept_id INTEGER NOT NULL,
    day        REAL NOT NULL,
    kind       TEXT NOT NULL CHECK (kind IN ('study', 'assessment')),
    correct    INTEGER,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_user ON study_events(user_id, day);

-- Every recommendation served, for the adviser view and for the user study.
CREATE TABLE IF NOT EXISTS recommendation_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER,
    student_id  INTEGER NOT NULL,
    planner     TEXT NOT NULL,
    payload     TEXT NOT NULL,
    created_at  REAL NOT NULL
);
"""


@dataclass
class User:
    id: int
    username: str
    display_name: str
    role: str
    module: str | None
    student_id: int | None


def _hash(password: str, salt: bytes) -> bytes:
    # scrypt with the parameters recommended for interactive logins.
    return hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)


class AppStore:
    def __init__(self, path: Path = DB_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._connect().executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, check_same_thread=False)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        return con

    # -- accounts ---------------------------------------------------------------
    def register(
        self, username: str, password: str, display_name: str, role: str,
        module: str | None = None,
    ) -> User:
        username = username.strip().lower()
        if not username or len(username) < 3:
            raise ValueError("username must be at least 3 characters")
        if len(password) < 8:
            raise ValueError("password must be at least 8 characters")
        if role not in ("student", "adviser"):
            raise ValueError("role must be student or adviser")

        salt = secrets.token_bytes(16)
        con = self._connect()
        with con:
            existing = con.execute(
                "SELECT 1 FROM users WHERE username = ?", (username,)
            ).fetchone()
            if existing:
                raise ValueError("that username is taken")

            cursor = con.execute(
                "INSERT INTO users (username, display_name, role, salt, password_hash,"
                " module, created_at) VALUES (?,?,?,?,?,?,?)",
                (username, display_name or username, role, salt,
                 _hash(password, salt), module, time.time()),
            )
            user_id = cursor.lastrowid
            student_id = None
            if role == "student":
                student_id = SYNTHETIC_ID_BASE + user_id
                con.execute(
                    "UPDATE users SET student_id = ? WHERE id = ?", (student_id, user_id)
                )
        return User(user_id, username, display_name or username, role, module, student_id)

    def authenticate(self, username: str, password: str) -> User | None:
        row = self._connect().execute(
            "SELECT * FROM users WHERE username = ?", (username.strip().lower(),)
        ).fetchone()
        if row is None:
            # Hash anyway, so a missing username and a wrong password take the same
            # time and cannot be told apart by measuring the response.
            _hash(password, secrets.token_bytes(16))
            return None
        if not secrets.compare_digest(_hash(password, row["salt"]), row["password_hash"]):
            return None
        return User(row["id"], row["username"], row["display_name"], row["role"],
                    row["module"], row["student_id"])

    # -- sessions ---------------------------------------------------------------
    def create_session(self, user_id: int, hours: int = 12) -> str:
        token = secrets.token_urlsafe(32)
        now = time.time()
        con = self._connect()
        with con:
            con.execute(
                "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?,?,?,?)",
                (token, user_id, now, now + hours * 3600),
            )
        return token

    def user_for_session(self, token: str | None) -> User | None:
        if not token:
            return None
        row = self._connect().execute(
            "SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id "
            "WHERE s.token = ? AND s.expires_at > ?", (token, time.time())
        ).fetchone()
        if row is None:
            return None
        return User(row["id"], row["username"], row["display_name"], row["role"],
                    row["module"], row["student_id"])

    def end_session(self, token: str) -> None:
        con = self._connect()
        with con:
            con.execute("DELETE FROM sessions WHERE token = ?", (token,))

    # -- study activity ---------------------------------------------------------
    def add_event(
        self, user_id: int, concept_id: int, kind: str, correct: bool | None = None
    ) -> None:
        con = self._connect()
        with con:
            last = con.execute(
                "SELECT MAX(day) AS d FROM study_events WHERE user_id = ?", (user_id,)
            ).fetchone()["d"]
            day = (last or 0.0) + 3.0     # each recorded activity advances the clock
            con.execute(
                "INSERT INTO study_events (user_id, concept_id, day, kind, correct, created_at)"
                " VALUES (?,?,?,?,?,?)",
                (user_id, concept_id, day, kind,
                 None if correct is None else int(correct), time.time()),
            )

    def events(self, user_id: int) -> list[dict]:
        rows = self._connect().execute(
            "SELECT concept_id, day, kind, correct FROM study_events "
            "WHERE user_id = ? ORDER BY day, id", (user_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def undo_last_event(self, user_id: int) -> bool:
        con = self._connect()
        with con:
            row = con.execute(
                "SELECT id FROM study_events WHERE user_id = ? ORDER BY id DESC LIMIT 1",
                (user_id,),
            ).fetchone()
            if row is None:
                return False
            con.execute("DELETE FROM study_events WHERE id = ?", (row["id"],))
        return True

    # -- adviser view -----------------------------------------------------------
    def registered_students(self) -> list[dict]:
        rows = self._connect().execute(
            "SELECT u.id AS user_id, u.student_id, u.display_name, u.username, u.module,"
            "       u.created_at,"
            "       COUNT(e.id) AS n_events,"
            "       SUM(CASE WHEN e.kind = 'assessment' THEN 1 ELSE 0 END) AS n_assessments,"
            "       MAX(e.created_at) AS last_active"
            " FROM users u LEFT JOIN study_events e ON e.user_id = u.id"
            " WHERE u.role = 'student'"
            " GROUP BY u.id ORDER BY COALESCE(MAX(e.created_at), u.created_at) DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def user_by_student_id(self, student_id: int) -> User | None:
        row = self._connect().execute(
            "SELECT * FROM users WHERE student_id = ?", (student_id,)
        ).fetchone()
        if row is None:
            return None
        return User(row["id"], row["username"], row["display_name"], row["role"],
                    row["module"], row["student_id"])

    def log_recommendation(
        self, user_id: int | None, student_id: int, planner: str, payload: dict
    ) -> None:
        con = self._connect()
        with con:
            con.execute(
                "INSERT INTO recommendation_log (user_id, student_id, planner, payload, created_at)"
                " VALUES (?,?,?,?,?)",
                (user_id, student_id, planner, json.dumps(payload), time.time()),
            )

    def stats(self) -> dict:
        con = self._connect()
        q = lambda s: con.execute(s).fetchone()[0]
        return {
            "n_users": q("SELECT COUNT(*) FROM users"),
            "n_students": q("SELECT COUNT(*) FROM users WHERE role='student'"),
            "n_advisers": q("SELECT COUNT(*) FROM users WHERE role='adviser'"),
            "n_events": q("SELECT COUNT(*) FROM study_events"),
            "n_recommendations": q("SELECT COUNT(*) FROM recommendation_log"),
        }
