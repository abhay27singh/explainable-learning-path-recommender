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

# Minimum password length, for every role. Set low deliberately: this is a classroom
# demonstration that people sign into once, in front of an audience, and a long password
# is friction with no benefit here. It is NOT a defensible value for a deployed system.
# Raise it before this is ever served beyond localhost. See docs in the privacy page.
MIN_PASSWORD = 4

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT UNIQUE NOT NULL,
    display_name  TEXT NOT NULL,
    role          TEXT NOT NULL CHECK (role IN ('student', 'adviser', 'admin')),
    salt          BLOB NOT NULL,
    password_hash BLOB NOT NULL,
    module        TEXT,
    student_id    INTEGER UNIQUE,
    created_at    REAL NOT NULL
);

-- Background answers a student chose to give. Stored as JSON of OULAD category values.
CREATE TABLE IF NOT EXISTS learner_profiles (
    user_id    INTEGER PRIMARY KEY REFERENCES users(id),
    profile    TEXT NOT NULL,
    updated_at REAL NOT NULL
);

-- Courses a student saved from the Course Finder, newest first when read back.
CREATE TABLE IF NOT EXISTS saved_courses (
    user_id    INTEGER NOT NULL REFERENCES users(id),
    course_key TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (user_id, course_key)
);

-- What happened on each account, so an admin can answer "it is not working for me".
-- Plain facts only: never a password, never the contents of an answer.
CREATE TABLE IF NOT EXISTS user_log (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER REFERENCES users(id),
    at      REAL NOT NULL,
    kind    TEXT NOT NULL,
    detail  TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_log_user ON user_log(user_id, id DESC);

-- An adviser's own notes on a learner, keyed by learner id so they work for both
-- registered students and dataset learners.
CREATE TABLE IF NOT EXISTS learner_notes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL,
    author_id  INTEGER REFERENCES users(id),
    at         REAL NOT NULL,
    text       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_notes_student ON learner_notes(student_id, id DESC);

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
    stage: str | None = None      # class_10, class_12, diploma, ug or pg
    stream: str | None = None     # class 12 stream, for class_12 students


def _user(row) -> User:
    return User(row["id"], row["username"], row["display_name"], row["role"],
                row["module"], row["student_id"], row["stage"], row["stream"])


def _hash(password: str, salt: bytes) -> bytes:
    # scrypt with the parameters recommended for interactive logins.
    return hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)


class AppStore:
    def __init__(self, path: Path = DB_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._connect().executescript(SCHEMA)
        self._migrate_roles()
        self._migrate_stage()

    def _migrate_stage(self) -> None:
        """Add the study-level and stream columns to databases created before them."""
        con = self._connect()
        columns = {r["name"] for r in con.execute("PRAGMA table_info(users)")}
        with con:
            if "stage" not in columns:
                con.execute("ALTER TABLE users ADD COLUMN stage TEXT")
            if "stream" not in columns:
                con.execute("ALTER TABLE users ADD COLUMN stream TEXT")

    def _migrate_roles(self) -> None:
        """Widen the role CHECK constraint on databases created before admin existed.

        SQLite cannot alter a CHECK in place, so the table is rebuilt. Done only when
        the stored DDL lacks 'admin', which makes this a no-op on every later start.
        """
        con = self._connect()
        row = con.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='users'"
        ).fetchone()
        if row is None or "admin" in row["sql"]:
            return
        with con:
            con.execute("PRAGMA foreign_keys = OFF")
            con.execute(
                """CREATE TABLE users_new (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    username      TEXT UNIQUE NOT NULL,
                    display_name  TEXT NOT NULL,
                    role          TEXT NOT NULL CHECK (role IN ('student','adviser','admin')),
                    salt          BLOB NOT NULL,
                    password_hash BLOB NOT NULL,
                    module        TEXT,
                    student_id    INTEGER UNIQUE,
                    created_at    REAL NOT NULL)"""
            )
            con.execute("INSERT INTO users_new SELECT * FROM users")
            con.execute("DROP TABLE users")
            con.execute("ALTER TABLE users_new RENAME TO users")
            con.execute("PRAGMA foreign_keys = ON")

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, check_same_thread=False)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        return con

    # -- accounts ---------------------------------------------------------------
    def register(
        self, username: str, password: str, display_name: str, role: str,
        module: str | None = None, stage: str | None = None, stream: str | None = None,
    ) -> User:
        username = username.strip().lower()
        if not username or len(username) < 3:
            raise ValueError("username must be at least 3 characters")
        if len(password) < MIN_PASSWORD:
            raise ValueError(f"password must be at least {MIN_PASSWORD} characters")
        if role not in ("student", "adviser"):
            # 'admin' is intentionally absent: an admin can only be created from the
            # command line by someone with filesystem access to the database.
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
                " module, stage, stream, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (username, display_name or username, role, salt,
                 _hash(password, salt), module, stage, stream, time.time()),
            )
            user_id = cursor.lastrowid
            student_id = None
            if role == "student":
                student_id = SYNTHETIC_ID_BASE + user_id
                con.execute(
                    "UPDATE users SET student_id = ? WHERE id = ?", (student_id, user_id)
                )
        return User(user_id, username, display_name or username, role, module, student_id,
                    stage, stream)

    def set_studies(self, user_id: int, stage: str, module: str | None,
                    stream: str | None = None) -> None:
        """Where a student is now: their level, their course if they are on one, and
        their class 12 stream if they are in class 12."""
        con = self._connect()
        with con:
            con.execute("UPDATE users SET stage = ?, module = ?, stream = ? WHERE id = ?",
                        (stage, module, stream, user_id))

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
        return _user(row)

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
        return _user(row)

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
            "SELECT concept_id, day, kind, correct, created_at FROM study_events "
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
    # A student is "quiet" after this many days with nothing recorded. Long enough that
    # an ordinary busy week does not flag someone, short enough to act on.
    QUIET_AFTER_DAYS = 7

    def registered_students(self, now: float | None = None) -> list[dict]:
        """Registered students with their activity, and how long each has been quiet.

        `quiet_days` counts from their last recorded activity, or from the day they
        signed up if they have never recorded anything, which is the case worth chasing."""
        rows = self._connect().execute(
            "SELECT u.id AS user_id, u.student_id, u.display_name, u.username, u.module,"
            "       u.stage, u.created_at,"
            "       COUNT(e.id) AS n_events,"
            "       SUM(CASE WHEN e.kind = 'assessment' THEN 1 ELSE 0 END) AS n_assessments,"
            "       MAX(e.created_at) AS last_active"
            " FROM users u LEFT JOIN study_events e ON e.user_id = u.id"
            " WHERE u.role = 'student'"
            " GROUP BY u.id ORDER BY COALESCE(MAX(e.created_at), u.created_at) DESC"
        ).fetchall()
        now = time.time() if now is None else now
        out = []
        for row in rows:
            record = dict(row)
            since = record["last_active"] or record["created_at"]
            record["quiet_days"] = max(0, int((now - since) // 86400))
            record["quiet"] = record["quiet_days"] >= self.QUIET_AFTER_DAYS
            record["never_started"] = not record["n_events"]
            out.append(record)
        return out

    def user_by_username(self, username: str) -> User | None:
        row = self._connect().execute(
            "SELECT * FROM users WHERE username = ?", (username.strip().lower(),)
        ).fetchone()
        return None if row is None else _user(row)

    def user_by_student_id(self, student_id: int) -> User | None:
        row = self._connect().execute(
            "SELECT * FROM users WHERE student_id = ?", (student_id,)
        ).fetchone()
        if row is None:
            return None
        return _user(row)

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

    # -- learner background -------------------------------------------------------
    def get_profile(self, user_id: int) -> dict:
        row = self._connect().execute(
            "SELECT profile FROM learner_profiles WHERE user_id = ?", (user_id,)
        ).fetchone()
        return json.loads(row["profile"]) if row else {}

    def set_profile(self, user_id: int, profile: dict) -> None:
        con = self._connect()
        with con:
            con.execute(
                "INSERT INTO learner_profiles (user_id, profile, updated_at) VALUES (?,?,?)"
                " ON CONFLICT(user_id) DO UPDATE SET profile = excluded.profile,"
                " updated_at = excluded.updated_at",
                (user_id, json.dumps(profile, sort_keys=True), time.time()),
            )

    # -- adviser notes ----------------------------------------------------------
    NOTE_MAX = 1000

    def add_note(self, student_id: int, author_id: int, text: str) -> int:
        """Record one adviser note about a learner. Returns its id."""
        text = text.strip()
        if not text:
            raise ValueError("a note cannot be empty")
        con = self._connect()
        with con:
            cur = con.execute(
                "INSERT INTO learner_notes (student_id, author_id, at, text) VALUES (?,?,?,?)",
                (int(student_id), author_id, time.time(), text[:self.NOTE_MAX]),
            )
        return cur.lastrowid

    def notes(self, student_id: int) -> list[dict]:
        """Every note on one learner, newest first, with who wrote it."""
        rows = self._connect().execute(
            "SELECT n.id, n.at, n.text, u.display_name AS author"
            " FROM learner_notes n LEFT JOIN users u ON u.id = n.author_id"
            " WHERE n.student_id = ? ORDER BY n.id DESC", (int(student_id),)
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_note(self, note_id: int, author_id: int) -> bool:
        """Remove a note. An adviser may only remove their own."""
        con = self._connect()
        with con:
            cur = con.execute("DELETE FROM learner_notes WHERE id = ? AND author_id = ?",
                              (int(note_id), author_id))
        return bool(cur.rowcount)

    def delete_note_any(self, note_id: int) -> bool:
        """Remove any note, whoever wrote it. For admins only."""
        con = self._connect()
        with con:
            cur = con.execute("DELETE FROM learner_notes WHERE id = ?", (int(note_id),))
        return bool(cur.rowcount)

    # -- support log ------------------------------------------------------------
    LOG_KEEP = 200

    def log(self, user_id: int | None, kind: str, detail: str = "") -> None:
        """Record one thing that happened on an account, keeping the last 200."""
        con = self._connect()
        with con:
            con.execute("INSERT INTO user_log (user_id, at, kind, detail) VALUES (?,?,?,?)",
                        (user_id, time.time(), kind, detail[:300]))
            if user_id is not None:
                con.execute(
                    "DELETE FROM user_log WHERE user_id = ? AND id NOT IN ("
                    " SELECT id FROM user_log WHERE user_id = ? ORDER BY id DESC LIMIT ?)",
                    (user_id, user_id, self.LOG_KEEP))

    def logs(self, username: str, limit: int = 100) -> list[dict]:
        """The most recent entries for one account, newest first."""
        rows = self._connect().execute(
            "SELECT l.at, l.kind, l.detail FROM user_log l JOIN users u ON u.id = l.user_id"
            " WHERE u.username = ? ORDER BY l.id DESC LIMIT ?",
            (username.strip().lower(), int(limit)),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- saved courses ----------------------------------------------------------
    def save_course(self, user_id: int, course_key: str) -> None:
        """Add a course to the student's shortlist. Saving twice changes nothing."""
        con = self._connect()
        with con:
            con.execute(
                "INSERT INTO saved_courses (user_id, course_key, created_at) VALUES (?,?,?)"
                " ON CONFLICT(user_id, course_key) DO NOTHING",
                (user_id, course_key, time.time()),
            )

    def remove_course(self, user_id: int, course_key: str) -> bool:
        con = self._connect()
        with con:
            cur = con.execute(
                "DELETE FROM saved_courses WHERE user_id = ? AND course_key = ?",
                (user_id, course_key),
            )
        return bool(cur.rowcount)

    def saved_courses(self, user_id: int) -> list[str]:
        """Course keys the student saved, newest first."""
        rows = self._connect().execute(
            "SELECT course_key FROM saved_courses WHERE user_id = ?"
            " ORDER BY created_at DESC, course_key", (user_id,)
        ).fetchall()
        return [r["course_key"] for r in rows]

    # -- administration ---------------------------------------------------------
    def create_admin(self, username: str, password: str, display_name: str = "") -> User:
        """Create an administrator. Callable only from the command line, never the API."""
        username = username.strip().lower()
        if not username or len(username) < 3:
            raise ValueError("username must be at least 3 characters")
        if len(password) < MIN_PASSWORD:
            raise ValueError(f"password must be at least {MIN_PASSWORD} characters")

        salt = secrets.token_bytes(16)
        con = self._connect()
        with con:
            if con.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone():
                raise ValueError("that username is taken")
            cursor = con.execute(
                "INSERT INTO users (username, display_name, role, salt, password_hash,"
                " module, created_at) VALUES (?,?,?,?,?,?,?)",
                (username, display_name or username, "admin", salt,
                 _hash(password, salt), None, time.time()),
            )
        return User(cursor.lastrowid, username, display_name or username, "admin", None, None)

    def set_password(self, username: str, password: str) -> bool:
        """Reset a password from the command line. Returns False if no such user."""
        if len(password) < MIN_PASSWORD:
            raise ValueError(f"password must be at least {MIN_PASSWORD} characters")
        salt = secrets.token_bytes(16)
        con = self._connect()
        with con:
            cur = con.execute(
                "UPDATE users SET salt = ?, password_hash = ? WHERE username = ?",
                (salt, _hash(password, salt), username.strip().lower()),
            )
            if cur.rowcount:
                con.execute(
                    "DELETE FROM sessions WHERE user_id ="
                    " (SELECT id FROM users WHERE username = ?)",
                    (username.strip().lower(),),
                )
        return bool(cur.rowcount)

    def list_accounts(self) -> list[dict]:
        """Every registered account with its activity, for the admin view."""
        con = self._connect()
        rows = con.execute(
            "SELECT u.id, u.username, u.display_name, u.role, u.module, u.stage, u.student_id,"
            "       u.created_at,"
            "       (SELECT COUNT(*) FROM study_events e WHERE e.user_id = u.id) AS n_events,"
            "       (SELECT MAX(e.created_at) FROM study_events e WHERE e.user_id = u.id)"
            "         AS last_active"
            " FROM users u ORDER BY u.created_at"
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_account(self, username: str) -> bool:
        """Remove an account and everything belonging to it. Returns False if absent."""
        con = self._connect()
        with con:
            row = con.execute(
                "SELECT id FROM users WHERE username = ?", (username.strip().lower(),)
            ).fetchone()
            if row is None:
                return False
            uid = row["id"]
            con.execute("DELETE FROM sessions WHERE user_id = ?", (uid,))
            con.execute("DELETE FROM learner_profiles WHERE user_id = ?", (uid,))
            con.execute("DELETE FROM study_events WHERE user_id = ?", (uid,))
            con.execute("DELETE FROM saved_courses WHERE user_id = ?", (uid,))
            con.execute("DELETE FROM user_log WHERE user_id = ?", (uid,))
            con.execute("UPDATE recommendation_log SET user_id = NULL WHERE user_id = ?", (uid,))
            con.execute("DELETE FROM users WHERE id = ?", (uid,))
        return True

    def stats(self) -> dict:
        con = self._connect()
        q = lambda s: con.execute(s).fetchone()[0]
        return {
            "n_users": q("SELECT COUNT(*) FROM users"),
            "n_students": q("SELECT COUNT(*) FROM users WHERE role='student'"),
            "n_advisers": q("SELECT COUNT(*) FROM users WHERE role='adviser'"),
            "n_admins": q("SELECT COUNT(*) FROM users WHERE role='admin'"),
            "n_sessions": q("SELECT COUNT(*) FROM sessions WHERE expires_at > strftime('%s','now')"),
            "n_events": q("SELECT COUNT(*) FROM study_events"),
            "n_recommendations": q("SELECT COUNT(*) FROM recommendation_log"),
        }
