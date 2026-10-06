"""Copy the accounts database to data/backups/, safely, while the site is running.

data/app.db holds every account and every recorded week, and nothing else on the
machine can rebuild it. SQLite's own backup call takes a consistent copy even while
the server is writing, which a plain file copy does not promise.

    .venv/bin/python scripts/11_backup.py            # keep the newest 14 copies
    .venv/bin/python scripts/11_backup.py --keep 30

Run it daily from cron, for example:
    0 3 * * *  cd /path/to/site && .venv/bin/python scripts/11_backup.py
and copy data/backups/ off the machine too: a backup on the same disk does not survive
the disk.
"""
from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "app.db"
OUT = ROOT / "data" / "backups"


def backup(keep: int) -> Path:
    if not DB.exists():
        raise SystemExit(f"no database at {DB}")
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"app-{time.strftime('%Y%m%d-%H%M%S')}.db"
    src, dst = sqlite3.connect(DB), sqlite3.connect(target)
    with dst:
        src.backup(dst)
    src.close(); dst.close()
    # A copy that cannot be read back is not a backup.
    check = sqlite3.connect(target)
    ok = check.execute("PRAGMA integrity_check").fetchone()[0]
    users = check.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    check.close()
    if ok != "ok":
        target.unlink()
        raise SystemExit(f"backup failed its integrity check: {ok}")
    target.chmod(0o600)                      # accounts: readable by the owner only
    for old in sorted(OUT.glob("app-*.db"))[:-keep]:
        old.unlink()
    print(f"{target.name}: {users} accounts, integrity ok")
    return target


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--keep", type=int, default=14, help="how many copies to keep")
    backup(ap.parse_args().keep)
