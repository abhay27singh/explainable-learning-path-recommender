"""Execute the numbered SQL files that make up the ETL against a DuckDB database.

The ETL lives in ``sql/`` as plain SQL so that every transformation is reviewable
without reading Python. This module only sequences those files and reports what
each one produced.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
SQL_DIR = ROOT / "sql"
DB_PATH = ROOT / "data" / "oulad.duckdb"
RAW_DIR = ROOT / "data" / "raw"


@dataclass
class StepResult:
    name: str
    seconds: float
    tables: dict[str, int]


class SQLRunner:
    """Runs ``sql/NN_*.sql`` files in order against a persistent DuckDB database."""

    def __init__(self, db_path: Path = DB_PATH, raw_dir: Path = RAW_DIR):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(db_path))
        self.raw_dir = raw_dir
        # 8 GB machine: cap memory so DuckDB spills to disk instead of being killed.
        self.con.execute("SET memory_limit='4GB'")
        self.con.execute("SET preserve_insertion_order=false")

    def _render(self, sql: str) -> str:
        return sql.replace("${RAW}", str(self.raw_dir))

    def run_file(self, path: Path) -> StepResult:
        sql = self._render(path.read_text())
        start = time.perf_counter()
        self.con.execute(sql)
        elapsed = time.perf_counter() - start
        created = re.findall(
            r"CREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMP\s+)?(?:TABLE|VIEW)\s+(?:IF\s+NOT\s+EXISTS\s+)?([\w.]+)",
            sql,
            flags=re.IGNORECASE,
        )
        tables = {}
        for name in dict.fromkeys(created):
            try:
                tables[name] = self.con.sql(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            except duckdb.Error:
                pass
        return StepResult(path.name, elapsed, tables)

    def run_range(self, lo: int, hi: int) -> list[StepResult]:
        """Run every ``sql/NN_*.sql`` whose numeric prefix falls in ``[lo, hi]``."""
        files = sorted(
            f for f in SQL_DIR.glob("*.sql") if lo <= int(f.name.split("_", 1)[0]) <= hi
        )
        results = []
        for f in files:
            res = self.run_file(f)
            results.append(res)
            summary = ", ".join(f"{k}={v:,}" for k, v in res.tables.items()) or "-"
            print(f"  {res.name:<28} {res.seconds:6.2f}s  {summary}")
        return results

    def df(self, sql: str):
        return self.con.sql(sql).df()

    def scalar(self, sql: str):
        return self.con.sql(sql).fetchone()[0]

    def close(self) -> None:
        self.con.close()
