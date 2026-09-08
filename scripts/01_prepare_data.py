#!/usr/bin/env python
"""Stage 0, part 1: load OULAD into DuckDB and build the unified event stream.

Runs sql/00 through sql/05 and writes measured statistics to results/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from elpr.db.duckdb_runner import SQLRunner  # noqa: E402

RESULTS = ROOT / "results"


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    runner = SQLRunner()

    print("Running SQL ETL (sql/00-05):")
    runner.run_range(0, 5)

    stats = {
        row.metric: row.value
        for row in runner.df("SELECT * FROM dataset_stats").itertuples()
    }
    stats["outcome_distribution"] = (
        runner.df("SELECT * FROM outcome_distribution").to_dict("records")
    )
    stats["concept_provenance"] = (
        runner.df("SELECT * FROM concept_provenance").to_dict("records")
    )
    stats["event_stats"] = runner.df("SELECT * FROM event_stats").to_dict("records")
    stats["n_concepts"] = runner.scalar("SELECT COUNT(*) FROM concepts")

    out = RESULTS / "dataset_stats.json"
    out.write_text(json.dumps(stats, indent=2, default=str))
    print(f"\nWrote {out.relative_to(ROOT)}")

    print("\nMeasured statistics:")
    for k in (
        "n_students", "n_enrolments", "n_vle_events", "n_assessment_events",
        "n_vle_sites", "n_vle_sites_with_week", "n_concepts",
        "vle_events_per_student_median", "asm_events_per_student_median",
        "pass_rate_at_40",
    ):
        if k in stats:
            v = stats[k]
            print(f"  {k:<32} {v:>14,.2f}" if isinstance(v, float) else f"  {k:<32} {v:>14}")

    print("\nEvent stream:")
    print(runner.df("SELECT * FROM event_stats").to_string(index=False))

    runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
