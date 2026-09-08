#!/usr/bin/env python
"""Bundle everything Colab needs into a single upload.

The pipeline is artifact-based, so training elsewhere needs no code changes: upload
this zip, unpack, run the same scripts, download the checkpoints and results.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "elpr_colab_bundle.zip"

INCLUDE = [
    "elpr/**/*.py",
    "scripts/05_train_kt.py",
    "scripts/run_all_kt.py",
    "data/processed/sequences.parquet",
    "data/processed/student_features.parquet",
    "artifacts/graph/adjacency.pt",
    "artifacts/graph/concepts.parquet",
    "artifacts/graph/edges.parquet",
    "pyproject.toml",
]


def main() -> int:
    files: list[Path] = []
    for pattern in INCLUDE:
        matched = sorted(ROOT.glob(pattern))
        if not matched:
            print(f"  WARNING: nothing matched {pattern}")
        files.extend(p for p in matched if p.is_file())

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, path.relative_to(ROOT))

    size_mb = OUT.stat().st_size / 1e6
    print(f"Wrote {OUT.name}  ({size_mb:.1f} MB, {len(files)} files)")
    for path in files:
        print(f"  {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
