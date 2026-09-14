#!/usr/bin/env bash
# Compile paper.tex to IEEE two-column PDF using tectonic (no TeX Live needed).
#   brew install tectonic
set -e
cd "$(dirname "$0")"
tectonic -X compile paper.tex --outdir build --keep-logs
cp build/paper.pdf Revised-Paper-IEEE.pdf
echo
echo "built: Revised-Paper-IEEE.pdf"
python3 - <<'PY'
import re
d = open('Revised-Paper-IEEE.pdf', 'rb').read()
print("pages:", len(re.findall(rb'/Type\s*/Page[^s]', d)))
PY
echo "overfull boxes:"
grep -c 'Overfull' build/paper.log || echo 0
