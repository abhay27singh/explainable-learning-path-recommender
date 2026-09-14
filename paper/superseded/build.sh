#!/usr/bin/env bash
# Rebuild Revised-Paper.pdf and .docx from paper.md
set -e
cd "$(dirname "$0")"
pandoc paper.md -o paper.html --standalone --mathml --css=style.css \
  --metadata title="Explainable AI-Based Personalized Learning Path Recommendation System"
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless --disable-gpu \
  --no-pdf-header-footer --print-to-pdf="$PWD/Revised-Paper.pdf" "file://$PWD/paper.html"
pandoc paper.md -o Revised-Paper.docx
echo "built: Revised-Paper.pdf  Revised-Paper.docx"
