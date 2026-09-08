PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: venv install data graph sequences test clean stage0

venv:
	/opt/homebrew/bin/python3.11 -m venv .venv || /usr/local/bin/python3.11 -m venv .venv
	$(PIP) install --upgrade pip

install: venv
	$(PIP) install -e ".[dev]"

## Stage 0 ---------------------------------------------------------------
data:
	$(PY) scripts/01_prepare_data.py

graph:
	$(PY) scripts/02_build_graph.py

sequences:
	$(PY) scripts/03_build_sequences.py

stage0: data graph sequences

## Quality ---------------------------------------------------------------
test:
	$(PY) -m pytest -q

lint:
	.venv/bin/ruff check elpr scripts tests

clean:
	rm -rf data/oulad.duckdb data/processed/* artifacts/graph/* results/*.json

## Demo ------------------------------------------------------------------
api:
	$(PY) -m uvicorn api.main:app --port 8420 --reload

demo: api
