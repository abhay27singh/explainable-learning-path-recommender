PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: venv install data graph sequences test clean stage0 api serve

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

# Production: no auto-reload, behind a reverse proxy that terminates HTTPS on this
# machine. Set ELPR_SITE_URL to the public address first, for example
#   ELPR_SITE_URL=https://example.in make serve
serve:
	ELPR_HTTPS=1 ELPR_TRUST_PROXY=1 $(PY) -m uvicorn api.main:app --host 127.0.0.1 --port 8420 \
		--proxy-headers --forwarded-allow-ips 127.0.0.1 --no-server-header

demo: api
