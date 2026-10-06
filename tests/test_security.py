"""What a hostile or careless visitor runs into before launch.

Each test names the attack it closes: scripted sign-ups, a flood of model calls, a page
framed by another site, a guessed admin address, a malicious model file. They call the
real middleware and handlers, not copies of them.
"""
from __future__ import annotations

import asyncio
import re
import stat
from pathlib import Path

from fastapi import Response
from starlette.requests import Request

from api import main

ROOT = Path(__file__).resolve().parents[1]


def _request(path="/", method="GET", ip="203.0.113.7", headers=()):
    return Request({"type": "http", "method": method, "path": path, "raw_path": path.encode(),
                    "query_string": b"", "headers": [(k.encode(), v.encode()) for k, v in headers],
                    "client": (ip, 5000), "server": ("testserver", 80), "scheme": "http",
                    "root_path": ""})


def _through_guard(path="/", method="GET", ip="203.0.113.7"):
    async def ok(_):
        return Response("fine")
    return asyncio.run(main.guard(_request(path, method, ip), ok))


def test_every_response_carries_the_security_headers():
    out = _through_guard("/")
    assert out.headers["x-frame-options"] == "DENY", "the site must not be framed by another"
    assert out.headers["x-content-type-options"] == "nosniff"
    csp = out.headers["content-security-policy"]
    assert "frame-ancestors 'none'" in csp and "object-src 'none'" in csp
    assert "default-src 'self'" in csp


def test_hsts_only_once_the_site_is_on_https():
    """Sending HSTS over plain HTTP during development would lock a browser out."""
    assert "strict-transport-security" not in _through_guard("/").headers or main.HTTPS


def test_scripted_sign_ups_are_refused_after_five_an_hour():
    ip = "198.51.100.21"
    codes = [_through_guard("/api/auth/register", "POST", ip).status_code for _ in range(6)]
    assert codes[:5] == [200] * 5 and codes[5] == 429


def test_the_what_if_cannot_be_used_to_tie_up_the_model():
    ip = "198.51.100.22"
    codes = [_through_guard("/api/me/what-if", "GET", ip).status_code for _ in range(21)]
    assert codes.count(429) == 1 and codes[-1] == 429


def test_limits_are_per_address():
    assert _through_guard("/api/auth/register", "POST", "198.51.100.23").status_code == 200


def test_a_rate_limit_forgets_old_requests():
    limit = main.RateLimit(2, 60)
    assert limit.allow("k", 0) and limit.allow("k", 1) and not limit.allow("k", 2)
    assert limit.allow("k", 61.5), "a minute later the window has moved on"


def test_the_client_address_is_not_taken_from_headers_unless_told_to():
    """Otherwise anyone could dodge every limit by sending a made-up X-Forwarded-For."""
    req = _request(headers=[("x-forwarded-for", "1.1.1.1")], ip="203.0.113.9")
    assert main.client_key(req) == ("1.1.1.1" if main.TRUST_PROXY else "203.0.113.9")


def test_the_generated_api_pages_are_off_by_default():
    assert main.app.docs_url is None and main.app.redoc_url is None and main.app.openapi_url is None


def test_a_missing_page_gets_a_page_and_a_missing_api_route_gets_json():
    from starlette.exceptions import HTTPException
    page = asyncio.run(main.not_found(_request("/no-such-page"), HTTPException(404)))
    assert page.status_code == 404 and b"That page is not here" in page.body
    api = asyncio.run(main.not_found(_request("/api/nothing"), HTTPException(404, "Not Found")))
    assert api.status_code == 404 and api.media_type == "application/json"


def test_the_page_is_sent_with_the_site_address_for_link_previews():
    out = main.index(_request("/"))
    assert b"{{BASE_URL}}" not in out.body
    assert re.search(rb'og:image" content="http://testserver/static/icons/share.png"', out.body)
    assert out.headers["cache-control"].startswith("no-store")


def test_model_files_are_loaded_as_data_only():
    """torch 2.2 (the last build for Intel Macs) can run code hidden in a model file."""
    for path in ("elpr/planner/engine.py", "api/service.py"):
        loads = re.findall(r"torch\.load\([^)]*\)", (ROOT / path).read_text())
        assert loads and all("weights_only=True" in call for call in loads), path


def test_a_new_database_is_readable_by_its_owner_only(tmp_path):
    from elpr.db.app_store import AppStore
    AppStore(tmp_path / "app.db")
    mode = stat.S_IMODE((tmp_path / "app.db").stat().st_mode)
    assert mode & 0o077 == 0, oct(mode)


def test_nothing_secret_is_tracked_by_git():
    import subprocess
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split()
    leaks = [f for f in tracked if re.search(r"(^|/)(app\.db|\.env(\..*)?|.*\.pem|id_rsa)$", f)
             or f.startswith("data/backups/")]
    assert leaks == []
