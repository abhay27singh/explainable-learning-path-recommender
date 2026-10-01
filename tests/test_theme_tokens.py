"""The page's colour tokens live in three blocks that must stay in step.

Bare :root is the light theme. The dark theme is written twice: once under
prefers-color-scheme for people whose system is dark, and once under
[data-theme="dark"] for people who pick it with the switch. A token missing from one of
them silently falls back to the light value in that case only, which is how the
activity calendar once showed light cells on a dark page for anyone using the switch.
"""
from __future__ import annotations

import re
from pathlib import Path

PAGE = (Path(__file__).resolve().parents[1] / "web" / "index.html").read_text()


def tokens(body: str) -> dict[str, str]:
    """Token to value, with whitespace collapsed: a value may wrap over lines."""
    return {k: " ".join(v.split())
            for k, v in re.findall(r"(--[a-z0-9-]+)\s*:\s*([^;]+);", body)}


def block(pattern: str) -> str:
    m = re.search(pattern, PAGE, re.S)
    assert m, pattern
    return m.group(1)


LIGHT = block(r"\n  :root\{\n(.*?)\n  \}\n")
SYSTEM_DARK = block(r':root:not\(\[data-theme="light"\]\)\{\n(.*?)\n    \}\n')
CHOSEN_DARK = block(r'\n  :root\[data-theme="dark"\]\{\n(.*?)\n  \}\n')


def test_both_dark_blocks_define_exactly_the_same_tokens():
    assert tokens(SYSTEM_DARK) == tokens(CHOSEN_DARK)


def test_every_dark_token_also_exists_in_the_light_block():
    """A token that exists only under a dark selector has no value at all in light."""
    assert set(tokens(SYSTEM_DARK)) <= set(tokens(LIGHT))


def test_no_token_swallowed_the_next_one():
    """Regression: a missing semicolon made --surface-alt read "#212A31 --seg-on:..."
    in dark mode, which is not a colour, and left --seg-on undefined."""
    for name, body in (("light", LIGHT), ("system dark", SYSTEM_DARK),
                       ("chosen dark", CHOSEN_DARK)):
        for token, value in tokens(body).items():
            assert "--" not in value.split("var(")[0], f"{name}: {token} = {value}"


def test_the_glass_falls_back_to_solid_for_reduced_transparency():
    assert "prefers-reduced-transparency" in PAGE
