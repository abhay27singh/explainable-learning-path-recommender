"""Draw the site's icon files from the logo: favicon.ico, the home-screen icon, and
the picture shown when a link to the site is shared (WhatsApp, Telegram, email).

Run again after changing the name or the logo:
    .venv/bin/python scripts/10_icons.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "icons"
NAME = "GyanGraph"
LINE = "Your next week of study, and why it comes next."
SUB = "For students in India, from class 10 to postgraduate. Free."
ACCENT, INK, SOFT, GROUND = "#0369A1", "#0F243D", "#536B85", "#F3F9FE"


def logo(size: int) -> Image.Image:
    """The mark from the page header: a rounded square, a rising line, three dots."""
    s = size * 8                                   # draw big, shrink smooth
    k = s / 32
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, s - 1, s - 1], radius=7 * k, fill=ACCENT)
    pts = [(7 * k, 24 * k), (16 * k, 16 * k), (25 * k, 8 * k)]
    d.line(pts, fill="white", width=round(2.2 * k), joint="curve")
    for x, y in pts:
        r = 3.2 * k
        d.ellipse([x - r, y - r, x + r, y + r], fill="white")
    return im.resize((size, size), Image.LANCZOS)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for path, index in (("/System/Library/Fonts/HelveticaNeue.ttc", 1 if bold else 0),
                        ("/System/Library/Fonts/Helvetica.ttc", 1 if bold else 0)):
        try:
            return ImageFont.truetype(path, size, index=index)
        except OSError:
            continue
    return ImageFont.load_default()


def share_card() -> Image.Image:
    im = Image.new("RGB", (1200, 630), GROUND)
    im.paste(logo(160), (96, 120), logo(160))
    d = ImageDraw.Draw(im)
    d.text((96, 330), NAME, font=font(64, bold=True), fill=INK)
    d.text((96, 420), LINE, font=font(36), fill=INK)
    d.text((96, 474), SUB, font=font(30), fill=SOFT)
    return im


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    logo(256).save(OUT / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)])
    logo(180).convert("RGB").save(OUT / "apple-touch-icon.png", optimize=True)
    logo(512).save(OUT / "icon-512.png", optimize=True)
    share_card().save(OUT / "share.png", optimize=True)
    for f in sorted(OUT.iterdir()):
        print(f"{f.name}: {f.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
