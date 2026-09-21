"""Calendar files, so a study plan can leave this site and land in a real calendar.

One all-day event per week, in iCalendar (RFC 5545). A plan nobody can carry is a plan
nobody follows: this is the difference between a page a student reads once and a
reminder that arrives on a Monday morning.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

PRODID = "-//Learning Path Recommender//Study plan//EN"
# RFC 5545 counts a line as at most 75 octets; longer ones are folded with CRLF + space.
LINE_LIMIT = 75


def _escape(text: str) -> str:
    """Commas, semicolons and backslashes are separators in iCalendar and must be
    escaped, and a newline is written as the two characters backslash-n."""
    out = (text.replace("\\", "\\\\").replace(";", "\;")
               .replace(",", "\\,").replace("\r\n", "\n").replace("\r", "\n"))
    return out.replace("\n", "\\n")


def _fold(line: str) -> str:
    raw = line.encode("utf-8")
    if len(raw) <= LINE_LIMIT:
        return line
    chunks, at = [], 0
    while at < len(raw):
        take = LINE_LIMIT if not chunks else LINE_LIMIT - 1
        cut = at + take
        while cut < len(raw) and (raw[cut] & 0xC0) == 0x80:      # never split a character
            cut -= 1
        chunk = raw[at:cut].decode("utf-8")
        chunks.append(chunk if not chunks else " " + chunk)
        at = cut
    return "\r\n".join(chunks)


def calendar(name: str, events: list[dict], now: datetime | None = None) -> str:
    """Build an .ics file. Each event needs uid, start (a date), days, summary, and
    may carry description and url. Events are all-day, which is what a week is."""
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", f"PRODID:{PRODID}", "CALSCALE:GREGORIAN",
             "METHOD:PUBLISH", f"X-WR-CALNAME:{_escape(name)}"]
    for event in events:
        start: date = event["start"]
        end = start + timedelta(days=int(event.get("days", 7)))   # DTEND is exclusive
        lines += [
            "BEGIN:VEVENT",
            f"UID:{event['uid']}",
            f"DTSTAMP:{stamp}",
            f"DTSTART;VALUE=DATE:{start.strftime('%Y%m%d')}",
            f"DTEND;VALUE=DATE:{end.strftime('%Y%m%d')}",
            f"SUMMARY:{_escape(event['summary'])}",
        ]
        if event.get("description"):
            lines.append(f"DESCRIPTION:{_escape(event['description'])}")
        if event.get("url"):
            lines.append(f"URL:{event['url']}")
        lines.append("END:VEVENT")
    lines.append("END:VCALENDAR")
    return "\r\n".join(_fold(line) for line in lines) + "\r\n"
