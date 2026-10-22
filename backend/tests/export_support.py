"""Shared fixtures for the Export & Share tests (not a test module)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

from olive.layout import meeting_paths

T0 = 1_700_000_000_000  # 2023-11-14T22:13:20Z

CANONICAL = {
    "segments": [
        {"startMs": 0, "endMs": 4000, "speaker": "Alice", "text": "Hello team."},
        {"startMs": 4000, "endMs": 6000, "speaker": "Alice", "text": "Let's begin."},
        {"startMs": 7000, "endMs": 9500, "speaker": "Bob", "text": "Sounds good."},
    ],
    "durationMs": 600_000,
}


def seed_meeting(
    db: Engine,
    meetings_dir: Path | str,
    meeting_id: str = "m1",
    *,
    title: str = "Weekly Sync",
    start: int = T0,
    end: int | None = None,
    transcript: Any = CANONICAL,
    summary_md: str | None = None,
    tags: list[str] | None = None,
) -> str:
    """Insert a meeting (+ optional transcript/summary artifacts written to disk)."""
    end = end if end is not None else start + 600_000
    folder = Path(meeting_paths(str(meetings_dir), start, title, meeting_id).folder)
    folder.mkdir(parents=True, exist_ok=True)
    with db.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO meetings (id, title, start_time, end_time, source, status, tags, "
                "created_at, updated_at) VALUES (:id, :t, :s, :e, 'upload', 'ready', :tags, 0, 0)"
            ),
            {"id": meeting_id, "t": title, "s": start, "e": end, "tags": json.dumps(tags or [])},
        )
        if transcript is not None:
            (folder / "t.json").write_text(json.dumps(transcript), encoding="utf-8")
            conn.execute(
                text(
                    "INSERT INTO artifacts (id, meeting_id, kind, provider, format, path, created_at) "
                    "VALUES (:a, :m, 'transcript', 'x', 'json', 't.json', 0)"
                ),
                {"a": f"t-{meeting_id}", "m": meeting_id},
            )
            conn.execute(
                text("UPDATE meetings SET primary_transcript_artifact_id = :a WHERE id = :m"),
                {"a": f"t-{meeting_id}", "m": meeting_id},
            )
        if summary_md is not None:
            (folder / "s.md").write_text(summary_md, encoding="utf-8")
            conn.execute(
                text(
                    "INSERT INTO artifacts (id, meeting_id, kind, provider, format, path, created_at) "
                    "VALUES (:a, :m, 'summary', 'x', 'md', 's.md', 1)"
                ),
                {"a": f"s-{meeting_id}", "m": meeting_id},
            )
            conn.execute(
                text("UPDATE meetings SET primary_summary_artifact_id = :a WHERE id = :m"),
                {"a": f"s-{meeting_id}", "m": meeting_id},
            )
    return meeting_id
