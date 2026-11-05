"""Backfill of Plaud transcripts for meetings ingested audio-only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy import text

from olive.db import create_db
from olive.plaud import PlaudPoller

NOW = 5_000_000
FILE_ID = "of_backfill1"
MEETING_ID = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"

TRANSCRIPT_DETAIL: dict[str, Any] = {
    "id": FILE_ID,
    "presigned_url": "https://audio.test/x.m4a",
    "source_list": [
        {
            "data_type": "transcript",
            "data_content": json.dumps(
                [{"start_time": 0, "end_time": 5000, "speaker": "Alice", "content": "Ship it."}]
            ),
        }
    ],
    "note_list": [],
}
EMPTY_DETAIL: dict[str, Any] = {"id": FILE_ID, "presigned_url": "https://audio.test/x.m4a"}


class _Client:
    def __init__(self, detail: dict[str, Any]) -> None:
        self.detail = detail
        self.calls = 0

    async def get_file(self, _fid: str) -> dict[str, Any]:
        self.calls += 1
        return self.detail


def _seed(tmp_path: Path, detail: dict[str, Any]) -> tuple[Any, PlaudPoller, _Client]:
    handle = create_db(":memory:")
    meetings_dir = tmp_path / "meetings"
    client = _Client(detail)
    start = NOW - 60_000
    with handle.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO meetings (id, title, start_time, end_time, source, status, tags, "
                "primary_transcript_artifact_id, primary_summary_artifact_id, last_error, "
                "created_at, updated_at) VALUES (:id, 'Audio only', :s, :e, 'plaud', 'ready', "
                "'[]', NULL, NULL, NULL, :n, :n)"
            ),
            {"id": MEETING_ID, "s": start, "e": start + 60_000, "n": NOW},
        )
        conn.execute(
            text(
                "INSERT INTO plaud_ingest_state (meeting_id, plaud_file_id, first_seen_at, "
                "pcs_deadline_at, pcs_resolved) VALUES (:m, :f, :n, :n, 1)"
            ),
            {"m": MEETING_ID, "f": FILE_ID, "n": NOW - 1_000},
        )
    poller = PlaudPoller(
        {
            "db": handle.engine,
            "meetingsDir": str(meetings_dir),
            "client": client,
            "fetchImpl": _fake_fetch,
            "now": lambda: NOW,
        }
    )
    return handle, poller, client


class _Resp:
    status = 200
    headers = {"content-type": "audio/mp4"}
    ok = True

    async def arrayBuffer(self) -> bytes:
        return b"fake plaud audio"


async def _fake_fetch(_url: str) -> _Resp:
    return _Resp()


def _transcript_paths(handle: Any) -> list[str]:
    with handle.engine.connect() as conn:
        return [
            r[0]
            for r in conn.execute(
                text("SELECT path FROM artifacts WHERE meeting_id = :m AND kind = 'transcript'"),
                {"m": MEETING_ID},
            )
        ]


async def test_backfill_upgrades_audio_only_meeting(tmp_path: Path) -> None:
    handle, poller, _ = _seed(tmp_path, TRANSCRIPT_DETAIL)
    summary = await poller.backfill_plaud_transcripts()
    assert summary["upgraded"] == 1
    assert summary["checked"] == 1
    assert "transcripts/plaud.json" in _transcript_paths(handle)
    with handle.engine.connect() as conn:
        row = conn.execute(
            text("SELECT status, primary_transcript_artifact_id FROM meetings WHERE id = :m"),
            {"m": MEETING_ID},
        ).one()
    assert row[0] == "ready"
    assert row[1] is not None
    handle.engine.dispose()


async def test_backfill_leaves_still_untranscribed_meeting(tmp_path: Path) -> None:
    handle, poller, _ = _seed(tmp_path, EMPTY_DETAIL)
    summary = await poller.backfill_plaud_transcripts()
    assert summary["stillPending"] == 1
    assert summary["upgraded"] == 0
    assert _transcript_paths(handle) == []
    handle.engine.dispose()


async def test_backfill_cooldown_skips_recently_checked(tmp_path: Path) -> None:
    handle, poller, client = _seed(tmp_path, EMPTY_DETAIL)
    first = await poller.backfill_plaud_transcripts(cooldown_ms=0)
    assert first["checked"] == 1
    assert client.calls == 1

    skipped = await poller.backfill_plaud_transcripts(cooldown_ms=10**9)
    assert skipped["checked"] == 0
    assert client.calls == 1

    again = await poller.backfill_plaud_transcripts(cooldown_ms=0)
    assert again["checked"] == 1
    assert client.calls == 2
    handle.engine.dispose()


async def test_backfill_skips_tombstoned_recording(tmp_path: Path) -> None:
    handle, poller, client = _seed(tmp_path, TRANSCRIPT_DETAIL)
    with handle.engine.begin() as conn:
        conn.execute(
            text("INSERT INTO plaud_deleted_files (plaud_file_id, deleted_at) VALUES (:f, :n)"),
            {"f": "backfill1", "n": NOW},
        )
    summary = await poller.backfill_plaud_transcripts()
    assert summary["upgraded"] == 0
    assert client.calls == 0
    assert _transcript_paths(handle) == []
    handle.engine.dispose()
