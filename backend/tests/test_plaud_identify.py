"""Plaud ASR-first speaker identification tests (plaud_sync queue job)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy import text

from olive.db import create_db
from olive.layout import ensure_meeting_folder, meeting_paths
from olive.ml.protocols import DiscoveredSpeakerVoiceprint
from olive.plaud import PlaudPoller
from olive.plaud.poller import plaud_segments_to_transcript
from olive.schemas import Transcript, TranscriptSegment
from olive.workflows import QUEUE_PLAUD_SYNC


def test_conversion_keeps_millisecond_scale() -> None:
    transcript = plaud_segments_to_transcript(
        [
            {"start_time": 0, "end_time": 5000, "speaker": "Alice", "content": "Hi."},
            {"start_time": 5000, "end_time": 12000, "speaker": "Speaker 1", "content": "Yo."},
        ],
        12000,
    )
    assert transcript is not None
    assert [(s.startMs, s.endMs, s.speaker) for s in transcript.segments] == [
        (0, 5000, "Alice"),
        (5000, 12000, "Speaker 1"),
    ]


def test_conversion_detects_seconds_scale() -> None:
    transcript = plaud_segments_to_transcript(
        [
            {"start_time": 0, "end_time": 5, "speaker": "A", "content": "Hi."},
            {"start_time": 5, "end_time": 12, "speaker": "B", "content": "Yo."},
        ],
        12000,
    )
    assert transcript is not None
    assert [(s.startMs, s.endMs) for s in transcript.segments] == [(0, 5000), (5000, 12000)]


def test_conversion_does_not_rescale_short_millisecond_turns() -> None:
    transcript = plaud_segments_to_transcript(
        [{"start_time": 0, "end_time": 500, "speaker": "A", "content": "Hi."}],
        60000,
    )
    assert transcript is not None
    assert (transcript.segments[0].startMs, transcript.segments[0].endMs) == (0, 500)


def test_conversion_maps_missing_speaker_to_uu_and_drops_bad_timings() -> None:
    assert plaud_segments_to_transcript([], 1000) is None
    assert plaud_segments_to_transcript([{"start_time": 5, "end_time": 5}], 1000) is None
    transcript = plaud_segments_to_transcript(
        [
            {"start_time": 0, "end_time": 1000, "speaker": None, "content": "Hi."},
            {"start_time": 9, "end_time": 2, "speaker": "A", "content": "Bad."},
            {"start_time": "x", "end_time": 3, "speaker": "B", "content": "Bad."},
        ],
        2000,
    )
    assert transcript is not None
    assert [(s.speaker, s.text) for s in transcript.segments] == [("UU", "Hi.")]


def _seed_identify_meeting(
    tmp_path: Path, segments: list[dict[str, Any]], now: int = 2_000_000
) -> tuple[Any, PlaudPoller, str, str, str]:
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    meeting_id = "11111111-2222-4333-8333-111111111111"
    title = "Plaud Identify Fixture"
    start = now - 60_000
    paths = ensure_meeting_folder(meeting_paths(meetings_dir, start, title, meeting_id))
    audio_rel = "audio/rec.m4a"
    (Path(paths.folder) / audio_rel).parent.mkdir(parents=True, exist_ok=True)
    (Path(paths.folder) / audio_rel).write_bytes(b"fake audio")
    (Path(paths.folder) / "transcripts").mkdir(parents=True, exist_ok=True)
    (Path(paths.folder) / "transcripts/plaud.json").write_text(
        f"{json.dumps(segments, indent=2)}\n", encoding="utf-8"
    )
    (Path(paths.folder) / "transcripts/plaud.txt").write_text("old\n", encoding="utf-8")
    with handle.engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO meetings (
                  id, title, start_time, end_time, source, status, tags,
                  primary_transcript_artifact_id, primary_summary_artifact_id,
                  last_error, created_at, updated_at
                ) VALUES (
                  :id, :title, :start, :end, 'plaud', 'ready', '[]',
                  NULL, NULL, NULL, :now, :now
                )
                """
            ),
            {"id": meeting_id, "title": title, "start": start, "end": start + 60_000, "now": now},
        )
        conn.execute(
            text(
                """
                INSERT INTO recordings (
                  id, meeting_id, path, mime, duration_ms, size_bytes, sha256,
                  provider, provider_recording_id, created_at
                ) VALUES (
                  'rec-1', :mid, :path, 'audio/mp4', 60000, 10, 'x',
                  'plaud', 'of_abc', :now
                )
                """
            ),
            {"mid": meeting_id, "path": audio_rel, "now": now},
        )
        conn.execute(
            text(
                """
                INSERT INTO artifacts (
                  id, meeting_id, recording_id, kind, provider, format, path, created_at
                ) VALUES (
                  'art-plaud-json', :mid, 'rec-1', 'transcript', 'plaud', 'json',
                  'transcripts/plaud.json', :now
                )
                """
            ),
            {"mid": meeting_id, "now": now},
        )
        conn.execute(
            text(
                """
                INSERT INTO speakers (
                  id, name, provider_ids, enrolled_at, enrollment_clip_paths, created_at
                ) VALUES (
                  'spk-matt', 'Matt', :pids, :now, '[]', :now
                )
                """
            ),
            {"pids": json.dumps({"local": [json.dumps([0.5, 0.5, 0.5, 0.5])]}), "now": now},
        )
        conn.execute(
            text(
                """
                INSERT INTO speakers (
                  id, name, provider_ids, enrolled_at, enrollment_clip_paths, created_at
                ) VALUES (
                  'spk-alice', 'Alice', '{}', NULL, '[]', :now
                )
                """
            ),
            {"now": now},
        )
        conn.execute(
            text("UPDATE meetings SET primary_transcript_artifact_id = 'art-plaud-json' WHERE id = :mid"),
            {"mid": meeting_id},
        )
        conn.execute(
            text(
                """
                INSERT INTO meeting_speakers (meeting_id, speaker_id, evidence_artifact_id)
                VALUES (:mid, 'spk-alice', 'art-plaud-json')
                """
            ),
            {"mid": meeting_id},
        )
    poller = PlaudPoller({"db": handle.engine, "meetingsDir": meetings_dir, "now": lambda: now})
    folder = Path(paths.folder)
    return handle, poller, meeting_id, str(folder), "art-plaud-json"


def _plaud_segments() -> list[dict[str, Any]]:
    return [
        {"start_time": 0, "end_time": 5000, "speaker": "Speaker 1", "content": "Hello there."},
        {"start_time": 5000, "end_time": 12000, "speaker": "Alice", "content": "Hi back."},
    ]


async def test_identify_job_renames_matched_speakers_and_prunes_stale_links(
    tmp_path: Path, monkeypatch: Any
) -> None:
    handle, poller, meeting_id, folder, _art = _seed_identify_meeting(tmp_path, _plaud_segments())

    async def _fake_identify(
        _audio: str, transcript: Transcript, _enrolled: Any, **_kwargs: Any
    ) -> tuple[Transcript, list[DiscoveredSpeakerVoiceprint]]:
        # Both Plaud labels voice-match enrolled Matt — including the
        # unverified "Alice" name, which must lose its link afterwards.
        renamed = Transcript(
            segments=[
                TranscriptSegment(startMs=s.startMs, endMs=s.endMs, speaker="Matt", text=s.text)
                for s in transcript.segments
            ]
        )
        return renamed, [
            DiscoveredSpeakerVoiceprint(
                speakerId="spk-matt",
                name="Matt",
                voiceprint=[0.5, 0.5, 0.5, 0.5],
                isEnrolled=True,
                similarityScore=0.9,
            )
        ]

    monkeypatch.setattr("olive.ml.identify.identify_cloud_transcript_speakers", _fake_identify)
    res = await poller.run_plaud_speaker_identification(meeting_id)

    assert res["status"] == "renamed"
    assert res["matched"] == ["Matt"]
    rewritten = json.loads((Path(folder) / "transcripts/plaud.json").read_text(encoding="utf-8"))
    assert [s["speaker"] for s in rewritten] == ["Matt", "Matt"]
    assert [s["content"] for s in rewritten] == ["Hello there.", "Hi back."]
    txt = (Path(folder) / "transcripts/plaud.txt").read_text(encoding="utf-8")
    assert "Matt: Hello there." in txt
    assert "Matt: Hi back." in txt
    assert "Speaker 1" not in txt

    with handle.engine.connect() as conn:
        links = {
            r["speaker_id"]: r["evidence_artifact_id"]
            for r in conn.execute(text("SELECT * FROM meeting_speakers")).mappings().all()
        }
        speakers = {r["id"]: r["name"] for r in conn.execute(text("SELECT * FROM speakers")).mappings().all()}
    # Matt linked to the Plaud artifact; the unverified "Alice" link was pruned
    # because that label no longer appears in the transcript.
    assert links == {"spk-matt": "art-plaud-json"}
    assert set(speakers) == {"spk-matt", "spk-alice"}
    handle.engine.dispose()


async def test_identify_job_keeps_labels_when_nothing_matches(tmp_path: Path, monkeypatch: Any) -> None:
    handle, poller, meeting_id, folder, _art = _seed_identify_meeting(tmp_path, _plaud_segments())

    async def _fake_identify_none(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr("olive.ml.identify.identify_cloud_transcript_speakers", _fake_identify_none)
    before = (Path(folder) / "transcripts/plaud.json").read_text(encoding="utf-8")
    res = await poller.run_plaud_speaker_identification(meeting_id)

    assert res["status"] == "kept-labels"
    assert (Path(folder) / "transcripts/plaud.json").read_text(encoding="utf-8") == before
    with handle.engine.connect() as conn:
        links = [
            r["speaker_id"] for r in conn.execute(text("SELECT * FROM meeting_speakers")).mappings().all()
        ]
    assert links == ["spk-alice"]
    handle.engine.dispose()


def _mark_ingested(handle: Any, meeting_id: str, now: int) -> None:
    with handle.engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO plaud_ingest_state (
                  meeting_id, plaud_file_id, first_seen_at, pcs_deadline_at, pcs_resolved
                ) VALUES (:mid, 'of_abc', :seen, :deadline, 1)
                """
            ),
            {"mid": meeting_id, "seen": now - 600_000, "deadline": now - 300_000},
        )


async def test_backfill_queues_unresolved_meetings(tmp_path: Path, monkeypatch: Any) -> None:
    handle, poller, meeting_id, _folder, _art = _seed_identify_meeting(tmp_path, _plaud_segments())
    _mark_ingested(handle, meeting_id, 2_000_000)
    queued: list[str] = []

    async def _fake_ensure(**_kwargs: Any) -> str:
        return "sqlite:///:memory:"

    async def _fake_enqueue(queue_name: str, fn: Any, **kwargs: Any) -> Any:
        queued.append(str(kwargs.get("deduplication_id")))
        return None

    monkeypatch.setattr("olive.workflows.ensure_dbos_launched_async", _fake_ensure)
    monkeypatch.setattr("olive.workflows.enqueue_stage_job", _fake_enqueue)
    summary = await poller.backfill_plaud_speaker_identification()

    assert summary == {"checked": 1, "queued": 1, "skipped": 0}
    assert queued == [f"plaud-identify:{meeting_id}"]
    handle.engine.dispose()


async def test_backfill_skips_fully_resolved_meetings(tmp_path: Path, monkeypatch: Any) -> None:
    resolved = [
        {"start_time": 0, "end_time": 5000, "speaker": "Matt", "content": "Hi."},
        {"start_time": 5000, "end_time": 9000, "speaker": "matt", "content": "Yo."},
    ]
    handle, poller, meeting_id, _folder, _art = _seed_identify_meeting(tmp_path, resolved)
    _mark_ingested(handle, meeting_id, 2_000_000)

    async def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("must not enqueue")

    monkeypatch.setattr("olive.workflows.enqueue_stage_job", _boom)
    summary = await poller.backfill_plaud_speaker_identification()

    assert summary == {"checked": 1, "queued": 0, "skipped": 1}
    handle.engine.dispose()


async def test_backfill_noops_without_enrolled_voiceprints(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    poller = PlaudPoller({"db": handle.engine, "meetingsDir": str(tmp_path)})
    assert await poller.backfill_plaud_speaker_identification() == {
        "checked": 0,
        "queued": 0,
        "skipped": 0,
    }
    handle.engine.dispose()


async def test_enqueue_skipped_without_enrolled_voiceprints(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    with handle.engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO speakers (
                  id, name, provider_ids, enrolled_at, enrollment_clip_paths, created_at
                ) VALUES ('spk-a', 'Alice', '{}', NULL, '[]', 1)
                """
            )
        )
    poller = PlaudPoller({"db": handle.engine, "meetingsDir": str(tmp_path)})
    assert await poller._enqueue_plaud_speaker_identification("mid-1") is False
    handle.engine.dispose()


async def test_enqueue_queues_plaud_sync_job(tmp_path: Path, monkeypatch: Any) -> None:
    handle, poller, meeting_id, _folder, _art = _seed_identify_meeting(tmp_path, _plaud_segments())
    captured: dict[str, Any] = {}

    async def _fake_ensure(**_kwargs: Any) -> str:
        return "sqlite:///:memory:"

    async def _fake_enqueue(queue_name: str, fn: Any, **kwargs: Any) -> Any:
        captured["queue"] = queue_name
        captured.update(kwargs)
        captured["fn"] = fn
        return None

    monkeypatch.setattr("olive.workflows.ensure_dbos_launched_async", _fake_ensure)
    monkeypatch.setattr("olive.workflows.enqueue_stage_job", _fake_enqueue)
    assert await poller._enqueue_plaud_speaker_identification(meeting_id) is True
    assert captured["queue"] == QUEUE_PLAUD_SYNC
    assert captured["deduplication_id"] == f"plaud-identify:{meeting_id}"
    assert captured["label"] == "plaud-identify"
    assert captured["meeting_id"] == meeting_id
    handle.engine.dispose()


def test_conversion_accepts_normalized_startms_segments() -> None:
    """Stored ``plaud.json`` is often ``{startMs, endMs, speaker, text}`` (not raw Plaud keys).

    Reading only ``start_time``/``end_time``/``content`` collapsed every such
    segment to ``0..0`` and returned ``None`` — re-identification then
    silently skipped the meeting.
    """
    transcript = plaud_segments_to_transcript(
        [
            {"startMs": 33820, "endMs": 84280, "speaker": "Speaker 1", "text": "Hi.", "verified": False, "words": []},
            {"startMs": 84280, "endMs": 90000, "speaker": "Speaker 2", "text": "Yo.", "verified": False, "words": []},
        ],
        2224000,
    )
    assert transcript is not None
    assert [(s.startMs, s.endMs, s.speaker, s.text) for s in transcript.segments] == [
        (33820, 84280, "Speaker 1", "Hi."),
        (84280, 90000, "Speaker 2", "Yo."),
    ]


def test_conversion_does_not_rescale_explicit_startms() -> None:
    # Short, explicit-ms transcript inside a long recording must not hit the seconds heuristic.
    transcript = plaud_segments_to_transcript(
        [{"startMs": 1000, "endMs": 5000, "speaker": "A", "text": "Hi."}],
        3_600_000,
    )
    assert transcript is not None
    assert (transcript.segments[0].startMs, transcript.segments[0].endMs) == (1000, 5000)
