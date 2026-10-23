"""Global cross-recording search: FTS5 indexer, query escaping, API, and index upkeep."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from olive.app import create_app
from olive.db import DbHandle, create_db
from olive.layout import meeting_paths
from olive.services import search as search_svc
from olive.services.meetings import delete_meeting, update_meeting_title

START = 1_760_000_000_000


class Env:
    def __init__(self, tmp_path: Path) -> None:
        self.db: DbHandle = create_db(":memory:")
        self.meetings_dir = str(tmp_path / "meetings")
        self.client = TestClient(
            create_app(
                {
                    "db": self.db.engine,
                    "meetingsDir": self.meetings_dir,
                    "configDir": str(tmp_path / "cfg"),
                }
            )
        )

    def meeting(
        self,
        mid: str,
        title: str,
        *,
        segments: list[dict[str, Any]] | None = None,
        summary: str | None = None,
        start: int = START,
    ) -> None:
        paths = meeting_paths(self.meetings_dir, start, title, mid)
        folder = Path(paths.folder)
        with self.db.engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO meetings (id, title, start_time, end_time, source, status, tags, "
                    "created_at, updated_at) VALUES (:id, :t, :s, :e, 'upload', 'ready', '[]', 0, 0)"
                ),
                {"id": mid, "t": title, "s": start, "e": start + 600_000},
            )
            if segments is not None:
                (folder / "transcripts").mkdir(parents=True, exist_ok=True)
                (folder / "transcripts/t.json").write_text(json.dumps({"segments": segments}))
                conn.execute(
                    text(
                        "INSERT INTO artifacts (id, meeting_id, kind, provider, format, path, created_at) "
                        "VALUES (:a, :m, 'transcript', 'local', 'json', 'transcripts/t.json', 1)"
                    ),
                    {"a": f"{mid}-t", "m": mid},
                )
            if summary is not None:
                (folder / "summaries").mkdir(parents=True, exist_ok=True)
                (folder / "summaries/s.md").write_text(summary)
                conn.execute(
                    text(
                        "INSERT INTO artifacts (id, meeting_id, kind, provider, format, path, created_at) "
                        "VALUES (:a, :m, 'summary', 'llm', 'md', 'summaries/s.md', 1)"
                    ),
                    {"a": f"{mid}-s", "m": mid},
                )

    def moment(self, mid: str, quote: str, start_ms: int) -> None:
        with self.db.engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO meeting_moments (id, meeting_id, start_ms, event_type, valence, quote, "
                    "rationale, created_at) VALUES (:i, :m, :s, 'praise', 'positive', :q, 'because', 0)"
                ),
                {"i": f"mo-{mid}-{start_ms}", "m": mid, "s": start_ms, "q": quote},
            )

    def reindex(self, mid: str) -> int:
        return search_svc.reindex_meeting(self.db.engine, mid, self.meetings_dir)

    def rows(self, mid: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT meeting_id, kind, start_ms, text FROM search_fts"
        params: dict[str, Any] = {}
        if mid:
            sql += " WHERE meeting_id = :m"
            params["m"] = mid
        with self.db.engine.connect() as conn:
            return [dict(r) for r in conn.execute(text(sql), params).mappings().all()]


@pytest.fixture
def env(tmp_path: Path) -> Env:
    return Env(tmp_path)


def seg(start: int, speaker: str, body: str) -> dict[str, Any]:
    return {"startMs": start, "endMs": start + 4000, "speaker": speaker, "text": body}


def test_reindex_covers_all_kinds_and_is_idempotent(env: Env) -> None:
    env.meeting(
        "m1",
        "Quarterly Budget Review",
        segments=[seg(0, "A", "Welcome everyone"), seg(65_000, "B", "The migration deadline slipped")],
        summary="# Notes\n\nWe agreed to renegotiate the vendor contract.",
    )
    env.moment("m1", "You never listen to the roadmap", 70_000)

    first = env.reindex("m1")
    snapshot = sorted(map(str, env.rows("m1")))
    second = env.reindex("m1")

    assert first == second > 0
    assert sorted(map(str, env.rows("m1"))) == snapshot
    kinds = {r["kind"] for r in env.rows("m1")}
    assert kinds == {"title", "transcript", "summary", "moment"}
    transcript = [r for r in env.rows("m1") if r["kind"] == "transcript"]
    assert transcript[0]["start_ms"] == 0
    assert any("migration" in r["text"] for r in transcript)


def test_reindex_isolated_per_meeting(env: Env) -> None:
    env.meeting("m1", "Alpha sync", segments=[seg(0, "A", "alpha topic")])
    env.meeting("m2", "Beta sync", segments=[seg(0, "A", "beta topic")])
    env.reindex("m1")
    env.reindex("m2")
    env.reindex("m1")
    assert {r["meeting_id"] for r in env.rows()} == {"m1", "m2"}
    assert len([r for r in env.rows("m2") if r["kind"] == "title"]) == 1


def test_api_ranks_groups_and_snippets(env: Env) -> None:
    env.meeting(
        "m1", "Budget planning", segments=[seg(0, "A", "hello"), seg(90_000, "B", "the budget is tight")]
    )
    env.meeting(
        "m2", "Offsite", segments=[seg(5_000, "A", "we should discuss the budget once")], start=START + 1000
    )
    env.meeting("m3", "Unrelated", segments=[seg(0, "A", "nothing to see")], start=START + 2000)
    for m in ("m1", "m2", "m3"):
        env.reindex(m)

    body = env.client.get("/api/search", params={"q": "budget"}).json()

    assert body["query"] == "budget"
    ids = [r["meetingId"] for r in body["results"]]
    assert set(ids) == {"m1", "m2"}
    assert ids[0] == "m1"  # title + transcript hit outranks a single transcript mention
    assert body["total"] == 2
    first = body["results"][0]
    assert first["title"] == "Budget planning"
    assert first["startTime"] == START
    transcript_hit = next(h for h in first["hits"] if h["kind"] == "transcript")
    assert transcript_hit["startMs"] == 0  # short segments share one chunk anchored at its first segment
    assert search_svc.MARK_OPEN in transcript_hit["snippet"]
    assert search_svc.MARK_CLOSE in transcript_hit["snippet"]


def test_api_transcript_hit_carries_start_ms(env: Env) -> None:
    env.meeting(
        "m1",
        "Standup",
        segments=[
            seg(0, "A", "short"),
            seg(120_000, "B", "x" * 700),
            seg(125_000, "B", "needle in haystack"),
        ],
    )
    env.reindex("m1")
    hits = env.client.get("/api/search", params={"q": "needle"}).json()["results"][0]["hits"]
    assert hits[0]["kind"] == "transcript"
    assert hits[0]["startMs"] >= 120_000


def test_api_kinds_filter_and_limit(env: Env) -> None:
    for i in range(5):
        env.meeting(f"m{i}", f"Roadmap {i}", segments=[seg(0, "A", "roadmap talk")], start=START + i)
        env.reindex(f"m{i}")
    only_titles = env.client.get("/api/search", params={"q": "roadmap", "kinds": "title"}).json()
    assert all(h["kind"] == "title" for r in only_titles["results"] for h in r["hits"])
    limited = env.client.get("/api/search", params={"q": "roadmap", "limit": 2}).json()
    assert len(limited["results"]) == 2
    assert limited["total"] == 5
    clamped = env.client.get("/api/search", params={"q": "roadmap", "limit": 10_000})
    assert clamped.status_code == 200


def test_moment_hits_deep_link_to_moment_time(env: Env) -> None:
    env.meeting("m1", "Review", segments=[seg(0, "A", "hi")])
    env.moment("m1", "That was a brilliant catch", 42_000)
    env.reindex("m1")
    hit = env.client.get("/api/search", params={"q": "brilliant"}).json()["results"][0]["hits"][0]
    assert hit["kind"] == "moment"
    assert hit["startMs"] == 42_000


@pytest.mark.parametrize(
    "q",
    [
        'foo OR bar"',
        '"',
        '""',
        "AND",
        "NOT",
        "a*",
        "(unbalanced",
        "col:val",
        "NEAR(a b)",
        "-x",
        "' OR 1=1 --",
    ],
)
def test_hostile_queries_do_not_crash(env: Env, q: str) -> None:
    env.meeting("m1", "foo bar", segments=[seg(0, "A", "foo and bar")])
    env.reindex("m1")
    resp = env.client.get("/api/search", params={"q": q})
    assert resp.status_code == 200
    assert isinstance(resp.json()["results"], list)


def test_query_is_treated_as_literal_terms(env: Env) -> None:
    env.meeting("m1", "foo only", segments=[seg(0, "A", "foo")])
    env.meeting("m2", "bar only", segments=[seg(0, "A", "bar")], start=START + 1)
    env.reindex("m1")
    env.reindex("m2")
    # "OR" is a plain word here, not an operator: both terms must be present.
    assert env.client.get("/api/search", params={"q": "foo OR bar"}).json()["results"] == []


def test_blank_query_returns_empty(env: Env) -> None:
    resp = env.client.get("/api/search", params={"q": "   "})
    assert resp.status_code == 200
    assert resp.json() == {"query": "", "results": [], "total": 0}


def test_prefix_match_on_last_term(env: Env) -> None:
    env.meeting("m1", "Migration plan", segments=[seg(0, "A", "hello")])
    env.reindex("m1")
    assert env.client.get("/api/search", params={"q": "migrat"}).json()["total"] == 1


def test_title_edit_reindexes(env: Env) -> None:
    env.meeting("m1", "Old name", segments=[seg(0, "A", "hello")])
    env.reindex("m1")
    update_meeting_title(env.db.engine, "m1", "Fresh quokka title", env.meetings_dir)
    assert env.client.get("/api/search", params={"q": "quokka"}).json()["total"] == 1
    assert env.client.get("/api/search", params={"q": "old"}).json()["total"] == 0


def test_reindex_picks_up_new_content(env: Env) -> None:
    env.meeting("m1", "Sync", segments=[seg(0, "A", "hello")])
    env.reindex("m1")
    assert env.client.get("/api/search", params={"q": "zeppelin"}).json()["total"] == 0
    env.moment("m1", "zeppelin launch was a disaster", 9_000)
    env.reindex("m1")
    assert env.client.get("/api/search", params={"q": "zeppelin"}).json()["total"] == 1


def test_delete_meeting_removes_rows(env: Env) -> None:
    env.meeting("m1", "Gone soon", segments=[seg(0, "A", "ephemeral words")])
    env.meeting("m2", "Stays", segments=[seg(0, "A", "ephemeral words")], start=START + 1)
    env.reindex("m1")
    env.reindex("m2")
    assert delete_meeting(env.db.engine, "m1", env.meetings_dir)
    assert env.rows("m1") == []
    assert env.client.get("/api/search", params={"q": "ephemeral"}).json()["total"] == 1


def test_remove_meeting_is_safe_when_never_indexed(env: Env) -> None:
    search_svc.remove_meeting(env.db.engine, "nope")


def test_backfill_indexes_only_unindexed_and_is_bounded(env: Env) -> None:
    for i in range(4):
        env.meeting(f"m{i}", f"Topic {i}", segments=[seg(0, "A", f"word{i}")], start=START + i)
    env.reindex("m0")
    done = search_svc.backfill(env.db.engine, env.meetings_dir, limit=2)
    assert done == 2
    done = search_svc.backfill(env.db.engine, env.meetings_dir)
    assert done == 1
    assert search_svc.backfill(env.db.engine, env.meetings_dir) == 0
    assert {r["meeting_id"] for r in env.rows()} == {"m0", "m1", "m2", "m3"}


def test_missing_or_corrupt_transcript_still_indexes_title(env: Env) -> None:
    env.meeting("m1", "Broken transcript", segments=[seg(0, "A", "hi")])
    folder = Path(meeting_paths(env.meetings_dir, START, "Broken transcript", "m1").folder)
    (folder / "transcripts/t.json").write_text("{not json")
    assert env.reindex("m1") >= 1
    assert env.client.get("/api/search", params={"q": "broken"}).json()["total"] == 1


def test_long_transcript_is_chunked_with_bounded_rows(env: Env) -> None:
    segs = [seg(i * 1000, "A", f"sentence number {i} about widgets") for i in range(300)]
    env.meeting("m1", "Long one", segments=segs)
    env.reindex("m1")
    chunks = [r for r in env.rows("m1") if r["kind"] == "transcript"]
    assert 1 < len(chunks) < 300
    assert all(len(r["text"]) <= search_svc.CHUNK_CHARS + 200 for r in chunks)
