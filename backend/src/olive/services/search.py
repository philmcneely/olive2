"""Global cross-recording search over an SQLite FTS5 index (local, no LLM).

``search_fts`` holds one row per searchable unit of a meeting:

- ``title``: the meeting title,
- ``transcript``: consecutive transcript segments merged into ~``CHUNK_CHARS``
  chunks, ``start_ms`` = first segment's start (the deep-link target),
- ``summary``: paragraph chunks of the primary summary markdown,
- ``moment``: Olive Insights moments (quote + rationale), ``start_ms`` = moment start.

Indexing is delete-then-insert per meeting, so it is idempotent. ``search_index_state``
records the rowid range each meeting occupies; one write transaction inserts a
meeting's rows contiguously, which lets a reindex delete by rowid range instead
of scanning the FTS table (``meeting_id`` is UNINDEXED).
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from ..layout import meeting_paths
from ..speaker_labels import is_generic_speaker_label

KINDS: tuple[str, ...] = ("title", "transcript", "summary", "moment")

# Control characters wrapped around matches in snippets (never present in prose).
MARK_OPEN = "\u0002"
MARK_CLOSE = "\u0003"

CHUNK_CHARS = 500
# A 5-hour ambient recording is ~100k words; cap what one meeting may add to the index.
MAX_TRANSCRIPT_CHARS = 1_500_000
MAX_SUMMARY_CHARS = 200_000
MAX_QUERY_TOKENS = 12
DEFAULT_LIMIT = 20
MAX_LIMIT = 50
HITS_PER_MEETING = 5
BACKFILL_BATCH = 50

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_COLUMN_TEXT = 3  # position of ``text`` in search_fts, for snippet()

Row = tuple[str, int | None, str]  # (kind, start_ms, text)


def fts_available(conn: Connection) -> bool:
    return conn.execute(text("SELECT 1 FROM sqlite_master WHERE name = 'search_fts'")).first() is not None


def build_match_query(raw: str) -> str | None:
    """Turn free user text into a safe FTS5 MATCH expression.

    Only word characters survive tokenisation, and every token is emitted as a
    quoted FTS string, so FTS operators (OR, NOT, NEAR, ``col:``, parentheses,
    quotes) in user input are matched as plain words and can never alter the
    query structure. All terms must match; the last one matches as a prefix so
    results appear while the user is still typing.
    """
    tokens = _TOKEN_RE.findall(raw)[:MAX_QUERY_TOKENS]
    if not tokens:
        return None
    parts = [f'"{t}"' for t in tokens]
    parts[-1] += "*"
    return " ".join(parts)


def _chunk(pieces: list[tuple[int | None, str]], limit_chars: int) -> list[tuple[int | None, str]]:
    """Merge ``(start_ms, text)`` pieces into chunks of about ``CHUNK_CHARS``."""
    out: list[tuple[int | None, str]] = []
    buf: list[str] = []
    buf_start: int | None = None
    size = 0
    total = 0
    for start, body in pieces:
        body = body.strip()
        if not body:
            continue
        if total + len(body) > limit_chars:
            break
        total += len(body)
        if not buf:
            buf_start = start
        buf.append(body)
        size += len(body) + 1
        if size >= CHUNK_CHARS:
            out.append((buf_start, " ".join(buf)))
            buf, size, buf_start = [], 0, None
    if buf:
        out.append((buf_start, " ".join(buf)))
    return out


def _transcript_rows(db: Engine, meeting: dict[str, Any], meetings_dir: str) -> list[Row]:
    from .census import load_transcript, recording_duration, resolve_transcript_row

    with db.connect() as conn:
        artifact = resolve_transcript_row(conn, meeting)
        duration_ms = recording_duration(conn, meeting["id"])
    if artifact is None:
        return []
    folder = meeting_paths(meetings_dir, int(meeting["start_time"]), meeting["title"], meeting["id"]).folder
    loaded = load_transcript(Path(folder) / artifact["path"], duration_ms)
    if isinstance(loaded, str):
        return []
    _shape, transcript = loaded
    pieces: list[tuple[int | None, str]] = []
    for seg in transcript.segments:
        label = "" if is_generic_speaker_label(seg.speaker) else f"{seg.speaker.strip()}: "
        pieces.append((int(seg.startMs), f"{label}{seg.text}"))
    return [("transcript", start, body) for start, body in _chunk(pieces, MAX_TRANSCRIPT_CHARS)]


def _summary_rows(db: Engine, meeting: dict[str, Any], meetings_dir: str) -> list[Row]:
    with db.connect() as conn:
        art = None
        if meeting.get("primary_summary_artifact_id"):
            art = (
                conn.execute(
                    text("SELECT path FROM artifacts WHERE id = :id AND kind = 'summary'"),
                    {"id": meeting["primary_summary_artifact_id"]},
                )
                .mappings()
                .first()
            )
        if art is None:
            art = (
                conn.execute(
                    text(
                        "SELECT path FROM artifacts WHERE meeting_id = :m AND kind = 'summary' "
                        "ORDER BY created_at DESC LIMIT 1"
                    ),
                    {"m": meeting["id"]},
                )
                .mappings()
                .first()
            )
    if art is None:
        return []
    folder = meeting_paths(meetings_dir, int(meeting["start_time"]), meeting["title"], meeting["id"]).folder
    try:
        body = (Path(folder) / art["path"]).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    paragraphs = [(None, p) for p in re.split(r"\n\s*\n", body[:MAX_SUMMARY_CHARS])]
    return [("summary", None, chunk) for _s, chunk in _chunk(paragraphs, MAX_SUMMARY_CHARS)]


def _moment_rows(db: Engine, meeting_id: str) -> list[Row]:
    with db.connect() as conn:
        moments = (
            conn.execute(
                text(
                    "SELECT start_ms, quote, rationale FROM meeting_moments "
                    "WHERE meeting_id = :m ORDER BY start_ms"
                ),
                {"m": meeting_id},
            )
            .mappings()
            .all()
        )
    rows: list[Row] = []
    for m in moments:
        body = " ".join(p for p in (m["quote"], m["rationale"]) if p and p.strip())
        if body:
            rows.append(("moment", int(m["start_ms"]), body))
    return rows


def remove_rows(conn: Connection, meeting_id: str) -> None:
    state = (
        conn.execute(
            text("SELECT first_rowid, last_rowid FROM search_index_state WHERE meeting_id = :m"),
            {"m": meeting_id},
        )
        .mappings()
        .first()
    )
    if state and state["first_rowid"] is not None:
        conn.execute(
            text("DELETE FROM search_fts WHERE rowid BETWEEN :a AND :b"),
            {"a": state["first_rowid"], "b": state["last_rowid"]},
        )
    conn.execute(text("DELETE FROM search_index_state WHERE meeting_id = :m"), {"m": meeting_id})


def remove_meeting(db: Engine, meeting_id: str) -> None:
    """Drop a meeting's rows from the index (no-op when never indexed)."""
    with db.begin() as conn:
        if fts_available(conn):
            remove_rows(conn, meeting_id)


def reindex_meeting(db: Engine, meeting_id: str, meetings_dir: str) -> int:
    """(Re)build one meeting's index rows from current DB + disk state; return rows written."""
    with db.connect() as conn:
        if not fts_available(conn):
            return 0
        meeting = (
            conn.execute(text("SELECT * FROM meetings WHERE id = :id"), {"id": meeting_id}).mappings().first()
        )
    if meeting is None:
        remove_meeting(db, meeting_id)
        return 0
    meeting_d = dict(meeting)

    # All disk reads happen before the write transaction opens.
    rows: list[Row] = []
    if meeting_d["title"].strip():
        rows.append(("title", None, meeting_d["title"]))
    rows += _transcript_rows(db, meeting_d, meetings_dir)
    rows += _summary_rows(db, meeting_d, meetings_dir)
    rows += _moment_rows(db, meeting_id)

    with db.begin() as conn:
        remove_rows(conn, meeting_id)
        first = last = None
        for kind, start_ms, body in rows:
            res = conn.execute(
                text("INSERT INTO search_fts (meeting_id, kind, start_ms, text) VALUES (:m, :k, :s, :t)"),
                {"m": meeting_id, "k": kind, "s": start_ms, "t": body},
            )
            rowid = res.lastrowid
            first = rowid if first is None else first
            last = rowid
        conn.execute(
            text(
                "INSERT INTO search_index_state (meeting_id, first_rowid, last_rowid, indexed_at) "
                "VALUES (:m, :a, :b, :now)"
            ),
            {"m": meeting_id, "a": first, "b": last, "now": int(time.time() * 1000)},
        )
    return len(rows)


def safe_reindex(db: Engine, meeting_id: str, meetings_dir: str | None) -> None:
    """Reindex hook for write paths: search upkeep must never break the caller."""
    if not meetings_dir:
        return
    try:
        reindex_meeting(db, meeting_id, meetings_dir)
    except Exception as exc:
        from ..logging_db import logger

        logger.warn(
            "Search reindex failed", {"category": "search", "meetingId": meeting_id, "error": str(exc)}
        )


def backfill(db: Engine, meetings_dir: str, limit: int | None = None) -> int:
    """Index meetings that have no index state yet (newest first); return how many were indexed."""
    with db.connect() as conn:
        if not fts_available(conn):
            return 0
        ids = [
            r[0]
            for r in conn.execute(
                text(
                    "SELECT m.id FROM meetings m LEFT JOIN search_index_state s ON s.meeting_id = m.id "
                    "WHERE s.meeting_id IS NULL ORDER BY m.start_time DESC LIMIT :n"
                ),
                {"n": limit if limit is not None else -1},
            ).all()
        ]
    done = 0
    for mid in ids:
        safe_reindex(db, mid, meetings_dir)
        done += 1
    return done


def search(
    db: Engine,
    query: str,
    *,
    limit: int = DEFAULT_LIMIT,
    kinds: list[str] | None = None,
) -> dict[str, Any]:
    """Ranked, per-meeting grouped search. ``snippet`` highlights use MARK_OPEN/MARK_CLOSE."""
    q = query.strip()
    match = build_match_query(q)
    empty: dict[str, Any] = {"query": q, "results": [], "total": 0}
    if match is None:
        return empty
    limit = min(max(int(limit), 1), MAX_LIMIT)
    wanted = [k for k in (kinds or []) if k in KINDS]

    kind_sql = ""
    params: dict[str, Any] = {"q": match, "o": MARK_OPEN, "c": MARK_CLOSE, "cap": limit * 10}
    if wanted:
        names = {f"k{i}": k for i, k in enumerate(wanted)}
        kind_sql = f" AND kind IN ({', '.join(':' + n for n in names)})"
        params.update(names)

    with db.connect() as conn:
        if not fts_available(conn):
            return empty
        hit_rows = (
            conn.execute(
                text(
                    f"SELECT meeting_id, kind, start_ms, bm25(search_fts) AS rank, "
                    f"snippet(search_fts, {_COLUMN_TEXT}, :o, :c, '…', 24) AS snippet "
                    f"FROM search_fts WHERE search_fts MATCH :q{kind_sql} ORDER BY rank LIMIT :cap"
                ),
                params,
            )
            .mappings()
            .all()
        )
        total = conn.execute(
            text(f"SELECT COUNT(DISTINCT meeting_id) FROM search_fts WHERE search_fts MATCH :q{kind_sql}"),
            {k: v for k, v in params.items() if k in ("q", *[f"k{i}" for i in range(len(wanted))])},
        ).scalar_one()

        grouped: dict[str, list[dict[str, Any]]] = {}
        for r in hit_rows:  # already best-rank first, so group order is meeting relevance
            hits = grouped.setdefault(r["meeting_id"], [])
            if len(hits) < HITS_PER_MEETING:
                hits.append({"kind": r["kind"], "snippet": r["snippet"], "startMs": r["start_ms"]})

        ids = list(grouped)[:limit]
        meta: dict[str, Any] = {}
        if ids:
            names = {f"m{i}": mid for i, mid in enumerate(ids)}
            meta = {
                r["id"]: r
                for r in conn.execute(
                    text(
                        f"SELECT id, title, start_time FROM meetings WHERE id IN ({', '.join(':' + n for n in names)})"
                    ),
                    names,
                )
                .mappings()
                .all()
            }

    results = [
        {
            "meetingId": mid,
            "title": meta[mid]["title"],
            "startTime": int(meta[mid]["start_time"]),
            "hits": grouped[mid],
        }
        for mid in ids
        if mid in meta  # drop rows orphaned by a meeting removed outside delete_meeting
    ]
    return {"query": q, "results": results, "total": int(total)}
