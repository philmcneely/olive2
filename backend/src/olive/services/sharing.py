"""Share links: scoped, revocable, expiring, read-only transcript URLs.

Security model (read before changing):

* Olive's ``/api/*`` has no authentication, so a share link is NOT an access
  boundary: anyone who can reach the server can already ``GET /api/meetings/{id}``.
  What a link adds is a scoped, revocable, expiring URL with a minimal payload
  that can be handed to someone without telling them the meeting id or the
  rest of the API surface.
* Tokens are ``secrets.token_urlsafe(32)`` (256 bits). Only ``sha256(token)`` is
  stored, so a database leak does not yield working links. The raw token is
  returned exactly once, from ``create``.
* ``resolve`` rejects malformed tokens before touching the database and every
  failure mode (unknown, malformed, expired, revoked, deleted meeting, feature
  off) returns the same ``None`` so the HTTP layer can answer one uniform 404.
* Everything shown publicly goes through the ``PublicMeetingView`` whitelist.
  Transcript text, titles and speaker names are attacker-influenced, so every
  interpolated value in the HTML is escaped; the page has no JavaScript.
"""

from __future__ import annotations

import hashlib
import html
import logging
import math
import os
import re
import secrets
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal
from urllib.parse import quote

from sqlalchemy import text
from sqlalchemy.engine import Engine

from ..schemas import coalesce_speaker_segments
from . import export as export_svc
from .export import ExportDoc

TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{43}\Z")
ENABLED_ENV = "OLIVE_SHARE_ENABLED"
_TRUTHY = {"1", "true", "yes", "on"}
MAX_LABEL_CHARS = 120
MAX_EXPIRES_HOURS = 24 * 365 * 10

SHARE_RESPONSE_HEADERS: dict[str, str] = {
    "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; "
    "frame-ancestors 'none'",
    "Cache-Control": "no-store",
    "Referrer-Policy": "no-referrer",
    "X-Robots-Tag": "noindex",
    "X-Content-Type-Options": "nosniff",
    "Cross-Origin-Resource-Policy": "same-origin",
}


_SHARE_TOKEN_IN_PATH = re.compile(r"(/share/)[A-Za-z0-9_-]{43}")


def redact_share_tokens(value: str) -> str:
    return _SHARE_TOKEN_IN_PATH.sub(r"\1<redacted>", value)


class RedactShareTokens(logging.Filter):
    """Access-log filter: keeps live share tokens out of uvicorn's request lines.

    Rewrites the message and any string args in place and never drops a record.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_share_tokens(record.msg)
        if isinstance(record.args, tuple):
            record.args = tuple(redact_share_tokens(a) if isinstance(a, str) else a for a in record.args)
        return True


def _now_ms() -> int:
    return int(time.time() * 1000)


class ShareDisabledError(Exception):
    """Share links are switched off (``shareLinksEnabled`` / ``OLIVE_SHARE_ENABLED``)."""


def share_links_enabled(config: Any) -> bool:
    """``OLIVE_SHARE_ENABLED`` wins when set (unrecognized values fail closed), else the setting."""
    env = os.environ.get(ENABLED_ENV)
    if env is not None and env.strip():
        return env.strip().lower() in _TRUTHY
    return bool(getattr(config, "shareLinksEnabled", False)) if config is not None else False


@dataclass(frozen=True)
class ShareLink:
    id: str
    meeting_id: str
    label: str | None
    include_summary: bool
    created_at: int
    expires_at: int | None
    revoked_at: int | None
    last_accessed_at: int | None
    access_count: int
    status: Literal["active", "expired", "revoked"]


@dataclass(frozen=True)
class PublicSegment:
    start_ms: int
    speaker: str
    text: str


@dataclass(frozen=True)
class PublicMeetingView:
    """The ONLY data a share page may show (whitelist, not a filtered copy of the meeting)."""

    title: str
    date: str
    duration_ms: int | None
    speakers: list[str]
    segments: list[PublicSegment]
    summary_md: str | None


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _status(row: Any, now: int) -> Literal["active", "expired", "revoked"]:
    if row["revoked_at"] is not None:
        return "revoked"
    if row["expires_at"] is not None and row["expires_at"] <= now:
        return "expired"
    return "active"


class ShareService:
    def __init__(
        self,
        db: Engine,
        meetings_dir: str,
        *,
        now_ms: Callable[[], int] = _now_ms,
        is_enabled: Callable[[], bool] | None = None,
    ) -> None:
        self.db = db
        self.meetings_dir = meetings_dir
        self._now = now_ms
        # Fail closed: without an injected check only the env var can enable sharing.
        self._is_enabled = is_enabled or (lambda: share_links_enabled(None))

    def _to_link(self, row: Any, now: int) -> ShareLink:
        return ShareLink(
            id=row["id"],
            meeting_id=row["meeting_id"],
            label=row["label"],
            include_summary=bool(row["include_summary"]),
            created_at=int(row["created_at"]),
            expires_at=row["expires_at"],
            revoked_at=row["revoked_at"],
            last_accessed_at=row["last_accessed_at"],
            access_count=int(row["access_count"]),
            status=_status(row, now),
        )

    def create(
        self,
        meeting_id: str,
        *,
        expires_in_hours: float | None,
        label: str | None = None,
        include_summary: bool = False,
    ) -> tuple[ShareLink, str]:
        """Create a link; returns ``(link, raw_token)``. The raw token is never retrievable again."""
        if not self._is_enabled():
            raise ShareDisabledError("share links are disabled")
        if expires_in_hours is not None and not (
            isinstance(expires_in_hours, int | float)
            and math.isfinite(expires_in_hours)
            and 0 < expires_in_hours <= MAX_EXPIRES_HOURS
        ):
            raise ValueError("expiresInHours must be a positive number (or null for no expiry)")
        # Validates existence + a readable transcript, so links are never dead on arrival.
        export_svc.load_export_doc(self.db, self.meetings_dir, meeting_id)

        clean_label = export_svc.one_line(label or "")[:MAX_LABEL_CHARS] or None
        now = self._now()
        token = secrets.token_urlsafe(32)
        row = {
            "id": str(uuid.uuid4()),
            "meeting_id": meeting_id,
            "token_hash": hash_token(token),
            "label": clean_label,
            "include_summary": 1 if include_summary else 0,
            "created_at": now,
            "expires_at": None if expires_in_hours is None else now + int(expires_in_hours * 3_600_000),
            "revoked_at": None,
            "last_accessed_at": None,
            "access_count": 0,
        }
        with self.db.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO meeting_share_links (id, meeting_id, token_hash, label, include_summary, "
                    "created_at, expires_at) VALUES (:id, :meeting_id, :token_hash, :label, "
                    ":include_summary, :created_at, :expires_at)"
                ),
                row,
            )
        return self._to_link(row, now), token

    def list_for_meeting(self, meeting_id: str) -> list[ShareLink]:
        now = self._now()
        with self.db.connect() as conn:
            rows = (
                conn.execute(
                    text(
                        "SELECT * FROM meeting_share_links WHERE meeting_id = :m "
                        "ORDER BY created_at DESC, id ASC"
                    ),
                    {"m": meeting_id},
                )
                .mappings()
                .all()
            )
        return [self._to_link(r, now) for r in rows]

    def revoke(self, meeting_id: str, link_id: str) -> bool:
        """Soft-delete. Idempotent: already-revoked links stay revoked with their original timestamp."""
        with self.db.begin() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM meeting_share_links WHERE id = :id AND meeting_id = :m"),
                {"id": link_id, "m": meeting_id},
            ).first()
            if not exists:
                return False
            conn.execute(
                text(
                    "UPDATE meeting_share_links SET revoked_at = :now WHERE id = :id AND revoked_at IS NULL"
                ),
                {"now": self._now(), "id": link_id},
            )
        return True

    def resolve(self, token: str | None) -> ShareLink | None:
        """Return the live link for ``token`` (recording the access), else ``None`` for any failure."""
        if not isinstance(token, str) or not TOKEN_RE.match(token):
            return None
        if not self._is_enabled():
            return None
        now = self._now()
        with self.db.begin() as conn:
            row = (
                conn.execute(
                    text(
                        "SELECT l.* FROM meeting_share_links l JOIN meetings m ON m.id = l.meeting_id "
                        "WHERE l.token_hash = :h AND l.revoked_at IS NULL "
                        "AND (l.expires_at IS NULL OR l.expires_at > :now)"
                    ),
                    {"h": hash_token(token), "now": now},
                )
                .mappings()
                .first()
            )
            if row is None:
                return None
            conn.execute(
                text(
                    "UPDATE meeting_share_links SET last_accessed_at = :now, access_count = access_count + 1 "
                    "WHERE id = :id"
                ),
                {"now": now, "id": row["id"]},
            )
            updated = (
                conn.execute(text("SELECT * FROM meeting_share_links WHERE id = :id"), {"id": row["id"]})
                .mappings()
                .one()
            )
        return self._to_link(updated, now)

    def export_doc(self, link: ShareLink) -> ExportDoc:
        """The export document for a resolved link (summary only if the link opted in)."""
        return export_svc.load_export_doc(
            self.db, self.meetings_dir, link.meeting_id, include_summary=link.include_summary
        )

    def public_view(self, link: ShareLink) -> PublicMeetingView:
        doc = self.export_doc(link)
        segments = [
            PublicSegment(
                start_ms=max(0, s.startMs), speaker=export_svc.one_line(s.speaker), text=s.text.strip()
            )
            for s in coalesce_speaker_segments(doc.segments)
            if s.text.strip()
        ]
        return PublicMeetingView(
            title=export_svc.one_line(doc.title) or "Meeting",
            date=datetime.fromtimestamp(doc.start_time_ms / 1000.0, tz=UTC).strftime("%Y-%m-%d"),
            duration_ms=doc.duration_ms,
            speakers=list(doc.speakers),
            segments=segments,
            summary_md=doc.summary_md,
        )


_PAGE_CSS = """
:root{color-scheme:light dark}
body{font:16px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;max-width:46rem;margin:0 auto;padding:2rem 1.25rem 4rem;background:Canvas;color:CanvasText}
h1{font-size:1.6rem;margin:0 0 .25rem}
.meta{opacity:.7;font-size:.9rem;margin:0 0 1.25rem}
h2{font-size:1.05rem;margin:2rem 0 .5rem;text-transform:uppercase;letter-spacing:.06em;opacity:.75}
.summary{white-space:pre-wrap;border-left:3px solid currentColor;padding:.25rem 0 .25rem 1rem}
.turn{margin:0 0 1rem}
.who{font-weight:650}
.ts{font:.8rem ui-monospace,monospace;opacity:.6;margin-right:.5rem}
.dl{display:flex;flex-wrap:wrap;gap:.5rem;margin:1rem 0}
.dl a{border:1px solid currentColor;border-radius:.4rem;padding:.2rem .7rem;text-decoration:none;color:inherit;font-size:.9rem}
.foot{margin-top:3rem;font-size:.8rem;opacity:.55}
""".strip()


def render_share_html(view: PublicMeetingView, token: str) -> str:
    """Server-rendered read-only page. No JS; every interpolated value is HTML-escaped."""

    def esc(value: object) -> str:
        return html.escape(str(value), quote=True)

    meta = [view.date]
    if view.duration_ms:
        meta.append(export_svc.clock(view.duration_ms))
    if view.speakers:
        meta.append(", ".join(view.speakers))
    base = f"/share/{quote(token, safe='')}/download?format="
    downloads = "".join(
        f'<a href="{esc(base + f)}">{esc(f.upper())}</a>' for f in ("pdf", "docx", "txt", "md", "srt", "vtt")
    )
    turns = "".join(
        f'<p class="turn"><span class="ts">{esc(export_svc.clock(s.start_ms))}</span>'
        f'<span class="who">{esc(s.speaker)}</span><br>{esc(s.text)}</p>'
        for s in view.segments
    )
    summary = f'<h2>Summary</h2><div class="summary">{esc(view.summary_md)}</div>' if view.summary_md else ""
    return (
        "<!doctype html>"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="robots" content="noindex, nofollow">'
        '<meta name="referrer" content="no-referrer">'
        f"<title>{esc(view.title)}</title><style>{_PAGE_CSS}</style></head><body>"
        f"<h1>{esc(view.title)}</h1>"
        f'<p class="meta">{esc(" · ".join(meta))}</p>'
        f'<div class="dl">{downloads}</div>'
        f"{summary}<h2>Transcript</h2>{turns}"
        '<p class="foot">Shared read-only from Olive.</p>'
        "</body></html>"
    )
