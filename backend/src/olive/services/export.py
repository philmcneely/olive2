"""Transcript export (TXT / MD / SRT / VTT / PDF / DOCX).

Pure, local-only rendering: no network, no temp files. ``load_export_doc``
reads one meeting through the existing ``get_meeting`` aggregate and
normalizes every stored transcript shape via ``_parse_stored_transcript``;
``render`` turns the resulting :class:`ExportDoc` into bytes.

Transcript text, titles and speaker names are attacker-influenced (they come
from ``/api/ingest`` and transcript import), so every format either escapes
markup (VTT, DOCX) or strips control characters / collapses structure
(everything else).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, get_args
from urllib.parse import quote

from sqlalchemy.engine import Engine
from unidecode import unidecode

from ..layout import date_stamp, sanitize_filename
from ..schemas import TranscriptSegment, coalesce_speaker_segments

ExportFormat = Literal["txt", "md", "srt", "vtt", "pdf", "docx"]
EXPORT_FORMATS: tuple[str, ...] = get_args(ExportFormat)

EXPORT_MEDIA_TYPES: dict[ExportFormat, str] = {
    "txt": "text/plain; charset=utf-8",
    "md": "text/markdown; charset=utf-8",
    "srt": "application/x-subrip",
    "vtt": "text/vtt",
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

MIN_CUE_MS = 1000

# XML 1.0 / text-hostile control characters (keeps \t \n \r).
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS_RUN = re.compile(r"\s+")


class ExportError(Exception):
    """Raised for caller-visible export failures; ``code`` maps to an HTTP status."""

    def __init__(self, code: Literal["not_found", "no_transcript", "bad_format"], message: str = "") -> None:
        super().__init__(message or code)
        self.code = code


@dataclass(frozen=True)
class ExportDoc:
    meeting_id: str
    title: str
    start_time_ms: int
    duration_ms: int | None
    speakers: list[str]
    segments: list[TranscriptSegment]
    summary_md: str | None = None


# -- loading ---------------------------------------------------------------------


def load_export_doc(
    db: Engine,
    meetings_dir: str,
    meeting_id: str,
    *,
    include_summary: bool = False,
) -> ExportDoc:
    from .meetings import get_meeting
    from .transcription import _parse_stored_transcript

    detail = get_meeting(
        db,
        meeting_id,
        meetings_dir,
        {"includeTranscript": True, "includeSummaries": include_summary},
    )
    if detail is None:
        raise ExportError("not_found", "meeting not found")
    meeting = detail.meeting
    wall_ms = meeting.endTime - meeting.startTime
    raw_content = detail.transcriptContent
    if not raw_content:
        raise ExportError("no_transcript", "meeting has no transcript")
    try:
        transcript = _parse_stored_transcript(json.loads(raw_content), wall_ms if wall_ms > 0 else None)
    except Exception:
        raise ExportError("no_transcript", "transcript could not be parsed") from None
    segments = [s for s in transcript.segments if s.text.strip()]
    if not segments:
        raise ExportError("no_transcript", "transcript has no segments")

    duration_ms = transcript.durationMs or (wall_ms if wall_ms > 0 else None)
    speakers: list[str] = []
    for s in segments:
        name = _one_line(s.speaker)
        if name and name not in speakers:
            speakers.append(name)
    summary = detail.summaryContent if include_summary else None
    return ExportDoc(
        meeting_id=meeting.id,
        title=meeting.title,
        start_time_ms=meeting.startTime,
        duration_ms=duration_ms,
        speakers=speakers,
        segments=segments,
        summary_md=summary.strip() if summary and summary.strip() else None,
    )


# -- helpers ---------------------------------------------------------------------


def _strip_control(value: str) -> str:
    return _CONTROL_CHARS.sub("", value)


def _one_line(value: str) -> str:
    return _WS_RUN.sub(" ", _strip_control(value or "")).strip()


def _cue_text(value: str) -> str:
    """Multi-line text with blank lines (cue terminators) removed."""
    lines = (
        ln.strip() for ln in _strip_control(value or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    )
    return "\n".join(ln for ln in lines if ln)


def _stamp(ms: int, sep: str) -> str:
    ms = max(0, int(ms))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, milli = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{milli:03d}"


def format_srt_time(ms: int) -> str:
    return _stamp(ms, ",")


def format_vtt_time(ms: int) -> str:
    return _stamp(ms, ".")


def _clock(ms: int) -> str:
    """Human timestamp: ``mm:ss`` below an hour, ``h:mm:ss`` from an hour up."""
    total = max(0, int(ms)) // 1000
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _cues(segments: list[TranscriptSegment]) -> list[tuple[int, int, str, str]]:
    """Clean raw segments into ``(start, end, speaker, text)`` cues.

    Negative times clamp to 0, starts never go backwards, an end at or before
    its start becomes start + 1s, and empty-text segments are dropped.
    """
    cues: list[tuple[int, int, str, str]] = []
    prev_start = 0
    for seg in segments:
        text = _cue_text(seg.text)
        if not text:
            continue
        start = max(0, seg.startMs, prev_start)
        end = seg.endMs if seg.endMs > start else start + MIN_CUE_MS
        cues.append((start, end, _one_line(seg.speaker), text))
        prev_start = start
    return cues


def _turns(doc: ExportDoc) -> list[tuple[int, str, str]]:
    """Coalesced speaker turns as ``(start_ms, speaker, one-line text)``."""
    turns = []
    for seg in coalesce_speaker_segments(doc.segments):
        text = _one_line(seg.text)
        if text:
            turns.append((max(0, seg.startMs), _one_line(seg.speaker), text))
    return turns


def _meta(doc: ExportDoc) -> list[tuple[str, str]]:
    when = datetime.fromtimestamp(doc.start_time_ms / 1000.0, tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
    rows = [("Date", when)]
    if doc.duration_ms:
        rows.append(("Duration", _clock(doc.duration_ms)))
    if doc.speakers:
        rows.append(("Participants", ", ".join(_one_line(s) for s in doc.speakers)))
    return rows


def _turn_line(start: int, speaker: str, text: str, timestamps: bool) -> str:
    who = f"{speaker}: " if speaker else ""
    return f"[{_clock(start)}] {who}{text}" if timestamps else f"{who}{text}"


# -- text formats ----------------------------------------------------------------


def to_txt(doc: ExportDoc, timestamps: bool = True) -> str:
    out = [_one_line(doc.title) or "Meeting"]
    out += [f"{k}: {v}" for k, v in _meta(doc)]
    if doc.summary_md:
        out += ["", "Summary", _strip_control(doc.summary_md)]
    out += ["", "Transcript", ""]
    out += [_turn_line(s, who, text, timestamps) for s, who, text in _turns(doc)]
    return "\n".join(out) + "\n"


def to_markdown(doc: ExportDoc, timestamps: bool = True) -> str:
    out = [f"# {_one_line(doc.title) or 'Meeting'}", ""]
    out += [f"- **{k}:** {v}" for k, v in _meta(doc)]
    if doc.summary_md:
        out += ["", "## Summary", "", _strip_control(doc.summary_md)]
    out += ["", "## Transcript", ""]
    for start, who, text in _turns(doc):
        label = f"[{_clock(start)}] " if timestamps else ""
        head = f"**{label}{who}:**" if who else (f"**{label.strip()}**" if label else "")
        out.append(f"{head} {text}".strip())
        out.append("")
    return "\n".join(out).rstrip("\n") + "\n"


def to_srt(doc: ExportDoc) -> str:
    blocks = []
    for i, (start, end, who, text) in enumerate(_cues(doc.segments), start=1):
        body = text.replace("-->", "->")
        if who:
            body = f"{who.replace('-->', '->')}: {body}"
        blocks.append(f"{i}\n{format_srt_time(start)} --> {format_srt_time(end)}\n{body}\n")
    return "\n".join(blocks)


def _vtt_escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def to_vtt(doc: ExportDoc) -> str:
    blocks = ["WEBVTT\n"]
    for start, end, who, text in _cues(doc.segments):
        body = _vtt_escape(text)
        if who:
            body = f"<v {_vtt_escape(who)}>{body}"
        blocks.append(f"{format_vtt_time(start)} --> {format_vtt_time(end)}\n{body}\n")
    return "\n".join(blocks)


# -- dispatch & naming -----------------------------------------------------------


def render(doc: ExportDoc, fmt: ExportFormat, *, timestamps: bool = True) -> bytes:
    if fmt == "txt":
        return to_txt(doc, timestamps).encode("utf-8")
    if fmt == "md":
        return to_markdown(doc, timestamps).encode("utf-8")
    if fmt == "srt":
        return to_srt(doc).encode("utf-8")
    if fmt == "vtt":
        return to_vtt(doc).encode("utf-8")
    raise ExportError("bad_format", f"unsupported format: {fmt!r}")


def export_filename(doc: ExportDoc, fmt: ExportFormat) -> str:
    title = sanitize_filename(_one_line(doc.title)) or "meeting"
    return f"{date_stamp(doc.start_time_ms)}_{title}.{fmt}"


def content_disposition(filename: str) -> str:
    """``attachment`` header with an ASCII fallback and an RFC 5987 ``filename*``."""
    ascii_name = re.sub(r"[^A-Za-z0-9._-]", "_", unidecode(filename)) or "export"
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename, safe='')}"
