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

import io
import json
import logging
import re
import threading
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, get_args
from urllib.parse import quote
from xml.sax.saxutils import escape as xml_escape

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
    "vtt": "text/vtt; charset=utf-8",
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

MIN_CUE_MS = 1000

# PDF/DOCX rendering is CPU/memory heavy and the API is unauthenticated, so
# oversized transcripts are refused and concurrent heavy renders are bounded.
MAX_EXPORT_SEGMENTS = 50_000
MAX_EXPORT_TEXT_BYTES = 2_000_000
_HEAVY_RENDER = threading.BoundedSemaphore(3)
_HEAVY_FORMATS = ("pdf", "docx")

# XML 1.0 / text-hostile characters (keeps \t \n \r): C0 controls, DEL, lone
# surrogates (would crash UTF-8 encoding) and the U+FFFE/U+FFFF non-characters.
_CONTROL_CHARS = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ud800-\udfff\ufffe\uffff]")
FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
_log = logging.getLogger(__name__)
_WS_RUN = re.compile(r"\s+")


class ExportError(Exception):
    """Raised for caller-visible export failures; ``code`` maps to an HTTP status."""

    def __init__(
        self, code: Literal["not_found", "no_transcript", "bad_format", "too_large"], message: str = ""
    ) -> None:
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
        name = one_line(s.speaker)
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


def one_line(value: str) -> str:
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


def clock(ms: int) -> str:
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
        cues.append((start, end, one_line(seg.speaker), text))
        prev_start = start
    return cues


def _turns(doc: ExportDoc) -> list[tuple[int, str, str]]:
    """Coalesced speaker turns as ``(start_ms, speaker, one-line text)``."""
    turns = []
    for seg in coalesce_speaker_segments(doc.segments):
        text = one_line(seg.text)
        if text:
            turns.append((max(0, seg.startMs), one_line(seg.speaker), text))
    return turns


def _meta(doc: ExportDoc) -> list[tuple[str, str]]:
    when = datetime.fromtimestamp(doc.start_time_ms / 1000.0, tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
    rows = [("Date", when)]
    if doc.duration_ms:
        rows.append(("Duration", clock(doc.duration_ms)))
    if doc.speakers:
        rows.append(("Participants", ", ".join(one_line(s) for s in doc.speakers)))
    return rows


def _turn_line(start: int, speaker: str, text: str, timestamps: bool) -> str:
    who = f"{speaker}: " if speaker else ""
    return f"[{clock(start)}] {who}{text}" if timestamps else f"{who}{text}"


# -- text formats ----------------------------------------------------------------


def to_txt(doc: ExportDoc, timestamps: bool = True) -> str:
    out = [one_line(doc.title) or "Meeting"]
    out += [f"{k}: {v}" for k, v in _meta(doc)]
    if doc.summary_md:
        out += ["", "Summary", _strip_control(doc.summary_md)]
    out += ["", "Transcript", ""]
    out += [_turn_line(s, who, text, timestamps) for s, who, text in _turns(doc)]
    return "\n".join(out) + "\n"


def to_markdown(doc: ExportDoc, timestamps: bool = True) -> str:
    out = [f"# {one_line(doc.title) or 'Meeting'}", ""]
    out += [f"- **{k}:** {v}" for k, v in _meta(doc)]
    if doc.summary_md:
        out += ["", "## Summary", "", _strip_control(doc.summary_md)]
    out += ["", "## Transcript", ""]
    for start, who, text in _turns(doc):
        label = f"[{clock(start)}] " if timestamps else ""
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


# -- DOCX (stdlib zipfile, WordprocessingML) -------------------------------------

_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_ZIP_DATE = (1980, 1, 1, 0, 0, 0)

_CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.'
    'wordprocessingml.document.main+xml"/>'
    '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.'
    'wordprocessingml.styles+xml"/>'
    "</Types>"
)
_ROOT_RELS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/'
    'officeDocument" Target="word/document.xml"/>'
    "</Relationships>"
)
_DOC_RELS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/'
    'styles" Target="styles.xml"/>'
    "</Relationships>"
)
_STYLES = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    f'<w:styles xmlns:w="{_W_NS}">'
    '<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:cs="Calibri"/>'
    '<w:sz w:val="22"/></w:rPr></w:rPrDefault></w:docDefaults>'
    '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>'
    '<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/>'
    '<w:rPr><w:b/><w:sz w:val="40"/></w:rPr></w:style>'
    '<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/>'
    '<w:pPr><w:spacing w:before="240" w:after="80"/></w:pPr><w:rPr><w:b/><w:sz w:val="28"/></w:rPr></w:style>'
    "</w:styles>"
)


def _run(text: str, *, bold: bool = False) -> str:
    props = "<w:rPr><w:b/></w:rPr>" if bold else ""
    return f'<w:r>{props}<w:t xml:space="preserve">{xml_escape(_strip_control(text))}</w:t></w:r>'


def _para(*runs: str, style: str | None = None) -> str:
    ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f"<w:p>{ppr}{''.join(runs)}</w:p>"


def to_docx(doc: ExportDoc, timestamps: bool = True) -> bytes:
    paras = [_para(_run(one_line(doc.title) or "Meeting"), style="Title")]
    paras += [_para(_run(f"{k}: {v}")) for k, v in _meta(doc)]
    if doc.summary_md:
        paras.append(_para(_run("Summary"), style="Heading1"))
        paras += [_para(_run(ln)) for ln in _strip_control(doc.summary_md).splitlines() if ln.strip()]
    paras.append(_para(_run("Transcript"), style="Heading1"))
    for start, who, text in _turns(doc):
        label = f"[{clock(start)}] " if timestamps else ""
        head = f"{label}{who}: " if who else label
        paras.append(_para(_run(head, bold=True), _run(text)) if head else _para(_run(text)))
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{_W_NS}"><w:body>{"".join(paras)}'
        '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134" w:header="708" w:footer="708" w:gutter="0"/>'
        "</w:sectPr></w:body></w:document>"
    )
    parts = [
        ("[Content_Types].xml", _CONTENT_TYPES),
        ("_rels/.rels", _ROOT_RELS),
        ("word/document.xml", document),
        ("word/_rels/document.xml.rels", _DOC_RELS),
        ("word/styles.xml", _STYLES),
    ]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, xml in parts:
            info = zipfile.ZipInfo(name, date_time=_ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, xml.encode("utf-8"))
    return buf.getvalue()


# -- PDF (fpdf2) -----------------------------------------------------------------


def to_pdf(doc: ExportDoc, timestamps: bool = True) -> bytes:
    """Render a PDF with the bundled DejaVu Sans (full Latin/Greek/Cyrillic).

    If the font files are missing the built-in Helvetica core font is used and
    all text is transliterated to ASCII with ``unidecode`` (core fonts are
    Latin-1 only and fpdf2 raises on anything else). CJK/emoji need a wider
    font and are out of scope.
    """
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    regular = FONT_DIR / "DejaVuSans.ttf"
    bold = FONT_DIR / "DejaVuSans-Bold.ttf"
    unicode_font = regular.is_file() and bold.is_file()

    def t(value: str) -> str:
        value = _strip_control(value)
        return value if unicode_font else unidecode(value)

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(18, 18, 18)
    pdf.set_creation_date(datetime.fromtimestamp(max(doc.start_time_ms, 0) / 1000.0, tz=UTC))
    pdf.set_producer("Olive")
    pdf.set_title(t(one_line(doc.title)) or "Meeting")
    if unicode_font:
        pdf.add_font("DejaVu", "", str(regular))
        pdf.add_font("DejaVu", "B", str(bold))
        family = "DejaVu"
    else:
        _log.warning("DejaVu fonts missing from %s; PDF export falls back to ASCII core font", FONT_DIR)
        family = "Helvetica"
    pdf.add_page()

    def block(text: str, *, style: str = "", size: int = 10, h: float = 5.5, gap: float = 0) -> None:
        pdf.set_font(family, style, size)
        pdf.multi_cell(0, h, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        if gap:
            pdf.ln(gap)

    block(t(one_line(doc.title)) or "Meeting", style="B", size=17, h=8, gap=2)
    pdf.set_text_color(90, 90, 90)
    for k, v in _meta(doc):
        block(t(f"{k}: {v}"), size=9.5, h=5)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)
    if doc.summary_md:
        block("Summary", style="B", size=13, h=7, gap=1)
        for line in _strip_control(doc.summary_md).splitlines():
            if line.strip():
                block(t(line), size=10)
        pdf.ln(3)
    block("Transcript", style="B", size=13, h=7, gap=1)
    for start, who, text in _turns(doc):
        label = f"[{clock(start)}] " if timestamps else ""
        head = t(f"{label}{who}")
        if head:
            block(head, style="B", size=9.5, h=5)
        block(t(text), size=10, gap=2)
    return bytes(pdf.output())


# -- dispatch & naming -----------------------------------------------------------


def _check_size(doc: ExportDoc) -> None:
    if len(doc.segments) > MAX_EXPORT_SEGMENTS:
        raise ExportError("too_large", "transcript is too large to export in this format")
    total = 0
    for seg in doc.segments:
        total += len(seg.text.encode("utf-8", "replace"))
        if total > MAX_EXPORT_TEXT_BYTES:
            raise ExportError("too_large", "transcript is too large to export in this format")


def render(doc: ExportDoc, fmt: ExportFormat, *, timestamps: bool = True) -> bytes:
    if fmt == "txt":
        return to_txt(doc, timestamps).encode("utf-8")
    if fmt == "md":
        return to_markdown(doc, timestamps).encode("utf-8")
    if fmt == "srt":
        return to_srt(doc).encode("utf-8")
    if fmt == "vtt":
        return to_vtt(doc).encode("utf-8")
    if fmt in _HEAVY_FORMATS:
        _check_size(doc)
        with _HEAVY_RENDER:
            return to_pdf(doc, timestamps) if fmt == "pdf" else to_docx(doc, timestamps)
    raise ExportError("bad_format", f"unsupported format: {fmt!r}")


def export_filename(doc: ExportDoc, fmt: ExportFormat) -> str:
    title = sanitize_filename(one_line(doc.title)) or "meeting"
    return f"{date_stamp(doc.start_time_ms)}_{title}.{fmt}"


def content_disposition(filename: str) -> str:
    """``attachment`` header with an ASCII fallback and an RFC 5987 ``filename*``."""
    ascii_name = re.sub(r"[^A-Za-z0-9._-]", "_", unidecode(filename)) or "export"
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename, safe='')}"
