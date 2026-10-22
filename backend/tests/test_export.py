"""Export formatters: TXT / MD / SRT / VTT (and loader). Binary formats live in test_export_binary.py."""

from __future__ import annotations

import pytest
from export_support import CANONICAL, T0, seed_meeting

from olive.db import create_db
from olive.schemas import TranscriptSegment
from olive.services import export
from olive.services.export import ExportDoc, ExportError
from olive.services.transcript_parsers import parse_srt, parse_vtt


def seg(start: int, end: int, speaker: str, text: str) -> TranscriptSegment:
    return TranscriptSegment(startMs=start, endMs=end, speaker=speaker, text=text)


def make_doc(segments, *, title="Weekly Sync", summary_md=None, duration_ms=600_000) -> ExportDoc:
    speakers: list[str] = []
    for s in segments:
        if s.speaker not in speakers:
            speakers.append(s.speaker)
    return ExportDoc(
        meeting_id="m1",
        title=title,
        start_time_ms=T0,
        duration_ms=duration_ms,
        speakers=speakers,
        segments=list(segments),
        summary_md=summary_md,
    )


def _ms(ts: str) -> int:
    h, m, rest = ts.split(":")
    sec, milli = rest.split(",")
    return ((int(h) * 60 + int(m)) * 60 + int(sec)) * 1000 + int(milli)


BASIC = [
    seg(0, 4000, "Alice", "Hello team."),
    seg(4000, 6000, "Alice", "Let's begin."),
    seg(7000, 9500, "Bob", "Sounds good."),
]


# -- time formatting ----------------------------------------------------------


@pytest.mark.parametrize(
    ("ms", "srt", "vtt"),
    [
        (0, "00:00:00,000", "00:00:00.000"),
        (1, "00:00:00,001", "00:00:00.001"),
        (61_005, "00:01:01,005", "00:01:01.005"),
        (3_723_456, "01:02:03,456", "01:02:03.456"),
        (360_000_000, "100:00:00,000", "100:00:00.000"),
        (-5, "00:00:00,000", "00:00:00.000"),
    ],
)
def test_time_formats(ms: int, srt: str, vtt: str) -> None:
    assert export.format_srt_time(ms) == srt
    assert export.format_vtt_time(ms) == vtt


# -- SRT ----------------------------------------------------------------------


def test_srt_basic_shape() -> None:
    out = export.to_srt(make_doc(BASIC))
    blocks = out.strip().split("\n\n")
    assert len(blocks) == 3
    assert blocks[0].splitlines() == ["1", "00:00:00,000 --> 00:00:04,000", "Alice: Hello team."]
    assert blocks[2].splitlines()[0] == "3"
    assert out.endswith("\n")


def test_srt_round_trips_through_parser() -> None:
    segs = [seg(0, 3000, "Alice", "Hello."), seg(4000, 6000, "Bob", "Hi there."), seg(7000, 8000, "Alice", "Bye.")]
    parsed = parse_srt(export.to_srt(make_doc(segs))).transcript.segments
    assert [(s.speaker, s.text, s.startMs, s.endMs) for s in parsed] == [
        ("Alice", "Hello.", 0, 3000),
        ("Bob", "Hi there.", 4000, 6000),
        ("Alice", "Bye.", 7000, 8000),
    ]


def test_srt_empty_transcript_is_empty_string() -> None:
    assert export.to_srt(make_doc([])) == ""


def test_srt_cue_hygiene_inverted_equal_negative_nonmonotonic() -> None:
    segs = [
        seg(1000, 1000, "A", "equal"),
        seg(5000, 2000, "A", "inverted"),
        seg(-500, 800, "B", "negative start"),
        seg(3000, 4000, "B", "earlier than previous start"),
    ]
    out = export.to_srt(make_doc(segs))
    times = [ln for ln in out.splitlines() if "-->" in ln]
    assert times[0] == "00:00:01,000 --> 00:00:02,000"  # equal -> min 1s
    assert times[1] == "00:00:05,000 --> 00:00:06,000"  # inverted -> min 1s
    starts = [_ms(t.split(" --> ")[0]) for t in times]
    ends = [_ms(t.split(" --> ")[1]) for t in times]
    assert starts == sorted(starts)
    assert all(e > s for s, e in zip(starts, ends, strict=True))
    assert all(s >= 0 for s in starts)


def test_srt_collapses_blank_lines_and_neutralizes_arrow() -> None:
    out = export.to_srt(make_doc([seg(0, 2000, "A", "line one\n\n\nline two --> three")]))
    assert "\n\n" not in out.strip()  # only one cue, so no blank line inside
    assert "-->" not in out.split("\n", 2)[2]
    assert parse_srt(out).transcript.segments[0].text.startswith("line one")


def test_srt_skips_empty_text_cues() -> None:
    out = export.to_srt(make_doc([seg(0, 1000, "A", "   "), seg(1000, 2000, "A", "real")]))
    assert out.count("-->") == 1
    assert out.startswith("1\n")


def test_srt_unicode_roundtrip() -> None:
    out = export.to_srt(make_doc([seg(0, 2000, "Zoë", "Café ☕ 你好 😀")]))
    assert "Zoë: Café ☕ 你好 😀" in out


# -- VTT ----------------------------------------------------------------------


def test_vtt_header_and_voice_tags() -> None:
    out = export.to_vtt(make_doc(BASIC))
    assert out.startswith("WEBVTT\n\n")
    assert "00:00:00.000 --> 00:00:04.000" in out
    assert "<v Alice>Hello team." in out
    assert "," not in out.split("-->")[0].splitlines()[-1]


def test_vtt_escapes_markup_and_arrow() -> None:
    out = export.to_vtt(make_doc([seg(0, 2000, "A <b>", "x < y & z --> w </v>")]))
    body = out.split("\n\n", 1)[1]
    cue_text = body.splitlines()[1]
    assert "&lt;" in cue_text and "&amp;" in cue_text and "&gt;" in cue_text
    assert "-->" not in cue_text
    assert "</v>" not in cue_text.replace("&lt;/v&gt;", "")


def test_vtt_round_trips_through_parser() -> None:
    segs = [seg(0, 3000, "Alice", "Hello."), seg(4000, 6000, "Bob", "Hi there.")]
    parsed = parse_vtt(export.to_vtt(make_doc(segs))).transcript.segments
    assert [(s.speaker, s.text, s.startMs, s.endMs) for s in parsed] == [
        ("Alice", "Hello.", 0, 3000),
        ("Bob", "Hi there.", 4000, 6000),
    ]


def test_vtt_empty_transcript_only_header() -> None:
    assert export.to_vtt(make_doc([])).strip() == "WEBVTT"


# -- TXT / MD -----------------------------------------------------------------


def test_txt_header_and_coalesced_turns() -> None:
    out = export.to_txt(make_doc(BASIC), timestamps=True)
    lines = out.splitlines()
    assert lines[0] == "Weekly Sync"
    assert "Date: 2023-11-14" in out
    assert "Duration: 10:00" in out
    assert "Participants: Alice, Bob" in out
    assert "[00:00] Alice: Hello team. Let's begin." in lines  # coalesced
    assert "[00:07] Bob: Sounds good." in lines


def test_txt_without_timestamps() -> None:
    out = export.to_txt(make_doc(BASIC), timestamps=False)
    assert "Alice: Hello team. Let's begin." in out.splitlines()
    assert "[00:" not in out


def test_txt_hour_long_timestamps() -> None:
    out = export.to_txt(make_doc([seg(3_725_000, 3_730_000, "A", "late")]), timestamps=True)
    assert "[1:02:05] A: late" in out


def test_md_structure_and_summary() -> None:
    out = export.to_markdown(make_doc(BASIC, summary_md="- point one\n- point two"), timestamps=True)
    assert out.startswith("# Weekly Sync\n")
    assert "**Participants:** Alice, Bob" in out
    assert out.index("## Summary") < out.index("## Transcript")
    assert "- point one" in out
    assert "**[00:00] Alice:** Hello team. Let's begin." in out


def test_md_without_summary_has_no_summary_heading() -> None:
    assert "## Summary" not in export.to_markdown(make_doc(BASIC), timestamps=True)


def test_title_newlines_collapsed_in_headers() -> None:
    out = export.to_markdown(make_doc(BASIC, title="Bad\n## Injected"), timestamps=True)
    assert out.splitlines()[0] == "# Bad ## Injected"


def test_unknown_duration_omits_line() -> None:
    out = export.to_txt(make_doc(BASIC, duration_ms=None), timestamps=True)
    assert "Duration:" not in out


# -- render / media / filenames ------------------------------------------------


def test_render_dispatch_and_bad_format() -> None:
    doc = make_doc(BASIC)
    assert export.render(doc, "txt").decode().startswith("Weekly Sync")
    assert export.render(doc, "srt").decode().startswith("1\n")
    with pytest.raises(ExportError) as exc:
        export.render(doc, "exe")  # type: ignore[arg-type]
    assert exc.value.code == "bad_format"


def test_media_types() -> None:
    assert export.EXPORT_MEDIA_TYPES["txt"] == "text/plain; charset=utf-8"
    assert export.EXPORT_MEDIA_TYPES["srt"] == "application/x-subrip"
    assert export.EXPORT_MEDIA_TYPES["vtt"] == "text/vtt"
    assert export.EXPORT_MEDIA_TYPES["pdf"] == "application/pdf"
    assert export.EXPORT_MEDIA_TYPES["docx"].endswith("wordprocessingml.document")
    assert set(export.EXPORT_MEDIA_TYPES) == {"txt", "md", "srt", "vtt", "pdf", "docx"}


def test_export_filename_sanitized() -> None:
    doc = make_doc(BASIC, title='Q4/Plan: "draft"\r\nX')
    name = export.export_filename(doc, "md")
    assert name == "2023-11-14_Q4_Plan_draft_X.md"
    assert export.export_filename(make_doc(BASIC, title="///"), "txt") == "2023-11-14_meeting.txt"


def test_content_disposition_blocks_header_injection_and_encodes_unicode() -> None:
    cd = export.content_disposition('evil"\r\nSet-Cookie: x=1.txt')
    assert "\r" not in cd and "\n" not in cd
    assert cd.startswith("attachment; ")
    cd2 = export.content_disposition("2023-11-14_Café_你好.txt")
    assert 'filename="2023-11-14_Cafe_' in cd2 or 'filename="2023-11-14_Caf' in cd2
    assert "filename*=UTF-8''2023-11-14_Caf%C3%A9_%E4%BD%A0%E5%A5%BD.txt" in cd2
    cd2.encode("ascii")  # header must be pure ASCII


# -- loader -------------------------------------------------------------------


@pytest.fixture
def env(tmp_path):
    handle = create_db(":memory:")
    yield handle.engine, str(tmp_path / "meetings")


def test_load_canonical(env) -> None:
    db, mdir = env
    seed_meeting(db, mdir, "m1", summary_md="## S\nhi")
    doc = export.load_export_doc(db, mdir, "m1")
    assert doc.title == "Weekly Sync"
    assert doc.speakers == ["Alice", "Bob"]
    assert len(doc.segments) == 3
    assert doc.summary_md is None  # not requested
    assert doc.duration_ms == 600_000
    assert export.load_export_doc(db, mdir, "m1", include_summary=True).summary_md == "## S\nhi"


def test_load_bare_plaud_list(env) -> None:
    db, mdir = env
    plaud = [
        {"start_time": 0, "end_time": 2000, "speaker": "Pat", "content": "one"},
        {"start_time": 2000, "end_time": 4000, "speaker": "Quinn", "content": "two"},
    ]
    seed_meeting(db, mdir, "p1", transcript=plaud)
    doc = export.load_export_doc(db, mdir, "p1")
    assert [(s.speaker, s.text) for s in doc.segments] == [("Pat", "one"), ("Quinn", "two")]


def test_load_errors(env) -> None:
    db, mdir = env
    with pytest.raises(ExportError) as e1:
        export.load_export_doc(db, mdir, "nope")
    assert e1.value.code == "not_found"
    seed_meeting(db, mdir, "nt", transcript=None)
    with pytest.raises(ExportError) as e2:
        export.load_export_doc(db, mdir, "nt")
    assert e2.value.code == "no_transcript"
    seed_meeting(db, mdir, "empty", transcript={"segments": []})
    with pytest.raises(ExportError) as e3:
        export.load_export_doc(db, mdir, "empty")
    assert e3.value.code == "no_transcript"
    seed_meeting(db, mdir, "junk", transcript="not a transcript")
    with pytest.raises(ExportError) as e4:
        export.load_export_doc(db, mdir, "junk")
    assert e4.value.code == "no_transcript"


def test_canonical_fixture_constant_is_sane() -> None:
    assert len(CANONICAL["segments"]) == 3


def test_large_transcript_performance() -> None:
    segs = [seg(i * 1000, i * 1000 + 900, f"S{i % 5}", f"word {i}") for i in range(10_000)]
    doc = make_doc(segs)
    assert export.to_srt(doc).count("-->") == 10_000
    assert len(export.to_txt(doc, timestamps=True)) > 10_000
