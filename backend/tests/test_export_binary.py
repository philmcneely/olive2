"""PDF (fpdf2) and DOCX (stdlib zipfile) export."""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from olive.services import export
from test_export import BASIC, make_doc, seg

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _docx_text(data: bytes) -> list[str]:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        root = ET.fromstring(zf.read("word/document.xml"))
    return ["".join(t.text or "" for t in p.iter(f"{W}t")) for p in root.iter(f"{W}p")]


# -- fonts -------------------------------------------------------------------------


def test_bundled_fonts_present() -> None:
    assert (export.FONT_DIR / "DejaVuSans.ttf").is_file()
    assert (export.FONT_DIR / "DejaVuSans-Bold.ttf").is_file()
    assert (export.FONT_DIR / "LICENSE-DejaVu.txt").is_file()


# -- PDF ---------------------------------------------------------------------------


def test_pdf_is_a_pdf() -> None:
    data = export.render(make_doc(BASIC, summary_md="- a\n- b"), "pdf")
    assert data.startswith(b"%PDF-")
    assert data.rstrip().endswith(b"%%EOF")
    assert b"Weekly Sync" in data  # /Title in the (uncompressed) info dict


def test_pdf_unicode_with_bundled_font() -> None:
    doc = make_doc([seg(0, 2000, "Zoë", "Café ☕ naïve — “quotes” Ελληνικά Кириллица")])
    assert export.to_pdf(doc).startswith(b"%PDF-")


def test_pdf_deterministic() -> None:
    doc = make_doc(BASIC)
    assert export.to_pdf(doc) == export.to_pdf(doc)


def test_pdf_missing_font_falls_back_to_core_font(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(export, "FONT_DIR", tmp_path / "nope")
    doc = make_doc([seg(0, 2000, "Zoë", "Café ☕ 你好 — “quotes” 😀")])
    data = export.to_pdf(doc)  # must not raise on non-Latin-1 text
    assert data.startswith(b"%PDF-")


def test_pdf_many_pages_and_unbreakable_word() -> None:
    segs = [seg(i * 1000, i * 1000 + 900, f"S{i % 3}", "lorem ipsum dolor sit amet " * 5) for i in range(600)]
    segs.append(seg(700_000, 701_000, "Z", "x" * 5000))
    data = export.to_pdf(make_doc(segs))
    assert len(re.findall(rb"/Type /Page\b(?!s)", data)) >= 5


def test_pdf_hostile_text_does_not_break_generation() -> None:
    doc = make_doc([seg(0, 1000, "(<A>)", "\\) ) ( stream endstream %%EOF \x00\x07 <script>")])
    assert export.to_pdf(doc).startswith(b"%PDF-")


def test_pdf_empty_doc() -> None:
    assert export.to_pdf(make_doc([])).startswith(b"%PDF-")


# -- DOCX --------------------------------------------------------------------------


def test_docx_is_valid_zip_with_required_parts() -> None:
    data = export.render(make_doc(BASIC), "docx")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        assert zf.testzip() is None
        names = zf.namelist()
        assert names[0] == "[Content_Types].xml"
        assert {"_rels/.rels", "word/document.xml"} <= set(names)
        for n in names:
            ET.fromstring(zf.read(n))  # every part is well-formed XML
        ct = zf.read("[Content_Types].xml").decode()
        assert "wordprocessingml.document.main+xml" in ct
        rels = zf.read("_rels/.rels").decode()
        assert "word/document.xml" in rels


def test_docx_content_has_title_meta_and_turns() -> None:
    paras = _docx_text(export.to_docx(make_doc(BASIC, summary_md="- point")))
    assert paras[0] == "Weekly Sync"
    assert any(p.startswith("Participants: Alice, Bob") for p in paras)
    assert "Summary" in paras and "- point" in paras
    assert any("Alice" in p and "Hello team. Let's begin." in p for p in paras)
    assert any(p.endswith("Sounds good.") and "Bob" in p for p in paras)


def test_docx_timestamps_flag() -> None:
    with_ts = _docx_text(export.to_docx(make_doc(BASIC), timestamps=True))
    without = _docx_text(export.to_docx(make_doc(BASIC), timestamps=False))
    assert any("[00:07]" in p for p in with_ts)
    assert not any("[00:" in p for p in without)


def test_docx_escapes_markup_and_strips_control_chars() -> None:
    doc = make_doc(
        [seg(0, 1000, "A&B", "<script>alert(1)</script> \x00\x01\x0b\x1f ok   & ]]>")], title="T<i>\x08"
    )
    data = export.to_docx(doc)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        raw = zf.read("word/document.xml")
    assert b"<script>" not in raw
    assert not any(b < 0x20 and b not in (9, 10, 13) for b in raw)
    paras = _docx_text(data)
    assert paras[0] == "T<i>"
    assert any("<script>alert(1)</script>" in p and "ok" in p and "A&B" in p for p in paras)


def test_docx_preserves_whitespace_attribute_and_unicode() -> None:
    data = export.to_docx(make_doc([seg(0, 1000, "Zoë", "  spaced   Café ☕ 你好 😀 ")]))
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        assert b'xml:space="preserve"' in zf.read("word/document.xml")
    assert any("Café ☕ 你好 😀" in p for p in _docx_text(data))


def test_docx_deterministic_bytes() -> None:
    doc = make_doc(BASIC)
    assert export.to_docx(doc) == export.to_docx(doc)
    with zipfile.ZipFile(io.BytesIO(export.to_docx(doc))) as zf:
        assert {i.date_time for i in zf.infolist()} == {(1980, 1, 1, 0, 0, 0)}


def test_docx_empty_doc_still_valid() -> None:
    data = export.to_docx(make_doc([]))
    assert _docx_text(data)[0] == "Weekly Sync"
