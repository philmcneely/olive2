"""Hardening: access-log token redaction and bounded heavy renders."""

from __future__ import annotations

import logging
import threading
import time

import pytest

from olive.main import get_uvicorn_log_config
from olive.services import export
from olive.services.export import ExportError
from olive.services.sharing import RedactShareTokens
from test_export import BASIC, make_doc, seg

TOKEN = "Zk3_-abcdEFGH0123456789abcdEFGH0123456789ab"[:43]


def _access_record(path: str) -> logging.LogRecord:
    return logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        __file__,
        1,
        '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:5000", "GET", path, "1.1", 200),
        None,
    )


def test_token_constant_is_43_chars() -> None:
    assert len(TOKEN) == 43


@pytest.mark.parametrize("suffix", ["", "/download?format=pdf"])
def test_filter_redacts_share_tokens(suffix: str) -> None:
    rec = _access_record(f"/share/{TOKEN}{suffix}")
    assert RedactShareTokens().filter(rec) is True  # never drops the record
    line = rec.getMessage()
    assert TOKEN not in line
    assert f"/share/<redacted>{suffix}" in line


def test_filter_leaves_other_paths_alone_and_handles_odd_records() -> None:
    f = RedactShareTokens()
    rec = _access_record("/api/meetings/abc/export?format=pdf")
    f.filter(rec)
    assert "/api/meetings/abc/export?format=pdf" in rec.getMessage()
    plain = logging.LogRecord("x", logging.INFO, __file__, 1, f"GET /share/{TOKEN}", None, None)
    f.filter(plain)
    assert TOKEN not in plain.getMessage()
    assert f.filter(logging.LogRecord("x", logging.INFO, __file__, 1, "msg %s", (None,), None))


def test_uvicorn_access_logger_is_wired_to_the_filter() -> None:
    cfg = get_uvicorn_log_config()
    assert "access" in cfg["handlers"]
    names = cfg["handlers"]["access"]["filters"]
    assert names and all(n in cfg["filters"] for n in names)
    factory = cfg["filters"][names[0]]["()"]
    assert isinstance(factory(), RedactShareTokens)


# -- bounded renders -------------------------------------------------------------------------------


def test_render_caps_raise_too_large(monkeypatch: pytest.MonkeyPatch) -> None:
    doc = make_doc([seg(i * 10, i * 10 + 5, "A", "word") for i in range(50)])
    monkeypatch.setattr(export, "MAX_EXPORT_SEGMENTS", 49)
    for fmt in ("pdf", "docx"):
        with pytest.raises(ExportError) as exc:
            export.render(doc, fmt)  # type: ignore[arg-type]
        assert exc.value.code == "too_large"
    assert export.render(doc, "txt")  # light formats are not capped
    monkeypatch.setattr(export, "MAX_EXPORT_SEGMENTS", 50)
    assert export.render(doc, "docx").startswith(b"PK")


def test_default_caps_are_sane() -> None:
    assert export.MAX_EXPORT_SEGMENTS == 50_000
    assert export.MAX_EXPORT_TEXT_BYTES == 2_000_000


def test_text_byte_cap_counts_utf8_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(export, "MAX_EXPORT_TEXT_BYTES", 10)
    with pytest.raises(ExportError):
        export.render(make_doc([seg(0, 1000, "A", "é" * 8)]), "docx")  # 16 bytes


def test_concurrent_heavy_renders_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(export, "_HEAVY_RENDER", threading.BoundedSemaphore(2))
    lock = threading.Lock()
    state = {"now": 0, "max": 0}

    def slow(_doc, _ts=True):
        with lock:
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
        time.sleep(0.05)
        with lock:
            state["now"] -= 1
        return b"%PDF-"

    monkeypatch.setattr(export, "to_pdf", slow)
    doc = make_doc(BASIC)
    threads = [threading.Thread(target=export.render, args=(doc, "pdf")) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert state["max"] == 2
