"""HTTP tests for Export & Share (/api/meetings/{id}/export, share links, /share/{token})."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from export_support import seed_meeting
from olive.app import create_app
from olive.db import DbHandle, create_db
from olive.settings import clear_app_config_cache


class Env:
    def __init__(self, tmp_path: Path, **opts) -> None:
        clear_app_config_cache()
        self.db: DbHandle = create_db(":memory:")
        self.meetings = str(tmp_path / "meetings")
        self.app = create_app(
            {
                "db": self.db.engine,
                "meetingsDir": self.meetings,
                "configDir": str(tmp_path / "cfg"),
                **opts,
            }
        )
        self.client = TestClient(self.app)

    def seed(self, meeting_id: str = "m1", **kw) -> str:
        return seed_meeting(self.db.engine, self.meetings, meeting_id, **kw)


@pytest.fixture
def env(tmp_path: Path):
    e = Env(tmp_path)
    yield e
    clear_app_config_cache()


# -- GET /api/meetings/{id}/export ----------------------------------------------------


@pytest.mark.parametrize(
    ("fmt", "media_type"),
    [
        ("txt", "text/plain; charset=utf-8"),
        ("md", "text/markdown; charset=utf-8"),
        ("srt", "application/x-subrip"),
        ("vtt", "text/vtt; charset=utf-8"),
        ("pdf", "application/pdf"),
        ("docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    ],
)
def test_export_each_format(env: Env, fmt: str, media_type: str) -> None:
    env.seed()
    res = env.client.get("/api/meetings/m1/export", params={"format": fmt})
    assert res.status_code == 200
    assert res.headers["content-type"] == media_type
    cd = res.headers["content-disposition"]
    assert cd.startswith("attachment;")
    assert f"2023-11-14_Weekly_Sync.{fmt}" in cd
    assert res.headers["x-content-type-options"] == "nosniff"
    assert len(res.content) > 0


def test_export_content_is_real(env: Env) -> None:
    env.seed(summary_md="## Notes\n- a")
    assert b"Hello team." in env.client.get("/api/meetings/m1/export?format=srt").content
    assert env.client.get("/api/meetings/m1/export?format=pdf").content.startswith(b"%PDF-")
    docx = env.client.get("/api/meetings/m1/export?format=docx").content
    assert zipfile.ZipFile(io.BytesIO(docx)).testzip() is None


def test_export_summary_and_timestamps_flags(env: Env) -> None:
    env.seed(summary_md="## Notes\n- a")
    plain = env.client.get("/api/meetings/m1/export?format=md").text
    assert "## Summary" not in plain and "[00:00]" in plain
    with_summary = env.client.get("/api/meetings/m1/export?format=md&summary=1&timestamps=0").text
    assert "## Summary" in with_summary and "- a" in with_summary
    assert "[00:" not in with_summary


def test_export_default_format_is_txt(env: Env) -> None:
    env.seed()
    res = env.client.get("/api/meetings/m1/export")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/plain")


def test_export_errors(env: Env) -> None:
    env.seed("m1")
    env.seed("nt", transcript=None)
    assert env.client.get("/api/meetings/nope/export?format=txt").status_code == 404
    assert env.client.get("/api/meetings/nt/export?format=txt").status_code == 409
    bad = env.client.get("/api/meetings/m1/export?format=exe")
    assert bad.status_code == 400
    assert env.client.get("/api/meetings/nope/export?format=exe").status_code in (400, 404)


def test_export_hostile_title_cannot_inject_headers(env: Env) -> None:
    env.seed(title='evil"\r\nSet-Cookie: pwn=1; <script>')
    res = env.client.get("/api/meetings/m1/export?format=txt")
    assert res.status_code == 200
    assert "set-cookie" not in {k.lower() for k in res.headers}
    cd = res.headers["content-disposition"]
    assert "\r" not in cd and "\n" not in cd
    assert "<" not in cd and '"\r' not in cd


def test_export_unicode_title_filename_star(env: Env) -> None:
    env.seed(title="Café 你好")
    cd = env.client.get("/api/meetings/m1/export?format=txt").headers["content-disposition"]
    assert "filename*=UTF-8''" in cd and "%E4%BD%A0" in cd
    cd.encode("ascii")


def test_export_routes_hidden_from_openapi(env: Env) -> None:
    paths = env.app.openapi()["paths"]
    assert not any(
        ("/meetings/" in p and ("/export" in p or "/share" in p)) or p.startswith("/share") for p in paths
    )
