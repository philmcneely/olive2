"""HTTP tests for Export & Share (/api/meetings/{id}/export, share links, /share/{token})."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from export_support import T0, seed_meeting
from olive.app import create_app
from olive.db import DbHandle, create_db
from olive.settings import AppConfig, clear_app_config_cache, resolve_paths, save_app_config


class Clock:
    def __init__(self, now: int = T0) -> None:
        self.now = now

    def __call__(self) -> int:
        return self.now


class Env:
    def __init__(self, tmp_path: Path, **opts) -> None:
        clear_app_config_cache()
        self.db: DbHandle = create_db(":memory:")
        self.meetings = str(tmp_path / "meetings")
        self.paths = resolve_paths({"OLIVE_CONFIG_DIR": str(tmp_path / "cfg")})
        self.clock = Clock()
        self.app = create_app(
            {
                "db": self.db.engine,
                "meetingsDir": self.meetings,
                "configDir": str(tmp_path / "cfg"),
                "shareNowMs": self.clock,
                **opts,
            }
        )
        self.client = TestClient(self.app)

    def set_config(self, **fields) -> None:
        save_app_config(AppConfig(**fields), self.paths)

    def enable(self, **fields) -> None:
        self.set_config(shareLinksEnabled=True, **fields)

    def share(self, meeting_id: str = "m1", **body):
        return self.client.post(f"/api/meetings/{meeting_id}/share", json=body)

    def seed(self, meeting_id: str = "m1", **kw) -> str:
        return seed_meeting(self.db.engine, self.meetings, meeting_id, **kw)


@pytest.fixture(autouse=True)
def _no_share_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLIVE_SHARE_ENABLED", raising=False)


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


# -- share links: create / list / revoke --------------------------------------------------------

HOUR = 3_600_000
NOT_FOUND = {"status": 404}


def _token_from(res) -> str:
    return res.json()["token"]


def test_share_create_forbidden_when_flag_off(env: Env) -> None:
    env.seed()
    res = env.share()
    assert res.status_code == 403
    assert "token" not in res.text


def test_share_env_var_enables_without_setting(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    env.seed()
    monkeypatch.setenv("OLIVE_SHARE_ENABLED", "1")
    assert env.share().status_code == 201
    monkeypatch.setenv("OLIVE_SHARE_ENABLED", "0")
    env.enable()
    assert env.share().status_code == 403  # env kill-switch beats the setting


def test_share_create_response_shape(env: Env) -> None:
    env.seed()
    env.enable()
    res = env.share(expiresInHours=2, label="For Bob", includeSummary=True)
    assert res.status_code == 201
    body = res.json()
    assert body["path"] == f"/share/{body['token']}" and len(body["token"]) == 43
    assert body["link"]["label"] == "For Bob"
    assert body["link"]["includeSummary"] is True
    assert body["link"]["expiresAt"] == T0 + 2 * HOUR
    assert body["link"]["status"] == "active"
    assert "token" not in body["link"] and "tokenHash" not in body["link"]
    assert res.headers["cache-control"] == "no-store"
    assert "http" not in body["path"]  # server never builds absolute URLs


def test_share_default_ttl_and_never_expire(tmp_path: Path) -> None:
    e = Env(tmp_path)
    e.seed()
    e.enable()
    assert e.share().json()["link"]["expiresAt"] == T0 + 168 * HOUR  # default ttl
    assert e.share(expiresInHours=None).json()["link"]["expiresAt"] is None  # explicit null
    e.enable(shareDefaultTtlHours=0)
    assert e.share().json()["link"]["expiresAt"] is None  # 0 = never
    e.enable(shareDefaultTtlHours=6)
    assert e.share().json()["link"]["expiresAt"] == T0 + 6 * HOUR


def test_share_create_errors(env: Env) -> None:
    env.seed("m1")
    env.seed("nt", transcript=None)
    env.enable()
    assert env.share("ghost").status_code == 404
    assert env.share("nt").status_code == 409
    for bad in (
        {"expiresInHours": 0},
        {"expiresInHours": -3},
        {"expiresInHours": "soon"},
        {"expiresInHours": True},
    ):
        assert env.share(**bad).status_code == 400, bad
    assert env.share(label=123).status_code == 400
    assert env.share(includeSummary="yes").status_code == 400
    assert env.client.post("/api/meetings/m1/share", content=b"not json").status_code in (400, 422)
    assert env.client.post("/api/meetings/m1/share").status_code == 201  # empty body = defaults


def test_share_list_has_no_tokens_and_revoke_flow(env: Env) -> None:
    env.seed()
    env.enable()
    first = env.share(label="a").json()
    env.share(label="b")
    res = env.client.get("/api/meetings/m1/share")
    assert res.status_code == 200
    data = res.json()
    assert data["enabled"] is True and len(data["links"]) == 2
    assert first["token"] not in res.text
    assert env.client.get("/api/meetings/ghost/share").status_code == 404

    link_id = first["link"]["id"]
    assert env.client.delete(f"/api/meetings/m1/share/{link_id}").status_code == 200
    assert env.client.delete(f"/api/meetings/m1/share/{link_id}").status_code == 200  # idempotent
    assert env.client.delete("/api/meetings/m1/share/nope").status_code == 404
    env.seed("m2")
    assert env.client.delete(f"/api/meetings/m2/share/{link_id}").status_code == 404
    statuses = {
        link["label"]: link["status"] for link in env.client.get("/api/meetings/m1/share").json()["links"]
    }
    assert statuses == {"a": "revoked", "b": "active"}


def test_share_list_reports_flag_off(env: Env) -> None:
    env.seed()
    assert env.client.get("/api/meetings/m1/share").json() == {"enabled": False, "links": []}


# -- public pages ------------------------------------------------------------------------------


def _assert_share_headers(res) -> None:
    h = res.headers
    csp = h["content-security-policy"]
    assert "default-src 'none'" in csp and "frame-ancestors 'none'" in csp and "base-uri 'none'" in csp
    assert h["cache-control"] == "no-store"
    assert h["referrer-policy"] == "no-referrer"
    assert h["x-robots-tag"] == "noindex"
    assert h["x-content-type-options"] == "nosniff"
    assert "form-action 'none'" in csp
    assert h["cross-origin-resource-policy"] == "same-origin"


def test_public_page_content_and_headers(env: Env) -> None:
    env.seed(summary_md="## Decisions\n- ship it", tags=["internal-tag"])
    env.enable()
    token = _token_from(env.share())
    res = env.client.get(f"/share/{token}")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/html")
    _assert_share_headers(res)
    page = res.text
    assert "Weekly Sync" in page and "Hello team. Let&#x27;s begin." in page and "Bob" in page
    assert "ship it" not in page  # includeSummary defaults off
    assert "internal-tag" not in page and "m1" not in page.replace(token, "")
    assert "<script" not in page.lower()
    assert f"/share/{token}/download?format=pdf" in page

    with_summary = _token_from(env.share(includeSummary=True))
    assert "ship it" in env.client.get(f"/share/{with_summary}").text


def test_public_page_escapes_hostile_content(env: Env) -> None:
    env.seed(
        title="<script>alert(1)</script>",
        transcript={
            "segments": [
                {"startMs": 0, "endMs": 1000, "speaker": "<b>x</b>", "text": "<img src=x onerror=alert(1)>"}
            ]
        },
    )
    env.enable()
    page = env.client.get(f"/share/{_token_from(env.share())}").text
    assert "<script" not in page.lower() and "<img" not in page.lower() and "<b>" not in page


def test_public_download_formats_and_audio_never_exposed(env: Env) -> None:
    env.seed(title='Q4 "plan"\r\nX')
    env.enable()
    token = _token_from(env.share())
    pdf = env.client.get(f"/share/{token}/download?format=pdf")
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF-")
    assert pdf.headers["content-type"] == "application/pdf"
    assert pdf.headers["content-disposition"].startswith("attachment;")
    _assert_share_headers(pdf)
    docx = env.client.get(f"/share/{token}/download?format=docx")
    assert zipfile.ZipFile(io.BytesIO(docx.content)).testzip() is None
    assert b"Hello team." in env.client.get(f"/share/{token}/download?format=srt").content
    bad_fmt = env.client.get(f"/share/{token}/download?format=exe")
    assert bad_fmt.status_code == 404  # uniform 404, not a distinguishable 400
    assert bad_fmt.text == env.client.get(f"/share/{'A' * 43}/download?format=pdf").text
    _assert_share_headers(bad_fmt)
    assert env.client.get(f"/share/{token}/download").status_code == 200  # default txt
    for audio_path in ("audio", "recording", "recordings/r1"):
        assert env.client.get(f"/share/{token}/{audio_path}").status_code == 404


def test_public_download_respects_summary_optin(env: Env) -> None:
    env.seed(summary_md="- secret decision")
    env.enable()
    plain = _token_from(env.share())
    opted = _token_from(env.share(includeSummary=True))
    assert b"secret decision" not in env.client.get(f"/share/{plain}/download?format=md&summary=1").content
    assert b"secret decision" in env.client.get(f"/share/{opted}/download?format=md").content


def test_all_failure_modes_return_identical_404(env: Env) -> None:
    env.seed("m1")
    env.seed("m2")
    env.enable()
    live = env.share("m1")
    revoked = env.share("m1")
    env.client.delete(f"/api/meetings/m1/share/{revoked.json()['link']['id']}")
    expiring = _token_from(env.share("m1", expiresInHours=1))
    doomed = _token_from(env.share("m2"))
    from olive.services.meetings import delete_meeting

    delete_meeting(env.db.engine, "m2", env.meetings)
    env.clock.now = T0 + 2 * HOUR

    # Control: the live default-TTL link (168h) still works.
    assert env.client.get(f"/share/{_token_from(live)}").status_code == 200

    bad = {
        "malformed": "not-a-token",
        "unknown": "A" * 43,
        "revoked": _token_from(revoked),
        "expired": expiring,
        "deleted meeting": doomed,
        "trailing newline": "A" * 43 + "%0A",
        "sqlish": "%27%20OR%201%3D1%20--" + "A" * 30,
    }
    seen = {}
    for name, tok in bad.items():
        for suffix in ("", "/download?format=pdf"):
            res = env.client.get(f"/share/{tok}{suffix}")
            seen[(name, suffix)] = (
                res.status_code,
                res.text,
                res.headers["content-type"],
                res.headers["cache-control"],
            )
    assert {v[0] for v in seen.values()} == {404}
    assert len({v[1:] for k, v in seen.items() if k[1] == ""}) == 1
    assert len({v[1:] for k, v in seen.items() if k[1] != ""}) == 1
    for tok in bad.values():
        _assert_share_headers(env.client.get(f"/share/{tok}"))

    # flag off is indistinguishable too
    env.set_config(shareLinksEnabled=False)
    res = env.client.get(f"/share/{_token_from(live)}")
    assert res.status_code == 404 and res.text == seen[("unknown", "")][1]


def test_share_route_not_shadowed_by_spa_fallback(tmp_path: Path) -> None:
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("<html>SPA SHELL</html>")
    e = Env(tmp_path, webRoot=str(web))
    e.seed()
    e.enable()
    token = _token_from(e.share())
    assert "SPA SHELL" not in e.client.get(f"/share/{token}").text
    missing = e.client.get(f"/share/{'B' * 43}")
    assert missing.status_code == 404 and "SPA SHELL" not in missing.text
    assert "SPA SHELL" in e.client.get("/some/spa/route").text  # catch-all still works


def test_public_routes_do_not_need_api_prefix_or_leak_ids(env: Env) -> None:
    env.seed()
    env.enable()
    token = _token_from(env.share())
    res = env.client.get(f"/share/{token}/download?format=txt")
    assert "m1" not in res.headers["content-disposition"]


# -- end-to-end lifecycle -----------------------------------------------------------------------


def test_e2e_share_lifecycle(env: Env) -> None:
    env.seed(summary_md="- point")
    # 1. flag off: nothing works
    assert env.share().status_code == 403
    # 2. enable, create
    env.enable()
    created = env.share(expiresInHours=1, includeSummary=True)
    assert created.status_code == 201
    token, link_id = created.json()["token"], created.json()["link"]["id"]
    # 3. public page + downloads
    assert env.client.get(f"/share/{token}").status_code == 200
    assert env.client.get(f"/share/{token}/download?format=pdf").content.startswith(b"%PDF-")
    docx = env.client.get(f"/share/{token}/download?format=docx").content
    assert zipfile.ZipFile(io.BytesIO(docx)).testzip() is None
    stats = env.client.get("/api/meetings/m1/share").json()["links"][0]
    assert stats["accessCount"] == 3 and stats["lastAccessedAt"] == T0
    # 4. revoke -> 404
    assert env.client.delete(f"/api/meetings/m1/share/{link_id}").status_code == 200
    assert env.client.get(f"/share/{token}").status_code == 404
    # 5. new link, expire via clock
    token2 = _token_from(env.share(expiresInHours=1))
    assert env.client.get(f"/share/{token2}").status_code == 200
    env.clock.now = T0 + HOUR
    assert env.client.get(f"/share/{token2}").status_code == 404
    assert env.client.get("/api/meetings/m1/share").json()["links"][0]["status"] in {"expired", "revoked"}
    # 6. kill-switch: a fresh link stops resolving the moment the flag goes off
    token3 = _token_from(env.share(expiresInHours=None))
    assert env.client.get(f"/share/{token3}").status_code == 200
    env.set_config(shareLinksEnabled=False)
    assert env.client.get(f"/share/{token3}").status_code == 404
    # export is NOT gated by the flag
    assert env.client.get("/api/meetings/m1/export?format=txt").status_code == 200


# -- hardening: size cap, render failures, bad format ---------------------------------------------


def _big_transcript(n: int) -> dict:
    return {
        "segments": [
            {"startMs": i * 10, "endMs": i * 10 + 5, "speaker": f"S{i % 2}", "text": "word"} for i in range(n)
        ]
    }


def test_export_over_cap_is_413_for_heavy_formats(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    from olive.services import export

    env.seed(transcript=_big_transcript(40))
    monkeypatch.setattr(export, "MAX_EXPORT_SEGMENTS", 30)
    assert env.client.get("/api/meetings/m1/export?format=pdf").status_code == 413
    assert env.client.get("/api/meetings/m1/export?format=docx").status_code == 413
    monkeypatch.setattr(export, "MAX_EXPORT_SEGMENTS", 40)  # at the cap still works
    assert env.client.get("/api/meetings/m1/export?format=pdf").status_code == 200
    monkeypatch.setattr(export, "MAX_EXPORT_TEXT_BYTES", 100)
    assert env.client.get("/api/meetings/m1/export?format=docx").status_code == 413
    assert env.client.get("/api/meetings/m1/export?format=txt").status_code == 200  # cheap formats unaffected


def test_public_download_over_cap_is_413_with_headers(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    from olive.services import export

    env.seed(transcript=_big_transcript(40))
    env.enable()
    token = _token_from(env.share())
    monkeypatch.setattr(export, "MAX_EXPORT_SEGMENTS", 30)
    res = env.client.get(f"/share/{token}/download?format=pdf")
    assert res.status_code == 413
    _assert_share_headers(res)
    assert env.client.get(f"/share/{token}/download?format=txt").status_code == 200


def test_share_routes_survive_render_bugs_without_leaking_token(
    env: Env, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from olive.services import export
    from olive.services.sharing import ShareService

    env.seed()
    env.enable()
    token = _token_from(env.share())

    def boom(*_a, **_k):
        raise RuntimeError("renderer exploded")

    monkeypatch.setattr(export, "to_pdf", boom)
    with caplog.at_level("ERROR"):
        res = env.client.get(f"/share/{token}/download?format=pdf")
    assert res.status_code == 500
    _assert_share_headers(res)
    assert token not in res.text and token not in caplog.text

    monkeypatch.setattr(ShareService, "public_view", boom)
    with caplog.at_level("ERROR"):
        page = env.client.get(f"/share/{token}")
    assert page.status_code == 500
    _assert_share_headers(page)
    assert token not in page.text and token not in caplog.text


# -- /api/config reports the effective flag --------------------------------------------------------


@pytest.mark.parametrize(
    ("env_value", "stored", "expected"),
    [("1", False, True), (None, False, False), (None, True, True), ("0", True, False)],
)
def test_api_config_reports_effective_share_flag(
    env: Env, monkeypatch: pytest.MonkeyPatch, env_value: str | None, stored: bool, expected: bool
) -> None:
    env.set_config(shareLinksEnabled=stored)
    if env_value is not None:
        monkeypatch.setenv("OLIVE_SHARE_ENABLED", env_value)
    assert env.client.get("/api/config").json()["shareLinksEnabled"] is expected
    # the stored setting itself is untouched
    assert load_app_config(env.paths).shareLinksEnabled is stored
