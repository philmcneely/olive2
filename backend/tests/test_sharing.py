"""ShareService: token handling, expiry/revocation, projection whitelist, HTML escaping."""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import pytest
from sqlalchemy import text

from export_support import T0, seed_meeting
from olive.db import create_db
from olive.services.export import ExportError
from olive.services.sharing import (
    SHARE_RESPONSE_HEADERS,
    TOKEN_RE,
    ShareDisabledError,
    ShareService,
    render_share_html,
    share_links_enabled,
)
from olive.settings import AppConfig

HOUR = 3_600_000


class Clock:
    def __init__(self, now: int = T0) -> None:
        self.now = now

    def __call__(self) -> int:
        return self.now


class Env:
    def __init__(self, tmp_path: Path, *, enabled: bool = True) -> None:
        self.handle = create_db(":memory:")
        self.db = self.handle.engine
        self.meetings = str(tmp_path / "meetings")
        self.clock = Clock()
        self.enabled = enabled
        self.svc = ShareService(self.db, self.meetings, now_ms=self.clock, is_enabled=lambda: self.enabled)

    def seed(self, meeting_id: str = "m1", **kw) -> str:
        return seed_meeting(self.db, self.meetings, meeting_id, **kw)


@pytest.fixture
def env(tmp_path: Path) -> Env:
    e = Env(tmp_path)
    e.seed(summary_md="## Decisions\n- ship it")
    return e


# -- token shape & storage ---------------------------------------------------------------


def test_token_shape_and_hash_only_storage(env: Env) -> None:
    link, token = env.svc.create("m1", expires_in_hours=24)
    assert TOKEN_RE.match(token) and len(token) == 43
    with env.db.connect() as conn:
        row = conn.execute(text("SELECT * FROM meeting_share_links")).mappings().one()
    assert row["token_hash"] == hashlib.sha256(token.encode()).hexdigest()
    assert token not in "|".join(str(v) for v in row.values())
    assert row["id"] == link.id and link.id not in token
    assert not hasattr(link, "token") and not hasattr(link, "token_hash")


def test_tokens_are_unique(env: Env) -> None:
    tokens = {env.svc.create("m1", expires_in_hours=1)[1] for _ in range(25)}
    assert len(tokens) == 25


def test_token_regex_rejects_near_misses() -> None:
    good = "a" * 43
    assert TOKEN_RE.match(good)
    for bad in [
        "",
        "a" * 42,
        "a" * 44,
        good + "\n",
        "a" * 42 + "!",
        "a" * 42 + " ",
        "../" + "a" * 40,
        "é" * 43,
    ]:
        assert not TOKEN_RE.match(bad), bad


def test_list_never_returns_token_and_shows_status(env: Env) -> None:
    env.svc.create("m1", expires_in_hours=1, label="  for Bob  ")
    env.svc.create("m1", expires_in_hours=None, include_summary=True)
    links = env.svc.list_for_meeting("m1")
    assert len(links) == 2
    assert {lk.label for lk in links} == {"for Bob", None}
    assert all(lk.status == "active" for lk in links)
    assert env.svc.list_for_meeting("other") == []
    for lk in links:
        assert "token" not in {f.name for f in dataclasses.fields(lk)}


# -- create validation ----------------------------------------------------------------------


def test_create_validation(env: Env) -> None:
    env.seed("nt", transcript=None)
    with pytest.raises(ExportError) as e1:
        env.svc.create("ghost", expires_in_hours=1)
    assert e1.value.code == "not_found"
    with pytest.raises(ExportError) as e2:
        env.svc.create("nt", expires_in_hours=1)
    assert e2.value.code == "no_transcript"
    for bad in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            env.svc.create("m1", expires_in_hours=bad)
    link, _ = env.svc.create("m1", expires_in_hours=1, label="x" * 500)
    assert len(link.label or "") <= 120


def test_create_blocked_when_flag_off(env: Env) -> None:
    env.enabled = False
    with pytest.raises(ShareDisabledError):
        env.svc.create("m1", expires_in_hours=1)


# -- resolve: expiry, revoke, flag, malformed ------------------------------------------------


def test_resolve_active_touches_access_stats(env: Env) -> None:
    link, token = env.svc.create("m1", expires_in_hours=2)
    env.clock.now = T0 + 5000
    got = env.svc.resolve(token)
    assert got is not None and got.id == link.id
    assert got.access_count == 1 and got.last_accessed_at == T0 + 5000
    assert env.svc.resolve(token).access_count == 2  # type: ignore[union-attr]


def test_expiry_via_injected_clock_boundary(env: Env) -> None:
    _, token = env.svc.create("m1", expires_in_hours=1)
    env.clock.now = T0 + HOUR - 1
    assert env.svc.resolve(token) is not None
    env.clock.now = T0 + HOUR  # exactly expires_at -> expired
    assert env.svc.resolve(token) is None
    assert env.svc.list_for_meeting("m1")[0].status == "expired"


def test_never_expiring_link(env: Env) -> None:
    _, token = env.svc.create("m1", expires_in_hours=None)
    env.clock.now = T0 + 10_000 * HOUR
    assert env.svc.resolve(token) is not None


def test_revoke_is_idempotent_and_final(env: Env) -> None:
    link, token = env.svc.create("m1", expires_in_hours=None)
    env.clock.now = T0 + 1000
    assert env.svc.revoke("m1", link.id) is True
    first = env.svc.list_for_meeting("m1")[0]
    assert first.status == "revoked" and first.revoked_at == T0 + 1000
    env.clock.now = T0 + 9999
    assert env.svc.revoke("m1", link.id) is True  # idempotent
    assert env.svc.list_for_meeting("m1")[0].revoked_at == T0 + 1000  # not bumped
    assert env.svc.resolve(token) is None


def test_revoke_scoped_to_meeting(env: Env) -> None:
    env.seed("m2")
    link, token = env.svc.create("m1", expires_in_hours=None)
    assert env.svc.revoke("m2", link.id) is False
    assert env.svc.revoke("m1", "nope") is False
    assert env.svc.resolve(token) is not None


def test_flag_off_kills_resolve_but_not_revoke(env: Env) -> None:
    link, token = env.svc.create("m1", expires_in_hours=None)
    env.enabled = False
    assert env.svc.resolve(token) is None
    assert env.svc.revoke("m1", link.id) is True
    env.enabled = True
    assert env.svc.resolve(token) is None  # revoke stuck


def test_resolve_rejects_malformed_and_unknown(env: Env) -> None:
    env.svc.create("m1", expires_in_hours=None)
    for bad in ["", "x", "a" * 43, "' OR 1=1 --" + "a" * 32, "a" * 43 + "\n", "%00" * 20, None]:
        assert env.svc.resolve(bad) is None  # type: ignore[arg-type]
    with env.db.connect() as conn:
        assert conn.execute(text("SELECT access_count FROM meeting_share_links")).scalar_one() == 0


def test_deleted_meeting_link_does_not_resolve(env: Env, tmp_path: Path) -> None:
    from olive.services.meetings import delete_meeting

    _, token = env.svc.create("m1", expires_in_hours=None)
    delete_meeting(env.db, "m1", env.meetings)
    assert env.svc.resolve(token) is None


def test_meeting_missing_transcript_later_fails_closed(env: Env) -> None:
    link, token = env.svc.create("m1", expires_in_hours=None)
    with env.db.begin() as conn:
        conn.execute(text("UPDATE meetings SET primary_transcript_artifact_id = NULL WHERE id='m1'"))
    got = env.svc.resolve(token)
    assert got is not None
    with pytest.raises(ExportError):
        env.svc.public_view(got)


# -- projection whitelist -------------------------------------------------------------------


def test_public_view_whitelist_has_no_internal_fields(tmp_path: Path) -> None:
    e = Env(tmp_path)
    e.seed(tags=["secret-tag"], summary_md="## S\nsummary body")
    with e.db.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO recordings (id, meeting_id, provider, path, mime, duration_ms, size_bytes, sha256, "
                "created_at) VALUES ('r1','m1','upload','audio/SECRETPATH.opus','audio/ogg',1,1,'deadbeefsha',0)"
            )
        )
        conn.execute(text("UPDATE meetings SET last_error = 'LEAKY_ERROR' WHERE id='m1'"))
    link, token = e.svc.create("m1", expires_in_hours=None)
    view = e.svc.public_view(link)
    assert {f.name for f in dataclasses.fields(view)} == {
        "title",
        "date",
        "duration_ms",
        "speakers",
        "segments",
        "summary_md",
    }
    assert view.summary_md is None  # include_summary defaults to False
    blob = repr(view) + render_share_html(view, token)
    for leaked in ("secret-tag", "SECRETPATH", "deadbeefsha", "LEAKY_ERROR"):
        assert leaked not in blob
    assert view.title == "Weekly Sync" and view.date == "2023-11-14"
    assert view.speakers == ["Alice", "Bob"]
    assert [(s.start_ms, s.speaker) for s in view.segments] == [(0, "Alice"), (7000, "Bob")]  # coalesced


def test_public_view_summary_only_when_link_opted_in(env: Env) -> None:
    with_s, _ = env.svc.create("m1", expires_in_hours=None, include_summary=True)
    without, _ = env.svc.create("m1", expires_in_hours=None)
    assert "ship it" in (env.svc.public_view(with_s).summary_md or "")
    assert env.svc.public_view(without).summary_md is None
    assert env.svc.export_doc(without).summary_md is None
    assert env.svc.export_doc(with_s).summary_md is not None


# -- HTML escaping ---------------------------------------------------------------------------


def test_html_escapes_every_interpolated_value(tmp_path: Path) -> None:
    e = Env(tmp_path)
    xss = "<script>alert(1)</script>"
    e.seed(
        title=f"T {xss} \"q\" 'a'",
        transcript={
            "segments": [
                {"startMs": 0, "endMs": 1000, "speaker": f"S{xss}", "text": f"hi {xss} <img src=x onerror=1>"}
            ]
        },
        summary_md=f"## Sum {xss}\n[x](javascript:alert(1))",
    )
    link, token = e.svc.create("m1", expires_in_hours=None, include_summary=True)
    html_doc = render_share_html(e.svc.public_view(link), token)
    low = html_doc.lower()
    assert "<script" not in low and "<img" not in low and "onerror=1>" not in low.replace("&lt;img", "")
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html_doc
    assert html_doc.startswith("<!doctype html>")
    assert 'name="viewport"' in html_doc and "<style>" in html_doc
    assert "http://" not in html_doc and "https://" not in html_doc  # no external resources
    assert f"/share/{token}/download?format=pdf" in html_doc


def test_html_empty_transcript_view_renders(env: Env) -> None:
    link, token = env.svc.create("m1", expires_in_hours=None)
    assert "Weekly Sync" in render_share_html(env.svc.public_view(link), token)


def test_response_headers_policy() -> None:
    h = {k.lower(): v for k, v in SHARE_RESPONSE_HEADERS.items()}
    csp = h["content-security-policy"]
    for part in (
        "default-src 'none'",
        "style-src 'unsafe-inline'",
        "base-uri 'none'",
        "frame-ancestors 'none'",
    ):
        assert part in csp
    assert "script-src" not in csp or "script-src 'none'" in csp
    assert h["cache-control"] == "no-store"
    assert h["referrer-policy"] == "no-referrer"
    assert h["x-robots-tag"] == "noindex"
    assert h["x-content-type-options"] == "nosniff"
    assert "form-action 'none'" in csp
    assert h["cross-origin-resource-policy"] == "same-origin"


# -- feature flag ----------------------------------------------------------------------------


def test_share_links_enabled_env_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLIVE_SHARE_ENABLED", raising=False)
    assert share_links_enabled(AppConfig()) is False
    assert share_links_enabled(AppConfig(shareLinksEnabled=True)) is True
    assert share_links_enabled(None) is False
    monkeypatch.setenv("OLIVE_SHARE_ENABLED", "1")
    assert share_links_enabled(AppConfig()) is True
    assert share_links_enabled(None) is True
    monkeypatch.setenv("OLIVE_SHARE_ENABLED", "0")
    assert share_links_enabled(AppConfig(shareLinksEnabled=True)) is False
    monkeypatch.setenv("OLIVE_SHARE_ENABLED", "  ")
    assert share_links_enabled(AppConfig(shareLinksEnabled=True)) is True  # blank = unset
    monkeypatch.setenv("OLIVE_SHARE_ENABLED", "maybe")
    assert share_links_enabled(AppConfig(shareLinksEnabled=True)) is False  # unrecognized fails closed


def test_default_enabled_check_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLIVE_SHARE_ENABLED", raising=False)
    handle = create_db(":memory:")
    svc = ShareService(handle.engine, str(tmp_path))
    with pytest.raises(ShareDisabledError):
        svc.create("m1", expires_in_hours=1)
