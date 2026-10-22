"""App-wide CSRF Origin guard + opt-in Host check on mutating methods; env-controlled share flag."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from olive.app import create_app
from olive.db import create_db
from olive.services.ingest import IngestService
from olive.settings import AppConfig, clear_app_config_cache, load_app_config, resolve_paths, save_app_config

GUARD_MSG = "Cross-origin request blocked"
HOST_MSG = "Host not allowed"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    for name in ("OLIVE_ALLOWED_ORIGINS", "OLIVE_ALLOWED_HOSTS", "OLIVE_SHARE_ENABLED", "OLIVE_BIND_HOST"):
        monkeypatch.delenv(name, raising=False)
    clear_app_config_cache()
    yield
    clear_app_config_cache()


def make_client(tmp_path: Path, **opts) -> TestClient:
    handle = create_db(":memory:")
    app = create_app(
        {
            "db": handle.engine,
            "meetingsDir": str(tmp_path / "meetings"),
            "configDir": str(tmp_path / "cfg"),
            **opts,
        }
    )
    return TestClient(app)


def blocked(res) -> bool:
    return res.status_code == 403 and GUARD_MSG in res.text


# -- Origin guard ----------------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_foreign_origin_blocked_on_mutating_methods(tmp_path: Path, method: str) -> None:
    client = make_client(tmp_path)
    res = client.request(method, "/api/logs", headers={"Origin": "https://evil.example"})
    assert blocked(res)


def test_null_origin_blocked(tmp_path: Path) -> None:
    assert blocked(make_client(tmp_path).delete("/api/logs", headers={"Origin": "null"}))


@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost:5173",
        "http://127.0.0.1:4471",
        "https://localhost",
        "http://[::1]:4471",
        "http://testserver",  # same-origin: matches the request Host header
    ],
)
def test_local_and_same_origin_allowed(tmp_path: Path, origin: str) -> None:
    res = make_client(tmp_path).delete("/api/logs", headers={"Origin": origin})
    assert res.status_code == 200


def test_lan_same_origin_allowed_via_host_header(tmp_path: Path) -> None:
    res = make_client(tmp_path).delete(
        "/api/logs", headers={"Origin": "http://192.168.0.50:4471", "Host": "192.168.0.50:4471"}
    )
    assert res.status_code == 200
    # but a different origin than the Host is a cross-site request
    assert blocked(
        make_client(tmp_path).delete(
            "/api/logs", headers={"Origin": "http://evil.example", "Host": "192.168.0.50:4471"}
        )
    )


def test_localhost_lookalikes_blocked(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    for origin in ("http://localhost.evil.example", "http://127.0.0.1.evil.example", "ftp://localhost"):
        assert blocked(client.delete("/api/logs", headers={"Origin": origin})), origin


def test_bind_host_and_env_allowlist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = make_client(tmp_path, bindHost="192.168.0.50", mcpToken="t")
    assert client.delete("/api/logs", headers={"Origin": "http://192.168.0.50:9999"}).status_code == 200
    assert blocked(client.delete("/api/logs", headers={"Origin": "https://olive.example.com"}))
    monkeypatch.setenv("OLIVE_ALLOWED_ORIGINS", " https://olive.example.com , http://nas.local:8080 ")
    assert client.delete("/api/logs", headers={"Origin": "https://olive.example.com"}).status_code == 200
    assert client.delete("/api/logs", headers={"Origin": "http://nas.local:8080"}).status_code == 200
    assert blocked(client.delete("/api/logs", headers={"Origin": "http://nas.local:9"}))  # port matters


def test_wildcard_bind_host_is_not_an_allowed_origin(tmp_path: Path) -> None:
    client = make_client(tmp_path, bindHost="0.0.0.0", mcpToken="t")
    assert blocked(client.delete("/api/logs", headers={"Origin": "http://0.0.0.0:4471"}))


def test_no_origin_header_always_passes(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    for method in ("POST", "PUT", "PATCH", "DELETE"):
        assert not blocked(client.request(method, "/api/logs"))
    assert client.delete("/api/logs").status_code == 200


@pytest.mark.parametrize("method", ["GET", "HEAD", "OPTIONS"])
def test_safe_methods_never_blocked(tmp_path: Path, method: str) -> None:
    res = make_client(tmp_path).request(method, "/api/config", headers={"Origin": "https://evil.example"})
    assert not blocked(res)
    assert (
        make_client(tmp_path).get("/api/config", headers={"Origin": "https://evil.example"}).status_code
        == 200
    )


def test_foreign_origin_cannot_reach_share_creation(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    assert blocked(client.post("/api/meetings/m1/share", json={}, headers={"Origin": "https://evil.example"}))


# -- external contracts stay open to non-browser clients ------------------------------------------


def test_ingest_without_origin_still_works_and_foreign_origin_is_blocked(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    svc = IngestService(db=handle.engine, meetings_dir=meetings_dir, now=lambda: 1_700_000_000_000)
    client = TestClient(
        create_app(
            {
                "db": handle.engine,
                "meetingsDir": meetings_dir,
                "configDir": str(tmp_path / "cfg"),
                "ingestService": svc,
                "ingestToken": "secret-token-xyz",
            }
        )
    )
    files = {"file": ("a.m4a", b"unparseable audio bytes", "audio/mp4")}
    ok = client.post(
        "/api/ingest", headers={"Authorization": "Bearer secret-token-xyz"}, files=files, data={"title": "T"}
    )
    assert ok.status_code == 201
    unauth = client.post("/api/ingest", files=files)
    assert unauth.status_code == 401  # reaches the route's own auth, not the guard
    evil = client.post(
        "/api/ingest",
        headers={"Authorization": "Bearer secret-token-xyz", "Origin": "https://evil.example"},
        files=files,
    )
    assert blocked(evil)


def test_mcp_and_speechmatics_webhook_without_origin_reach_their_routes(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    mcp = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        headers={"Accept": "application/json, text/event-stream"},
    )
    assert not blocked(mcp)
    hook = client.post("/api/webhooks/speechmatics", json={})
    assert not blocked(hook)
    assert blocked(client.post("/mcp", json={}, headers={"Origin": "https://evil.example"}))


# -- Host check (opt-in) -----------------------------------------------------------------------------


def test_host_check_off_by_default(tmp_path: Path) -> None:
    res = make_client(tmp_path).delete("/api/logs", headers={"Host": "rebind.evil.example"})
    assert res.status_code == 200


def test_host_check_on_rejects_unknown_host_and_allows_listed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OLIVE_ALLOWED_HOSTS", "olive.lan, 192.168.0.50")
    client = make_client(tmp_path, bindHost="10.0.0.9", mcpToken="t")
    bad = client.delete("/api/logs", headers={"Host": "rebind.evil.example"})
    assert bad.status_code == 403 and HOST_MSG in bad.text
    for host in (
        "olive.lan",
        "olive.lan:4471",
        "192.168.0.50:4471",
        "localhost:4471",
        "127.0.0.1",
        "10.0.0.9:80",
    ):
        assert client.delete("/api/logs", headers={"Host": host}).status_code == 200, host
    # reads and no-Origin semantics are unchanged: only mutating methods are host-checked
    assert client.get("/api/config", headers={"Host": "rebind.evil.example"}).status_code == 200


# -- FIX A: env-controlled share flag is never persisted by a settings save ------------------------


def _stored_flag(tmp_path: Path) -> bool:
    raw = json.loads((tmp_path / "cfg" / "settings.json").read_text())
    return raw["shareLinksEnabled"]


@pytest.mark.parametrize("posted", [True, False])
def test_env_set_ignores_posted_share_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, posted: bool
) -> None:
    client = make_client(tmp_path)
    save_app_config(
        AppConfig(shareLinksEnabled=False), resolve_paths({"OLIVE_CONFIG_DIR": str(tmp_path / "cfg")})
    )
    monkeypatch.setenv("OLIVE_SHARE_ENABLED", "1")
    full = client.get("/api/config").json()
    assert full["shareLinksEnabled"] is True  # effective
    full["shareLinksEnabled"] = posted
    res = client.post("/api/config", json=full)
    assert res.status_code == 200
    assert _stored_flag(tmp_path) is False  # unchanged
    assert res.json()["config"]["shareLinksEnabled"] is True  # still effective-from-env
    clear_app_config_cache()
    monkeypatch.delenv("OLIVE_SHARE_ENABLED")
    assert client.get("/api/config").json()["shareLinksEnabled"] is False  # does not stick


def test_env_set_other_settings_still_saved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = make_client(tmp_path)
    monkeypatch.setenv("OLIVE_SHARE_ENABLED", "1")
    res = client.post("/api/config", json={"shareLinksEnabled": True, "pollIntervalMinutes": 7})
    assert res.status_code == 200
    assert (
        load_app_config(resolve_paths({"OLIVE_CONFIG_DIR": str(tmp_path / "cfg")})).pollIntervalMinutes == 7
    )


def test_env_unset_honors_posted_share_flag(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    assert client.post("/api/config", json={"shareLinksEnabled": True}).status_code == 200
    assert _stored_flag(tmp_path) is True
    assert client.post("/api/config", json={"shareLinksEnabled": False}).status_code == 200
    assert _stored_flag(tmp_path) is False
