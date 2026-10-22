"""Schema tests for meeting_share_links: table, index, constraints, delete_meeting cascade."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from olive.db import MeetingShareLinkModel, create_db, run_migrations
from olive.services.meetings import delete_meeting
from olive.settings import AppConfig, mask_app_config, resolve_paths, save_app_config

NOW = 1_700_000_000_000


def _meeting(conn, meeting_id: str = "m1") -> None:
    conn.execute(
        text(
            "INSERT INTO meetings (id, title, start_time, end_time, source, status, tags, created_at, updated_at) "
            "VALUES (:id, 't', :s, :e, 'upload', 'ready', '[]', :n, :n)"
        ),
        {"id": meeting_id, "s": NOW, "e": NOW + 1000, "n": NOW},
    )


def _link(conn, link_id: str, meeting_id: str = "m1", token_hash: str | None = None, **extra) -> None:
    cols = {
        "id": link_id,
        "meeting_id": meeting_id,
        "token_hash": token_hash or f"hash-{link_id}",
        "created_at": NOW,
        **extra,
    }
    conn.execute(
        text(
            f"INSERT INTO meeting_share_links ({', '.join(cols)}) VALUES ({', '.join(':' + c for c in cols)})"
        ),
        cols,
    )


def test_table_and_index_exist_and_migration_is_idempotent() -> None:
    handle = create_db(":memory:")
    run_migrations(handle.engine)
    with handle.engine.connect() as conn:
        tables = {r[0] for r in conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))}
        indexes = {r[0] for r in conn.execute(text("SELECT name FROM sqlite_master WHERE type='index'"))}
        cols = {r[1]: r for r in conn.execute(text("PRAGMA table_info(meeting_share_links)"))}
    assert "meeting_share_links" in tables
    assert "idx_share_links_meeting" in indexes
    assert set(cols) == {
        "id",
        "meeting_id",
        "token_hash",
        "label",
        "include_summary",
        "created_at",
        "expires_at",
        "revoked_at",
        "last_accessed_at",
        "access_count",
    }
    assert "token" not in cols  # hash-only storage: no raw token column


def test_defaults_and_round_trip() -> None:
    handle = create_db(":memory:")
    with handle.engine.begin() as conn:
        _meeting(conn)
        _link(conn, "l1")
    with handle.engine.connect() as conn:
        row = conn.execute(text("SELECT * FROM meeting_share_links")).mappings().one()
    assert row["include_summary"] == 0
    assert row["access_count"] == 0
    assert row["expires_at"] is None and row["revoked_at"] is None and row["label"] is None


def test_token_hash_is_unique() -> None:
    handle = create_db(":memory:")
    with handle.engine.begin() as conn:
        _meeting(conn)
        _link(conn, "l1", token_hash="same")
    with pytest.raises(IntegrityError):
        with handle.engine.begin() as conn:
            _link(conn, "l2", token_hash="same")


def test_meeting_foreign_key_enforced() -> None:
    handle = create_db(":memory:")
    with pytest.raises(IntegrityError):
        with handle.engine.begin() as conn:
            _link(conn, "l1", meeting_id="ghost")


def test_orm_model_matches_table() -> None:
    assert MeetingShareLinkModel.__tablename__ == "meeting_share_links"
    assert {c.name for c in MeetingShareLinkModel.__table__.columns} >= {"id", "token_hash", "expires_at"}


def test_delete_meeting_removes_its_links_only(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    with handle.engine.begin() as conn:
        _meeting(conn, "m1")
        _meeting(conn, "m2")
        _link(conn, "l1", "m1")
        _link(conn, "l2", "m1")
        _link(conn, "l3", "m2")
    assert delete_meeting(handle.engine, "m1", str(tmp_path)) is True  # no IntegrityError
    with handle.engine.connect() as conn:
        assert conn.execute(text("SELECT id FROM meeting_share_links")).scalars().all() == ["l3"]


# -- settings ---------------------------------------------------------------------


def test_settings_defaults_and_validation() -> None:
    cfg = AppConfig()
    assert cfg.shareLinksEnabled is False
    assert cfg.shareDefaultTtlHours == 168
    assert AppConfig(shareDefaultTtlHours=0).shareDefaultTtlHours == 0
    with pytest.raises(ValidationError):
        AppConfig(shareDefaultTtlHours=-1)


def test_settings_reach_frontend_and_persist(tmp_path: Path) -> None:
    paths = resolve_paths({"OLIVE_CONFIG_DIR": str(tmp_path)})
    saved = save_app_config(AppConfig(shareLinksEnabled=True, shareDefaultTtlHours=24), paths)
    masked = mask_app_config(saved)
    assert masked["shareLinksEnabled"] is True
    assert masked["shareDefaultTtlHours"] == 24
    # Old settings.json files without the new keys still load with defaults.
    assert AppConfig.model_validate({"pollIntervalMinutes": 5}).shareLinksEnabled is False
