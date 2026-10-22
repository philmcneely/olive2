"""Tests for the Insight Groups schema: tables, indexes, constraints, and cascades."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from olive.db import create_db, run_migrations
from olive.services.insight_groups import delete_group
from olive.services.meetings import delete_meeting

NOW = 1_700_000_000_000
GROUP_TABLES = ("insight_groups", "insight_group_members", "insight_group_snapshots")


def _insert_meeting(conn, meeting_id: str) -> None:
    conn.execute(
        text(
            """
            INSERT INTO meetings (
              id, title, start_time, end_time, source, status, tags, created_at, updated_at
            ) VALUES (:id, 'Group fixture', :start, :end, 'upload', 'ready', '[]', :now, :now)
            """
        ),
        {"id": meeting_id, "start": NOW, "end": NOW + 600_000, "now": NOW},
    )


def _insert_group(conn, group_id: str, name: str = "Weekly 1:1") -> None:
    conn.execute(
        text(
            "INSERT INTO insight_groups (id, name, description, axes, created_at, updated_at) "
            "VALUES (:id, :name, NULL, NULL, :now, :now)"
        ),
        {"id": group_id, "name": name, "now": NOW},
    )


def _insert_member(conn, member_id: str, group_id: str, meeting_id: str) -> None:
    conn.execute(
        text(
            "INSERT INTO insight_group_members (id, group_id, meeting_id, added_at) "
            "VALUES (:id, :gid, :mid, :now)"
        ),
        {"id": member_id, "gid": group_id, "mid": meeting_id, "now": NOW},
    )


def _insert_snapshot(conn, snap_id: str, group_id: str, *, created_at: int = NOW) -> None:
    conn.execute(
        text(
            "INSERT INTO insight_group_snapshots "
            "(id, group_id, created_at, member_count, axes, meeting_ids, reason) "
            "VALUES (:id, :gid, :ts, 1, '{\"warmth\": 0.4}', '[\"m1\"]', 'append')"
        ),
        {"id": snap_id, "gid": group_id, "ts": created_at},
    )


def _ids(conn, table: str, column: str = "id") -> list[str]:
    return sorted(conn.execute(text(f"SELECT {column} FROM {table}")).scalars().all())


def test_group_tables_and_indexes_exist_and_are_idempotent() -> None:
    handle = create_db(":memory:")
    run_migrations(handle.engine)  # second application must be a no-op

    with handle.engine.connect() as conn:
        tables = {r[0] for r in conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))}
        indexes = {r[0] for r in conn.execute(text("SELECT name FROM sqlite_master WHERE type='index'"))}
        unique = {
            r[1]
            for r in conn.execute(text("PRAGMA index_list(insight_group_members)"))
            if r[2] == 1
        }

    assert set(GROUP_TABLES) <= tables
    assert {
        "idx_group_members_group",
        "idx_group_members_unique",
        "idx_group_members_meeting",
        "idx_group_snapshots_group",
    } <= indexes
    assert "idx_group_members_unique" in unique


def test_group_tables_round_trip() -> None:
    handle = create_db(":memory:")
    with handle.engine.begin() as conn:
        _insert_meeting(conn, "m1")
        _insert_group(conn, "g1")
        _insert_member(conn, "gm1", "g1", "m1")
        _insert_snapshot(conn, "s1", "g1")
        conn.execute(
            text("UPDATE insight_groups SET axes = '[\"warmth\"]', description = 'd' WHERE id = 'g1'")
        )

    with handle.engine.connect() as conn:
        group = conn.execute(text("SELECT * FROM insight_groups")).mappings().one()
        member = conn.execute(text("SELECT * FROM insight_group_members")).mappings().one()
        snap = conn.execute(text("SELECT * FROM insight_group_snapshots")).mappings().one()
    assert (group["name"], group["axes"], group["description"]) == ("Weekly 1:1", '["warmth"]', "d")
    assert (member["group_id"], member["meeting_id"], member["added_at"]) == ("g1", "m1", NOW)
    assert (snap["member_count"], snap["axes"], snap["meeting_ids"], snap["reason"]) == (
        1,
        '{"warmth": 0.4}',
        '["m1"]',
        "append",
    )


def test_meeting_can_belong_to_many_groups_but_only_once_per_group() -> None:
    handle = create_db(":memory:")
    with handle.engine.begin() as conn:
        _insert_meeting(conn, "m1")
        _insert_group(conn, "g1")
        _insert_group(conn, "g2")
        _insert_member(conn, "gm1", "g1", "m1")
        _insert_member(conn, "gm2", "g2", "m1")

    with pytest.raises(IntegrityError):
        with handle.engine.begin() as conn:
            _insert_member(conn, "gm3", "g1", "m1")

    with handle.engine.connect() as conn:
        assert _ids(conn, "insight_group_members", "group_id") == ["g1", "g2"]


def test_member_foreign_keys_are_enforced() -> None:
    handle = create_db(":memory:")
    with handle.engine.begin() as conn:
        _insert_meeting(conn, "m1")
        _insert_group(conn, "g1")
    with pytest.raises(IntegrityError):
        with handle.engine.begin() as conn:
            _insert_member(conn, "gm1", "nope", "m1")
    with pytest.raises(IntegrityError):
        with handle.engine.begin() as conn:
            _insert_member(conn, "gm2", "g1", "nope")


def test_delete_meeting_removes_memberships_but_keeps_snapshots(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    with handle.engine.begin() as conn:
        _insert_meeting(conn, "m1")
        _insert_meeting(conn, "m2")
        _insert_group(conn, "g1")
        _insert_group(conn, "g2")
        _insert_member(conn, "gm1", "g1", "m1")
        _insert_member(conn, "gm2", "g2", "m1")
        _insert_member(conn, "gm3", "g1", "m2")
        _insert_snapshot(conn, "s1", "g1")
        _insert_snapshot(conn, "s2", "g2")

    assert delete_meeting(handle.engine, "m1", str(tmp_path)) is True

    with handle.engine.connect() as conn:
        assert _ids(conn, "insight_group_members") == ["gm3"]
        assert _ids(conn, "insight_group_snapshots") == ["s1", "s2"]
        assert _ids(conn, "insight_groups") == ["g1", "g2"]


def test_delete_group_removes_members_and_snapshots_only_for_that_group() -> None:
    handle = create_db(":memory:")
    with handle.engine.begin() as conn:
        _insert_meeting(conn, "m1")
        _insert_group(conn, "g1")
        _insert_group(conn, "g2")
        _insert_member(conn, "gm1", "g1", "m1")
        _insert_member(conn, "gm2", "g2", "m1")
        _insert_snapshot(conn, "s1", "g1")
        _insert_snapshot(conn, "s2", "g2")

    assert delete_group(handle.engine, "g1") is True
    assert delete_group(handle.engine, "g1") is False

    with handle.engine.connect() as conn:
        assert _ids(conn, "insight_groups") == ["g2"]
        assert _ids(conn, "insight_group_members") == ["gm2"]
        assert _ids(conn, "insight_group_snapshots") == ["s2"]
        assert _ids(conn, "meetings") == ["m1"]
