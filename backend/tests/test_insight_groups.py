"""Tests for InsightGroupService: CRUD, membership, aggregation, snapshots, trends."""

from __future__ import annotations

import json
from typing import Any

import pytest
from sqlalchemy import text

from olive.db import DbHandle, create_db
from olive.services.insight_groups import InsightGroupService

DAY = 86_400_000
T0 = 1_700_000_000_000


@pytest.fixture
def handle(tmp_path) -> DbHandle:
    return create_db(":memory:")


@pytest.fixture
def svc(handle: DbHandle, tmp_path) -> InsightGroupService:
    return InsightGroupService(db=handle, config_dir=str(tmp_path / "cfg"))


def _run(handle: DbHandle, sql: str, params: dict[str, Any] | None = None) -> None:
    with handle.engine.begin() as conn:
        conn.execute(text(sql), params or {})


def meeting(handle: DbHandle, mid: str, start: int) -> None:
    _run(
        handle,
        "INSERT INTO meetings (id, title, start_time, end_time, source, status, tags, created_at, updated_at) "
        "VALUES (:id, :t, :s, :e, 'upload', 'ready', '[]', 0, 0)",
        {"id": mid, "t": f"Title {mid}", "s": start, "e": start + 1000},
    )


_n = 0


def signal(handle: DbHandle, mid: str, label: str, dims: dict[str, float]) -> None:
    global _n
    _n += 1
    _run(
        handle,
        "INSERT INTO meeting_speaker_signals (id, meeting_id, speaker_label, dimensions, created_at) "
        "VALUES (:id, :m, :l, :d, 0)",
        {"id": f"sig{_n}", "m": mid, "l": label, "d": json.dumps(dims)},
    )


def moment(handle: DbHandle, mid: str, valence: str) -> None:
    global _n
    _n += 1
    _run(
        handle,
        "INSERT INTO meeting_moments (id, meeting_id, start_ms, event_type, valence, quote, created_at) "
        "VALUES (:id, :m, 5, 'evt', :v, 'q', 0)",
        {"id": f"mom{_n}", "m": mid, "v": valence},
    )


@pytest.fixture
def seeded(handle: DbHandle) -> DbHandle:
    # m_late is inserted first to prove ordering is by start time, not insertion.
    meeting(handle, "m_late", T0 + 14 * DAY)
    meeting(handle, "m1", T0)
    meeting(handle, "m2", T0 + 7 * DAY)
    meeting(handle, "m_bare", T0 + 21 * DAY)  # no signals
    signal(handle, "m1", "A", {"warmth": 0.2, "hostility": 0.4})
    signal(handle, "m1", "B", {"warmth": 0.4, "hostility": 0.0})  # m1 mean warmth 0.3, host 0.2
    signal(handle, "m2", "A", {"warmth": 0.6})  # m2 warmth 0.6
    signal(handle, "m_late", "A", {"warmth": 0.9, "hostility": 0.1})
    moment(handle, "m1", "good")
    moment(handle, "m1", "bad")
    moment(handle, "m2", "good")
    moment(handle, "m_late", "bad")
    return handle


def test_create_list_get_update_delete(svc: InsightGroupService) -> None:
    g = svc.create_group("Weekly", "desc", ["warmth"])
    assert g["name"] == "Weekly" and g["description"] == "desc" and g["axes"] == ["warmth"]
    assert svc.list_groups()[0]["memberCount"] == 0
    got = svc.get_group(g["id"])
    assert got is not None and got["members"] == [] and "aggregate" in got
    up = svc.update_group(g["id"], name="Renamed", axes=None, description="d2")
    assert up is not None and up["name"] == "Renamed" and up["description"] == "d2"
    assert svc.delete_group(g["id"]) is True
    assert svc.get_group(g["id"]) is None
    assert svc.delete_group(g["id"]) is False


def test_missing_group_returns_none(svc: InsightGroupService) -> None:
    assert svc.get_group("nope") is None
    assert svc.update_group("nope", name="x") is None
    assert svc.add_members("nope", ["m1"]) is None
    assert svc.remove_member("nope", "m1") is None
    assert svc.get_trends("nope") is None
    assert svc.snapshot_group("nope", "manual") is None


def test_create_requires_name(svc: InsightGroupService) -> None:
    with pytest.raises(ValueError):
        svc.create_group("   ")


def test_add_members_ignores_duplicates_and_unknown(seeded: DbHandle, svc: InsightGroupService) -> None:
    g = svc.create_group("G")
    svc.add_members(g["id"], ["m1", "m1"])
    svc.add_members(g["id"], ["m1", "m2", "ghost"])
    got = svc.get_group(g["id"])
    assert got is not None
    assert [m["meetingId"] for m in got["members"]] == ["m1", "m2"]
    assert svc.list_groups()[0]["memberCount"] == 2


def test_aggregate_all_axes(seeded: DbHandle, svc: InsightGroupService) -> None:
    g = svc.create_group("G")
    svc.add_members(g["id"], ["m_late", "m1", "m2", "m_bare"])
    agg = svc.aggregate_group(g["id"])
    assert agg is not None
    assert agg["axes"]["warmth"] == pytest.approx((0.3 + 0.6 + 0.9) / 3)
    assert agg["axes"]["hostility"] == pytest.approx((0.2 + 0.1) / 2)
    assert agg["moments"] == {"good": 2, "bad": 2}
    assert agg["coverage"] == {"analyzedMembers": 3, "totalMembers": 4}
    series = agg["series"]
    assert [p["meetingId"] for p in series] == ["m1", "m2", "m_late", "m_bare"]
    assert series[0]["startMs"] == T0 and series[0]["title"] == "Title m1"
    assert series[0]["axisScores"] == {"warmth": pytest.approx(0.3), "hostility": pytest.approx(0.2)}
    assert series[0]["moments"] == {"good": 1, "bad": 1}
    assert series[3]["axisScores"] == {}


def test_aggregate_respects_selected_axes(seeded: DbHandle, svc: InsightGroupService) -> None:
    g = svc.create_group("G", axes=["warmth"])
    svc.add_members(g["id"], ["m1", "m2"])
    agg = svc.aggregate_group(g["id"])
    assert agg is not None
    assert set(agg["axes"]) == {"warmth"}
    assert all(set(p["axisScores"]) <= {"warmth"} for p in agg["series"])


def test_meeting_in_two_groups_contributes_to_both(seeded: DbHandle, svc: InsightGroupService) -> None:
    a = svc.create_group("A")
    b = svc.create_group("B")
    svc.add_members(a["id"], ["m1"])
    svc.add_members(b["id"], ["m1", "m2"])
    assert svc.aggregate_group(a["id"])["axes"]["warmth"] == pytest.approx(0.3)  # type: ignore[index]
    assert svc.aggregate_group(b["id"])["axes"]["warmth"] == pytest.approx(0.45)  # type: ignore[index]


def test_add_creates_snapshot_and_history_grows(seeded: DbHandle, svc: InsightGroupService) -> None:
    g = svc.create_group("G")
    res = svc.add_members(g["id"], ["m1"])
    assert res is not None
    assert res["snapshot"]["reason"] == "append" and res["snapshot"]["memberCount"] == 1
    assert res["aggregate"]["axes"]["warmth"] == pytest.approx(0.3)
    svc.add_members(g["id"], ["m2"])
    trends = svc.get_trends(g["id"])
    assert trends is not None
    snaps = trends["snapshots"]
    assert [s["memberCount"] for s in snaps] == [1, 2]
    assert snaps[0]["axes"]["warmth"] == pytest.approx(0.3)
    assert snaps[1]["axes"]["warmth"] == pytest.approx(0.45)
    assert [p["meetingId"] for p in trends["memberSeries"]] == ["m1", "m2"]


def test_noop_add_still_snapshots_only_when_changed(seeded: DbHandle, svc: InsightGroupService) -> None:
    g = svc.create_group("G")
    svc.add_members(g["id"], ["m1"])
    svc.add_members(g["id"], ["m1"])
    assert len(svc.get_trends(g["id"])["snapshots"]) == 1  # type: ignore[index]


def test_remove_member_snapshots(seeded: DbHandle, svc: InsightGroupService) -> None:
    g = svc.create_group("G")
    svc.add_members(g["id"], ["m1", "m2"])
    res = svc.remove_member(g["id"], "m2")
    assert res is not None and res["snapshot"]["reason"] == "remove"
    assert res["aggregate"]["coverage"]["totalMembers"] == 1
    assert svc.remove_member(g["id"], "m2") is not None  # idempotent
    stored = json.loads(
        _scalar(seeded, "SELECT meeting_ids FROM insight_group_snapshots ORDER BY created_at DESC, rowid DESC")
    )
    assert stored == ["m1"]


def _scalar(handle: DbHandle, sql: str) -> Any:
    with handle.engine.connect() as conn:
        return conn.execute(text(sql)).scalar()


def test_manual_snapshot(seeded: DbHandle, svc: InsightGroupService) -> None:
    g = svc.create_group("G")
    svc.add_members(g["id"], ["m1"])
    snap = svc.snapshot_group(g["id"], "manual")
    assert snap is not None and snap["reason"] == "manual" and snap["memberCount"] == 1
    assert len(svc.get_trends(g["id"])["snapshots"]) == 2  # type: ignore[index]


# -- compare ----------------------------------------------------------------------


def test_compare_aligns_axes_and_includes_default(seeded: DbHandle, svc: InsightGroupService) -> None:
    a = svc.create_group("A")
    b = svc.create_group("B", axes=["warmth"])
    svc.add_members(a["id"], ["m1", "m2"])
    svc.add_members(b["id"], ["m_late", "m_bare"])
    out = svc.compare([a["id"], b["id"]])
    assert out["axes"] == ["hostility", "warmth"]
    by = {g["id"]: g for g in out["groups"]}
    assert [g["id"] for g in out["groups"]] == [a["id"], b["id"]]
    assert by[a["id"]]["name"] == "A" and by[a["id"]]["memberCount"] == 2
    assert by[a["id"]]["scores"]["warmth"] == pytest.approx(0.45)
    assert by[a["id"]]["scores"]["hostility"] == pytest.approx(0.2)
    assert by[b["id"]]["scores"] == {"warmth": pytest.approx(0.9)}  # missing axis stays absent
    assert by[b["id"]]["coverage"] == {"analyzedMembers": 1, "totalMembers": 2}
    assert by[a["id"]]["meetingIds"] == ["m1", "m2"]
    assert by[b["id"]]["meetingIds"] == ["m_late"]
    # default = whole corpus: mean of meeting means over analyzed meetings
    default = out["default"]
    assert default["scores"]["warmth"] == pytest.approx((0.3 + 0.6 + 0.9) / 3)
    assert default["scores"]["hostility"] == pytest.approx((0.2 + 0.1) / 2)
    assert default["coverage"] == {"analyzedMeetings": 3, "totalMeetings": 4}
    assert default["meetingIds"] == ["m1", "m2", "m_late"]


def test_compare_without_default(seeded: DbHandle, svc: InsightGroupService) -> None:
    a = svc.create_group("A")
    svc.add_members(a["id"], ["m1"])
    out = svc.compare([a["id"]], include_default=False)
    assert out["default"] is None
    assert out["axes"] == ["hostility", "warmth"]


def test_compare_empty_unknown_and_duplicate_groups(seeded: DbHandle, svc: InsightGroupService) -> None:
    empty = svc.create_group("Empty")
    a = svc.create_group("A")
    svc.add_members(a["id"], ["m1"])
    out = svc.compare([empty["id"], "ghost", a["id"], a["id"]])
    assert [g["id"] for g in out["groups"]] == [empty["id"], a["id"]]  # unknown skipped, duplicate collapsed
    assert out["groups"][0]["scores"] == {} and out["groups"][0]["coverage"] == {"analyzedMembers": 0, "totalMembers": 0}
    none = svc.compare([])
    assert none["groups"] == [] and none["default"] is not None
    assert none["axes"] == ["hostility", "warmth"]
    assert svc.compare([], include_default=False) == {"axes": [], "groups": [], "default": None}
