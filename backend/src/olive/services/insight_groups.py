"""Insight Groups: user-named collections of meetings with aggregated trend snapshots.

A group's analysis is a pure roll-up of the per-meeting rows the extraction stage already wrote
(``meeting_speaker_signals`` and ``meeting_moments``); nothing here calls an LLM. Each membership
change stores a snapshot of the aggregated axis scores so the group keeps a history of how it moved
as it grew. Every row carries ``meetingId`` and ``startMs`` for drill-down.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from ..db import DbHandle, get_db
from .insights.query import _parse_dimensions, _round


def _now_ms() -> int:
    return int(time.time() * 1000)


def _load_axes(raw: Any) -> list[str] | None:
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    return [str(a) for a in data] if isinstance(data, list) else None


def _clean_axes(axes: Iterable[str] | None) -> list[str] | None:
    if axes is None:
        return None
    seen: list[str] = []
    for a in axes:
        a = str(a).strip()
        if a and a not in seen:
            seen.append(a)
    return seen or None


class InsightGroupService:
    """CRUD, membership, aggregation and snapshots for insight groups."""

    def __init__(
        self,
        options: Mapping[str, Any] | None = None,
        *,
        db: Engine | DbHandle | None = None,
        config_dir: str | None = None,
    ) -> None:
        if isinstance(db, DbHandle):
            self.db: Engine = db.engine
        else:
            self.db = db or get_db()
        self.options = dict(options or {})
        self.config_dir = config_dir

    # -- groups -----------------------------------------------------------

    @staticmethod
    def _row_to_group(row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "id": row["id"],
            "name": row["name"],
            "description": row["description"],
            "axes": _load_axes(row["axes"]),
            "memberCount": int(row["member_count"]) if "member_count" in row else 0,
            "createdAt": int(row["created_at"]),
            "updatedAt": int(row["updated_at"]),
        }

    def _fetch(self, conn: Connection, group_id: str) -> dict[str, Any] | None:
        row = conn.execute(
            text(
                "SELECT g.*, (SELECT COUNT(*) FROM insight_group_members x WHERE x.group_id = g.id)"
                " AS member_count FROM insight_groups g WHERE g.id = :id"
            ),
            {"id": group_id},
        ).mappings().first()
        return self._row_to_group(row) if row else None

    def create_group(
        self, name: str, description: str | None = None, axes: Iterable[str] | None = None
    ) -> dict[str, Any]:
        name = (name or "").strip()
        if not name:
            raise ValueError("name is required")
        gid, now = uuid.uuid4().hex, _now_ms()
        cleaned = _clean_axes(axes)
        with self.db.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO insight_groups (id, name, description, axes, created_at, updated_at)"
                    " VALUES (:id, :n, :d, :a, :now, :now)"
                ),
                {"id": gid, "n": name, "d": description, "a": json.dumps(cleaned) if cleaned else None, "now": now},
            )
            group = self._fetch(conn, gid)
        assert group is not None
        return group

    def list_groups(self) -> list[dict[str, Any]]:
        with self.db.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT g.*, (SELECT COUNT(*) FROM insight_group_members x WHERE x.group_id = g.id)"
                    " AS member_count FROM insight_groups g ORDER BY g.updated_at DESC, g.created_at DESC, g.id"
                )
            ).mappings().all()
        return [self._row_to_group(r) for r in rows]

    def get_group(self, group_id: str) -> dict[str, Any] | None:
        with self.db.connect() as conn:
            group = self._fetch(conn, group_id)
            if group is None:
                return None
            members = conn.execute(
                text(
                    "SELECT mem.meeting_id, mem.added_at, m.title, m.start_time FROM insight_group_members mem"
                    " JOIN meetings m ON m.id = mem.meeting_id WHERE mem.group_id = :g"
                    " ORDER BY m.start_time, mem.meeting_id"
                ),
                {"g": group_id},
            ).mappings().all()
        group["members"] = [
            {
                "meetingId": r["meeting_id"],
                "title": r["title"],
                "startMs": int(r["start_time"]),
                "addedAt": int(r["added_at"]),
            }
            for r in members
        ]
        group["aggregate"] = self.aggregate_group(group_id)
        return group

    _UNSET: Any = object()

    def update_group(
        self,
        group_id: str,
        name: str | None = None,
        description: Any = _UNSET,
        axes: Any = _UNSET,
    ) -> dict[str, Any] | None:
        sets, params = ["updated_at = :now"], {"id": group_id, "now": _now_ms()}
        if name is not None:
            if not name.strip():
                raise ValueError("name must not be empty")
            sets.append("name = :n")
            params["n"] = name.strip()
        if description is not self._UNSET:
            sets.append("description = :d")
            params["d"] = description
        if axes is not self._UNSET:
            cleaned = _clean_axes(axes)
            sets.append("axes = :a")
            params["a"] = json.dumps(cleaned) if cleaned else None
        with self.db.begin() as conn:
            if conn.execute(text("SELECT 1 FROM insight_groups WHERE id = :id"), {"id": group_id}).first() is None:
                return None
            conn.execute(text(f"UPDATE insight_groups SET {', '.join(sets)} WHERE id = :id"), params)
            return self._fetch(conn, group_id)

    def delete_group(self, group_id: str) -> bool:
        return delete_group(self.db, group_id)

    # -- membership -------------------------------------------------------

    def _exists(self, group_id: str) -> bool:
        with self.db.connect() as conn:
            return conn.execute(text("SELECT 1 FROM insight_groups WHERE id = :id"), {"id": group_id}).first() is not None

    def exists(self, group_id: str) -> bool:
        return self._exists(group_id)

    def add_members(self, group_id: str, meeting_ids: Sequence[str]) -> dict[str, Any] | None:
        """Add meetings (duplicates and unknown ids are ignored); snapshot when membership changed."""
        if not self._exists(group_id):
            return None
        now, added = _now_ms(), 0
        with self.db.begin() as conn:
            for mid in dict.fromkeys(str(m) for m in meeting_ids):
                if conn.execute(text("SELECT 1 FROM meetings WHERE id = :m"), {"m": mid}).first() is None:
                    continue
                res = conn.execute(
                    text(
                        "INSERT OR IGNORE INTO insight_group_members (id, group_id, meeting_id, added_at)"
                        " VALUES (:id, :g, :m, :now)"
                    ),
                    {"id": uuid.uuid4().hex, "g": group_id, "m": mid, "now": now},
                )
                added += res.rowcount or 0
            if added:
                conn.execute(text("UPDATE insight_groups SET updated_at = :now WHERE id = :g"), {"now": now, "g": group_id})
        snapshot = self.snapshot_group(group_id, "append") if added else None
        return {"aggregate": self.aggregate_group(group_id), "snapshot": snapshot}

    def remove_member(self, group_id: str, meeting_id: str) -> dict[str, Any] | None:
        if not self._exists(group_id):
            return None
        with self.db.begin() as conn:
            res = conn.execute(
                text("DELETE FROM insight_group_members WHERE group_id = :g AND meeting_id = :m"),
                {"g": group_id, "m": meeting_id},
            )
            removed = res.rowcount or 0
            if removed:
                conn.execute(
                    text("UPDATE insight_groups SET updated_at = :now WHERE id = :g"),
                    {"now": _now_ms(), "g": group_id},
                )
        snapshot = self.snapshot_group(group_id, "remove") if removed else None
        return {"aggregate": self.aggregate_group(group_id), "snapshot": snapshot}

    # -- aggregation ------------------------------------------------------

    def aggregate_group(self, group_id: str) -> dict[str, Any] | None:
        """Mean-of-meeting-means per axis, moment counts, coverage and the per-member series."""
        with self.db.connect() as conn:
            group = self._fetch(conn, group_id)
            if group is None:
                return None
            members = conn.execute(
                text(
                    "SELECT m.id, m.title, m.start_time FROM insight_group_members mem"
                    " JOIN meetings m ON m.id = mem.meeting_id WHERE mem.group_id = :g"
                    " ORDER BY m.start_time, m.id"
                ),
                {"g": group_id},
            ).mappings().all()
            signals = conn.execute(
                text(
                    "SELECT s.meeting_id, s.dimensions FROM meeting_speaker_signals s"
                    " WHERE s.meeting_id IN (SELECT meeting_id FROM insight_group_members WHERE group_id = :g)"
                ),
                {"g": group_id},
            ).mappings().all()
            moments = conn.execute(
                text(
                    "SELECT meeting_id, valence, COUNT(*) AS n FROM meeting_moments"
                    " WHERE meeting_id IN (SELECT meeting_id FROM insight_group_members WHERE group_id = :g)"
                    " AND valence IN ('good', 'bad') GROUP BY meeting_id, valence"
                ),
                {"g": group_id},
            ).mappings().all()

        selected = group["axes"]
        per_meeting: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        has_signal: set[str] = set()
        for row in signals:
            has_signal.add(row["meeting_id"])
            for axis, val in _parse_dimensions(row["dimensions"]).items():
                if selected is None or axis in selected:
                    per_meeting[row["meeting_id"]][axis].append(val)
        counts: dict[str, dict[str, int]] = defaultdict(lambda: {"good": 0, "bad": 0})
        for row in moments:
            counts[row["meeting_id"]][row["valence"]] = int(row["n"])

        series: list[dict[str, Any]] = []
        by_axis: dict[str, list[float]] = defaultdict(list)
        totals = {"good": 0, "bad": 0}
        for m in members:
            scores = {
                axis: _round(math.fsum(vals) / len(vals)) for axis, vals in per_meeting.get(m["id"], {}).items()
            }
            for axis, val in scores.items():
                by_axis[axis].append(val)
            mc = counts.get(m["id"], {"good": 0, "bad": 0})
            totals["good"] += mc["good"]
            totals["bad"] += mc["bad"]
            series.append(
                {
                    "meetingId": m["id"],
                    "title": m["title"],
                    "startMs": int(m["start_time"]),
                    "axisScores": scores,
                    "moments": {"good": mc["good"], "bad": mc["bad"]},
                }
            )
        axes = {a: _round(math.fsum(v) / len(v)) for a, v in sorted(by_axis.items())}
        analyzed = sum(1 for m in members if m["id"] in has_signal)
        return {
            "axes": axes,
            "moments": totals,
            "coverage": {"analyzedMembers": analyzed, "totalMembers": len(members)},
            "series": series,
        }

    def _corpus_aggregate(self) -> dict[str, Any]:
        """Whole-corpus baseline: per-axis mean of per-meeting means over every analyzed meeting.

        The same roll-up ``aggregate_group`` applies to a group's members, over all meetings.
        """
        with self.db.connect() as conn:
            total = conn.execute(text("SELECT COUNT(*) FROM meetings")).scalar_one()
            signals = conn.execute(
                text(
                    "SELECT s.meeting_id, s.dimensions FROM meeting_speaker_signals s"
                    " JOIN meetings m ON m.id = s.meeting_id ORDER BY m.start_time, m.id"
                )
            ).mappings().all()
        per_meeting: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in signals:
            per_meeting[row["meeting_id"]]  # an analyzed meeting counts even when it has no numeric axis
            for axis, val in _parse_dimensions(row["dimensions"]).items():
                per_meeting[row["meeting_id"]][axis].append(val)
        by_axis: dict[str, list[float]] = defaultdict(list)
        for axes in per_meeting.values():
            for axis, vals in axes.items():
                by_axis[axis].append(_round(math.fsum(vals) / len(vals)))
        return {
            "scores": {a: _round(math.fsum(v) / len(v)) for a, v in sorted(by_axis.items())},
            "coverage": {"analyzedMeetings": len(per_meeting), "totalMeetings": int(total)},
            "meetingIds": list(per_meeting),
        }

    def compare(self, group_ids: Sequence[str], include_default: bool = True) -> dict[str, Any]:
        """Per-axis scores of several groups side by side, optionally against the whole corpus.

        Axes are the union across the compared groups and the default; a score is absent where a group
        has no data for that axis. Unknown ids are skipped and repeats collapse (the API 404s first).
        """
        groups: list[dict[str, Any]] = []
        for gid in dict.fromkeys(str(g) for g in group_ids):
            with self.db.connect() as conn:
                head = self._fetch(conn, gid)
            agg = self.aggregate_group(gid) if head else None
            if head is None or agg is None:
                continue
            groups.append(
                {
                    "id": head["id"],
                    "name": head["name"],
                    "memberCount": head["memberCount"],
                    "scores": agg["axes"],
                    "coverage": agg["coverage"],
                    "meetingIds": [p["meetingId"] for p in agg["series"] if p["axisScores"]],
                }
            )
        default = self._corpus_aggregate() if include_default else None
        axes = sorted({a for g in groups for a in g["scores"]} | set(default["scores"] if default else ()))
        return {"axes": axes, "groups": groups, "default": default}

    def snapshot_group(self, group_id: str, reason: str) -> dict[str, Any] | None:
        agg = self.aggregate_group(group_id)
        if agg is None:
            return None
        ids = [p["meetingId"] for p in agg["series"]]
        now = _now_ms()
        with self.db.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO insight_group_snapshots (id, group_id, created_at, member_count, axes, meeting_ids, reason)"
                    " VALUES (:id, :g, :now, :n, :a, :ids, :r)"
                ),
                {
                    "id": uuid.uuid4().hex,
                    "g": group_id,
                    "now": now,
                    "n": len(ids),
                    "a": json.dumps(agg["axes"]),
                    "ids": json.dumps(ids),
                    "r": reason,
                },
            )
        return {"createdAt": now, "memberCount": len(ids), "axes": agg["axes"], "reason": reason}

    def get_trends(self, group_id: str) -> dict[str, Any] | None:
        agg = self.aggregate_group(group_id)
        if agg is None:
            return None
        with self.db.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT created_at, member_count, axes, reason FROM insight_group_snapshots"
                    " WHERE group_id = :g ORDER BY created_at, rowid"
                ),
                {"g": group_id},
            ).mappings().all()
        snapshots = [
            {
                "createdAt": int(r["created_at"]),
                "memberCount": int(r["member_count"]),
                "axes": json.loads(r["axes"]) if r["axes"] else {},
                "reason": r["reason"],
            }
            for r in rows
        ]
        return {"snapshots": snapshots, "memberSeries": agg["series"]}


def delete_group(db: Engine, group_id: str) -> bool:
    """Delete a group with its snapshots and memberships in one transaction.

    Returns False when the group does not exist. Meetings are never touched.
    """
    with db.begin() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM insight_groups WHERE id = :id"), {"id": group_id}
        ).first()
        if not exists:
            return False
        for tbl in ("insight_group_snapshots", "insight_group_members"):
            conn.execute(text(f"DELETE FROM {tbl} WHERE group_id = :id"), {"id": group_id})
        conn.execute(text("DELETE FROM insight_groups WHERE id = :id"), {"id": group_id})
    return True
