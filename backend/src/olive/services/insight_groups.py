"""Insight Groups: user-named collections of meetings with aggregated trend snapshots."""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Engine


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
