"""Export & Share HTTP routes.

Kept out of ``app.py`` so the feature stays additive and cleanly removable:
delete this file, ``services/export.py``, ``services/sharing.py`` and the one
``register_export_share_routes`` call in ``create_app``.

All routes are ``include_in_schema=False`` (like ``/mcp``) so the generated
OpenAPI document and ``frontend/src/api/schema.d.ts`` do not drift.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from fastapi import Body, FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response
from sqlalchemy import text
from sqlalchemy.engine import Engine

from .services import export as export_svc
from .services.export import ExportError
from .services.sharing import (
    SHARE_RESPONSE_HEADERS,
    ShareDisabledError,
    ShareLink,
    ShareService,
    render_share_html,
    share_links_enabled,
)

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


def _flag(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    v = value.strip().lower()
    if v in _TRUTHY:
        return True
    if v in _FALSY:
        return False
    return default


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


_EXPORT_STATUS = {"not_found": 404, "no_transcript": 409, "bad_format": 400}


_NOT_FOUND_HTML = (
    '<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Not found</title></head>'
    "<body><p>This link is not available.</p></body></html>"
)


def _link_json(link: ShareLink) -> dict[str, Any]:
    return {
        "id": link.id,
        "meetingId": link.meeting_id,
        "label": link.label,
        "includeSummary": link.include_summary,
        "createdAt": link.created_at,
        "expiresAt": link.expires_at,
        "revokedAt": link.revoked_at,
        "lastAccessedAt": link.last_accessed_at,
        "accessCount": link.access_count,
        "status": link.status,
    }


def _public_not_found() -> Response:
    """The single response every public failure mode returns (no existence/expiry oracle)."""
    return HTMLResponse(_NOT_FOUND_HTML, status_code=404, headers=SHARE_RESPONSE_HEADERS)


def _parse_create_body(payload: Any, default_ttl_hours: int) -> tuple[float | None, str | None, bool]:
    """Validate the POST body; raises ``ValueError`` with a user-facing message."""
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError("body must be a JSON object")
    if "expiresInHours" in payload:
        hours = payload["expiresInHours"]
        if hours is not None and (
            isinstance(hours, bool)
            or not isinstance(hours, int | float)
            or not math.isfinite(hours)
            or hours <= 0
        ):
            raise ValueError("expiresInHours must be a positive number or null")
    else:
        hours = default_ttl_hours if default_ttl_hours > 0 else None
    label = payload.get("label")
    if label is not None and not isinstance(label, str):
        raise ValueError("label must be a string")
    include_summary = payload.get("includeSummary", False)
    if not isinstance(include_summary, bool):
        raise ValueError("includeSummary must be a boolean")
    return hours, label, include_summary


def register_export_share_routes(
    app: FastAPI,
    *,
    db: Engine,
    meetings_dir: str,
    get_config: Callable[[], Any] | None = None,
    now_ms: Callable[[], int] | None = None,
) -> None:
    def config() -> Any:
        return get_config() if get_config else None

    def enabled() -> bool:
        return share_links_enabled(config())

    share_kwargs: dict[str, Any] = {"is_enabled": enabled}
    if now_ms is not None:
        share_kwargs["now_ms"] = now_ms
    shares = ShareService(db, meetings_dir, **share_kwargs)

    @app.get("/api/meetings/{meeting_id}/export", include_in_schema=False)
    def export_meeting_route(
        meeting_id: str,
        format: str = "txt",
        summary: str | None = None,
        timestamps: str | None = None,
    ) -> Response:
        if format not in export_svc.EXPORT_MEDIA_TYPES:
            return _error(400, f"Unsupported format: {format[:20]!r}")
        fmt: export_svc.ExportFormat = format  # type: ignore[assignment]
        include_summary = _flag(summary, False)
        try:
            doc = export_svc.load_export_doc(db, meetings_dir, meeting_id, include_summary=include_summary)
            body = export_svc.render(doc, fmt, timestamps=_flag(timestamps, True))
        except ExportError as exc:
            return _error(_EXPORT_STATUS[exc.code], str(exc))
        return Response(
            content=body,
            media_type=export_svc.EXPORT_MEDIA_TYPES[fmt],
            headers={
                "Content-Disposition": export_svc.content_disposition(export_svc.export_filename(doc, fmt)),
                "X-Content-Type-Options": "nosniff",
            },
        )

    # -- share-link management (local UI; same no-auth posture as the rest of /api) ----------

    @app.post("/api/meetings/{meeting_id}/share", status_code=201, include_in_schema=False)
    def create_share_route(meeting_id: str, payload: Any = Body(default=None)) -> Response:
        if not enabled():
            return _error(403, "Share links are disabled")
        cfg = config()
        try:
            hours, label, include_summary = _parse_create_body(
                payload, int(getattr(cfg, "shareDefaultTtlHours", 168))
            )
            link, token = shares.create(
                meeting_id, expires_in_hours=hours, label=label, include_summary=include_summary
            )
        except ValueError as exc:
            return _error(400, str(exc))
        except ShareDisabledError:
            return _error(403, "Share links are disabled")
        except ExportError as exc:
            return _error(_EXPORT_STATUS[exc.code], str(exc))
        return JSONResponse(
            {"link": _link_json(link), "token": token, "path": f"/share/{token}"},
            status_code=201,
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/meetings/{meeting_id}/share", include_in_schema=False)
    def list_share_route(meeting_id: str) -> Response:
        with db.connect() as conn:
            if not conn.execute(text("SELECT 1 FROM meetings WHERE id = :id"), {"id": meeting_id}).first():
                return _error(404, "Meeting not found")
        return JSONResponse(
            {
                "enabled": enabled(),
                "links": [_link_json(link) for link in shares.list_for_meeting(meeting_id)],
            }
        )

    @app.delete("/api/meetings/{meeting_id}/share/{link_id}", include_in_schema=False)
    def revoke_share_route(meeting_id: str, link_id: str) -> Response:
        if not shares.revoke(meeting_id, link_id):
            return _error(404, "Share link not found")
        return JSONResponse({"ok": True})

    # -- public, token-addressed (registered before the SPA catch-all by the caller) ---------

    @app.get("/share/{token}", include_in_schema=False)
    def public_share_page(token: str) -> Response:
        link = shares.resolve(token)
        if link is None:
            return _public_not_found()
        try:
            view = shares.public_view(link)
        except ExportError:
            return _public_not_found()
        return HTMLResponse(render_share_html(view, token), headers=SHARE_RESPONSE_HEADERS)

    @app.get("/share/{token}/download", include_in_schema=False)
    def public_share_download(token: str, format: str = "txt", timestamps: str | None = None) -> Response:
        link = shares.resolve(token)
        if link is None:
            return _public_not_found()
        if format not in export_svc.EXPORT_MEDIA_TYPES:
            return _error(400, "Unsupported format")
        fmt: export_svc.ExportFormat = format  # type: ignore[assignment]
        try:
            doc = shares.export_doc(link)
            payload = export_svc.render(doc, fmt, timestamps=_flag(timestamps, True))
        except ExportError:
            return _public_not_found()
        return Response(
            content=payload,
            media_type=export_svc.EXPORT_MEDIA_TYPES[fmt],
            headers={
                **SHARE_RESPONSE_HEADERS,
                "Content-Disposition": export_svc.content_disposition(export_svc.export_filename(doc, fmt)),
            },
        )
