"""Export & Share HTTP routes.

Kept out of ``app.py`` so the feature stays additive and cleanly removable:
delete this file, ``services/export.py``, ``services/sharing.py`` and the one
``register_export_share_routes`` call in ``create_app``.

All routes are ``include_in_schema=False`` (like ``/mcp``) so the generated
OpenAPI document and ``frontend/src/api/schema.d.ts`` do not drift.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from sqlalchemy.engine import Engine

from .services import export as export_svc
from .services.export import ExportError

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


def register_export_share_routes(app: FastAPI, *, db: Engine, meetings_dir: str, **_: Any) -> None:
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
