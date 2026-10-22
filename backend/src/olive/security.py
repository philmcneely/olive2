"""App-wide request guards: CSRF Origin check (writes) and opt-in Host allowlist (all methods).

Olive's ``/api/*`` has no authentication, so a malicious web page the user visits
could otherwise make their browser change data on a LAN or localhost Olive (CSRF),
or rebind its own DNS name to the server's address (DNS rebinding) to read data.

* **Origin check (always on, writes only).** For POST/PUT/PATCH/DELETE, a request
  that carries an ``Origin`` header is rejected with 403 unless that origin is
  allowed. This blocks cross-site writes (CSRF). Requests with NO ``Origin``
  header always pass: non-browser clients (the iOS Shortcut's ``POST /api/ingest``,
  curl, MCP clients, provider webhooks) never send one, and browsers always send
  one on cross-site mutations. Allowed origins: loopback hosts (``localhost``,
  ``127.0.0.1``, ``[::1]``) on any port, by design; any origin equal to the
  request's own ``Host`` (same-origin, which covers LAN-by-IP and reverse proxies
  that forward Host); the configured bind host (any port, unless it is a wildcard
  such as ``0.0.0.0``); and every origin in the comma-separated
  ``OLIVE_ALLOWED_ORIGINS`` (full ``scheme://host[:port]`` values).
* **Host allowlist (opt-in, ALL methods).** Set ``OLIVE_ALLOWED_HOSTS`` to a
  comma-separated list of BARE hostnames (``olive.lan``, ``192.168.0.50``; not
  ``scheme://host``; ports are ignored) and every HTTP request, GET/HEAD/OPTIONS
  included (SPA, ``/api/*`` reads, ``/mcp``, SSE streams), whose ``Host`` header is
  not in that list gets 403. Loopback names (any port) and the bind host are always
  allowed. This is what defeats DNS rebinding, for reads and writes alike. Left
  unset it is off, so LAN-by-IP access keeps working; a startup warning is logged
  when serving on a non-loopback address without it.

Settings are read per request so they follow the environment.
"""

from __future__ import annotations

import logging
import os
from urllib.parse import urlsplit

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

_log = logging.getLogger(__name__)
MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
WILDCARD_HOSTS = frozenset({"", "0.0.0.0", "::"})  # noqa: S104 - compared against, never bound


def _split_env(name: str) -> list[str]:
    return [p.strip().lower() for p in os.environ.get(name, "").split(",") if p.strip()]


def _hostname(netloc: str) -> str:
    """Lowercased hostname of a Host header / netloc, without port or IPv6 brackets."""
    try:
        return (urlsplit(f"//{netloc}").hostname or "").lower()
    except ValueError:
        return ""


def _normalize_origin(origin: str) -> str:
    parts = urlsplit(origin.strip().lower())
    return f"{parts.scheme}://{parts.netloc}" if parts.scheme and parts.netloc else ""


def warn_if_rebinding_unprotected(bind_host: str) -> None:
    """One-time startup warning: non-loopback bind without an OLIVE_ALLOWED_HOSTS allowlist."""
    host = bind_host.strip().lower().strip("[]")
    if host in LOOPBACK_HOSTS or _split_env("OLIVE_ALLOWED_HOSTS"):
        return
    _log.warning(
        "Serving on non-loopback address %r without OLIVE_ALLOWED_HOSTS; DNS-rebinding protection is off. "
        "Set OLIVE_ALLOWED_HOSTS to a comma-separated list of bare hostnames to enable it.",
        bind_host,
    )


class MutationGuardMiddleware:
    def __init__(self, app: ASGIApp, *, bind_host: str = "127.0.0.1") -> None:
        self.app = app
        self.bind_host = bind_host.strip().lower()

    def _origin_allowed(self, origin: str, host_header: str) -> bool:
        parts = urlsplit(origin)
        if parts.scheme not in ("http", "https"):
            return False
        try:
            hostname = (parts.hostname or "").lower()
        except ValueError:
            return False
        if not hostname:
            return False
        if hostname in LOOPBACK_HOSTS:
            return True
        if host_header and parts.netloc.lower() == host_header.strip().lower():
            return True  # same-origin
        if self.bind_host not in WILDCARD_HOSTS and hostname == self.bind_host.strip("[]"):
            return True
        return _normalize_origin(origin) in _split_env("OLIVE_ALLOWED_ORIGINS")

    def _host_allowed(self, host_header: str) -> bool:
        allowed = _split_env("OLIVE_ALLOWED_HOSTS")
        if not allowed:
            return True  # opt-in: off unless a list is configured
        hostname = _hostname(host_header)
        if hostname in LOOPBACK_HOSTS or hostname in {_hostname(a) or a for a in allowed}:
            return True
        return self.bind_host not in WILDCARD_HOSTS and hostname == self.bind_host.strip("[]")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        host = headers.get("host", "")
        # Host allowlist: every method (reads are DNS-rebinding's main prize).
        if not self._host_allowed(host):
            await JSONResponse({"error": "Host not allowed"}, status_code=403)(scope, receive, send)
            return
        # Origin check: cross-site writes only.
        if scope["method"] in MUTATING_METHODS:
            origin = headers.get("origin")
            if origin is not None and not self._origin_allowed(origin, host):
                await JSONResponse({"error": "Cross-origin request blocked"}, status_code=403)(
                    scope, receive, send
                )
                return
        await self.app(scope, receive, send)
