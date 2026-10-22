"""Main server entrypoint for Olive (port of packages/server/src/index.ts)."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import uvicorn
from uvicorn.config import LOGGING_CONFIG

from .app import create_app
from .logging_db import logger
from .ml.model_store import ModelStorageError, download_local_models, ensure_models_volume
from .services.sharing import RedactShareTokens
from .settings import get_effective_hf_token, load_app_config, load_runtime_config, resolve_paths


def get_uvicorn_log_config() -> dict[str, Any]:
    cfg = copy.deepcopy(LOGGING_CONFIG)
    cfg["formatters"]["default"]["fmt"] = "%(asctime)s %(levelprefix)s %(message)s"
    cfg["formatters"]["default"]["datefmt"] = "%Y-%m-%dT%H:%M:%S%z"
    cfg["formatters"]["access"]["fmt"] = (
        '%(asctime)s %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'
    )
    cfg["formatters"]["access"]["datefmt"] = "%Y-%m-%dT%H:%M:%S%z"
    # Share links are bearer URLs: never write the raw token to the access log.
    cfg.setdefault("filters", {})["redact_share_tokens"] = {"()": RedactShareTokens}
    cfg["handlers"]["access"]["filters"] = ["redact_share_tokens"]
    return cfg


def build_default_app(*, start_plaud_poller: bool = False):
    runtime_config = load_runtime_config()
    app_config = load_app_config(runtime_config.paths)

    hf_models_dir = Path(runtime_config.paths.models_dir) / "huggingface"
    os.environ.setdefault("HF_HOME", str(hf_models_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_models_dir))

    default_web_root = Path(__file__).resolve().parents[3] / "frontend" / "dist"
    web_root = os.environ.get("OLIVE_WEB_ROOT") or str(default_web_root)

    return create_app(
        {
            "webRoot": web_root,
            "meetingsDir": runtime_config.paths.meetings_dir,
            "pollIntervalMinutes": app_config.pollIntervalMinutes,
            "defaultTranscriptionProvider": app_config.transcriptionProvider,
            "bindHost": runtime_config.bind_host,
            "startPlaudPoller": start_plaud_poller,
        }
    )


def prepare_container_models() -> None:
    """In the official Docker image, require a mounted models volume and fetch missing models before serving."""
    if os.environ.get("OLIVE_DOCKER") != "1":
        return
    try:
        ensure_models_volume(Path(resolve_paths().models_dir))
        download_local_models(get_effective_hf_token())
    except ModelStorageError as exc:
        raise SystemExit(f"olive: cannot start: {exc}") from exc


def run_server() -> None:
    runtime_config = load_runtime_config()
    app_config = load_app_config(runtime_config.paths)
    prepare_container_models()
    app = build_default_app(start_plaud_poller=True)

    logger.info(
        "Olive server listening",
        {
            "host": runtime_config.bind_host,
            "port": runtime_config.bind_port,
            "pollIntervalMinutes": app_config.pollIntervalMinutes,
        },
    )
    uvicorn.run(
        app,
        host=runtime_config.bind_host,
        port=runtime_config.bind_port,
        log_config=get_uvicorn_log_config(),
        timeout_graceful_shutdown=2,
    )


_app: Any | None = None


def get_app():
    """Lazily build the ASGI app so container model preflight runs before app construction."""
    global _app
    if _app is None:
        prepare_container_models()
        _app = build_default_app(start_plaud_poller=False)
    return _app


def __getattr__(name: str) -> Any:
    # Keeps `uvicorn olive.main:app` working while deferring app construction
    # (and its config requirements) until the attribute is actually accessed.
    if name == "app":
        return get_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    run_server()
